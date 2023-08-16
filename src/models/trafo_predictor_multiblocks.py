from typing import Optional, Any, Union, Callable
import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.models.positional_encoding import StandardPositionalEncoding as PositionalEncoding


class TransformerPredictor(nn.Module):

    def __init__(self, input_dim: int = 10, model_dim: int = 32, num_classes: int = 10, num_heads: int = 1,
                 dropout: float = 0.0, input_dropout: float = 0.0, add_positional_encoding: bool = True,
                 num_layers: int = 2,
                 **kwargs):
        """The predictor network based on multiple transformer encoder layers.

        Inputs:
            input_dim - Hidden dimensionality of the input
            model_dim - Hidden dimensionality to use inside the Transformer
            num_classes - Number of classes to predict per sequence element
            num_heads - Number of heads to use in the Multi-Head Attention blocks
            dropout - Dropout to apply inside the model
            input_dropout - Dropout to apply on the input features
            add_positional_encoding - if positional encoding added
        """
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.add_positional_encoding = add_positional_encoding
        self.num_layers = num_layers

        # Input dim -> Model dim
        self.input_net = nn.Sequential(
            nn.Dropout(self.input_dropout),
            nn.Linear(self.input_dim, self.model_dim)
        )
        # Positional encoding for sequences
        if add_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model=self.model_dim)

        self.transformer_layer = TrafoEncoderLayer(
            d_model=self.model_dim,
            dim_feedforward=2 * self.model_dim,
            nhead=self.num_heads,
            dropout=self.dropout,
            batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=self.num_layers)

        # Output classifier per sequence element
        self.output_net = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.LayerNorm(self.model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.model_dim, self.num_classes)
        )

    def forward(self, x) -> [torch.Tensor, torch.Tensor]:
        """
        Inputs:
            x - Input features of shape [Batch, SeqLen, input_dim]
        Returns:
            Output features of shape [Batch, SeqLen, input_dim]
        """
        x = self.input_net(x)
        if self.add_positional_encoding:
            x = self.positional_encoding(x)
        x2 = x

        hook1 = AttentionMapHook()
        hook_handles = []
        for layer in self.transformer_encoder.layers:
            hook_handles.append(layer.self_attn.register_forward_hook(hook1.forward_hook_fn))
        x = self.transformer_encoder(x)
        for handle in hook_handles:
            handle.remove()

        ## this approach can output the same result but will cause memory leak during training,
        ## I am not sure how the hook handle the memory so I'll leave it here for further discussion.
        # hook2 = AttentionMapHook()
        # for layer in self.transformer_encoder.layers:
        #     handle = layer.self_attn.register_forward_hook(hook2.forward_hook_fn)
        # x2 = self.transformer_encoder(x2)
        # handle.remove()

        x = self.output_net(x)
        return x, hook1.attention_map


class AttentionMapHook:
    """ Forward hook for getting attention map from nn.TransformerEncoderLayer()"""

    def __init__(self):
        self.attention_map = []

    def forward_hook_fn(self, module, input, output):
        """Get output from MultiHeadAttention layer, which outputs: [layer output, attention map]"""
        self.attention_map.append(output[1])


class TrafoEncoderLayer(torch.nn.TransformerEncoderLayer):
    """Override the _sa_block func of nn.TransformerEncoderLayer to change input arg "need_weights" """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = torch.nn.functional.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False, device=None,
                 dtype=None) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout,
                         activation, layer_norm_eps, batch_first,
                         norm_first, device, dtype)

    def _sa_block(self,
                  x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor],
                  key_padding_mask: Optional[torch.Tensor],
                  is_causal: bool = False) -> torch.Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True, is_causal=is_causal)[0]
        return self.dropout1(x)
