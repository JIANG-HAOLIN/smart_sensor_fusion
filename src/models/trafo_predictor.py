from typing import Optional, Any, Union, Callable
import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.models.positional_encoding import StandardPositionalEncoding as PositionalEncoding


class TransformerPredictor(nn.Module):

    def __init__(self, input_dim: int = 10, model_dim: int = 32, num_classes: int = 10, num_heads: int = 1,
                 dropout: float = 0.0, input_dropout: float = 0.0, add_positional_encoding: bool = True,
                 **kwargs):
        """The predictor network based on single transformer encoder layer.

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

        # Input dim -> Model dim
        self.input_net = nn.Sequential(
            nn.Dropout(self.input_dropout),
            nn.Linear(self.input_dim, self.model_dim)
        )
        # Positional encoding for sequences
        if add_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model=self.model_dim)
        # Transformer

        self.transformer = TrafoEncoderLayer(
            d_model=self.model_dim,
            dim_feedforward=2 * self.model_dim,
            nhead=self.num_heads,
            dropout=self.dropout,
            batch_first=True)

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

        hook = AttentionMapHook()
        submodule = self.transformer.self_attn
        hook_handle = submodule.register_forward_hook(hook.forward_hook_fn)
        x = self.transformer(x)
        hook_handle.remove()

        x = self.output_net(x)
        return x, hook.attention_map


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




