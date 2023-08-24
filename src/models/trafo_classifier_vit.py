from typing import Optional, Any, Union, Callable
import torch
import torch.nn as nn
from src.models.positional_encoding import StandardPositionalEncoding as PositionalEncoding
from src.models.transformer_implementations import TransformerEncoder


class TransformerClassifierVit(nn.Module):

    def __init__(self, input_dim: int = 10, model_dim: int = 32, num_classes: int = 10, num_heads: int = 2,
                 dropout: float = 0.0, input_dropout: float = 0.0, add_positional_encoding: bool = True,
                 num_layers: int = 2,
                 **kwargs):
        """The ViT based Classifier.

        Inputs:
            input_dim - Hidden dimensionality of the input
            model_dim - Hidden dimensionality to use inside the Transformer
            num_classes - Number of classes to predict per sequence element
            num_heads - Number of heads to use in the Multi-Head Attention blocks
            dropout - Dropout to apply inside the model
            input_dropout - Dropout to apply on the input features
            add_positional_encoding - if positional encoding added
            num_layers - number of attention layers
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
        self.cls = nn.Parameter(torch.randn(1, 1, model_dim))

        # Input dim -> Model dim
        self.input_net = nn.Sequential(
            nn.Dropout(self.input_dropout),
            nn.Linear(self.input_dim, self.model_dim)
        )
        # Positional encoding for sequences
        if add_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model=self.model_dim)
        self.transformer_encoder = TransformerEncoder(token_dim=self.model_dim,
                                                      num_blocks=self.num_layers,
                                                      num_heads=self.num_heads,
                                                      dropout=self.dropout,
                                                      batch_first=True)
        # Output classifier per sequence element
        self.output_net = nn.Sequential(nn.Linear(self.model_dim, self.model_dim),
                                        nn.LayerNorm(self.model_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(self.dropout),
                                        nn.Linear(self.model_dim, self.num_classes))

    def forward(self, x: torch.Tensor) -> [torch.Tensor, list]:
        """
        Inputs:
            x - Input features of shape [Batch, SeqLen, input_dim]
        Returns:
            x - Output features of shape [Batch, SeqLen, input_dim]
            attn_map - list of attention maps of different with shape
                        [ num_layers x tensor(batch_size, num_heads, seq_len, seq_len) ]
        """
        cls = self.cls.expand(x.shape[0], self.cls.shape[1], self.cls.shape[2])
        x = torch.cat([cls, x], dim=1)
        x = self.input_net(x)
        if self.add_positional_encoding:
            x = self.positional_encoding(x)
        x, attn_maps = self.transformer_encoder(x)
        x = x[:, 0]
        x = self.output_net(x)
        return x, attn_maps
