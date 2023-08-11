from typing import Optional
import torch
import torch.nn as nn
import pytorch_lightning as pl
from .positional_encoding import standard_PositionalEncoding as PositionalEncoding


class TransformerPredictor(nn.Module):

    def __init__(self, input_dim: int = 10, model_dim: int = 32, num_classes: int = 10, num_heads: int = 1,
                 dropout: float = 0.0, input_dropout: float = 0.0,
                 **kwargs):
        """The predictor network based on single transformer encoder layer.

        Inputs:
            input_dim - Hidden dimensionality of the input
            model_dim - Hidden dimensionality to use inside the Transformer
            num_classes - Number of classes to predict per sequence element
            num_heads - Number of heads to use in the Multi-Head Attention blocks
            dropout - Dropout to apply inside the model
            input_dropout - Dropout to apply on the input features
        """
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.dropout = dropout
        self.input_dropout = input_dropout

        # Input dim -> Model dim
        self.input_net = nn.Sequential(
            nn.Dropout(self.input_dropout),
            nn.Linear(self.input_dim, self.model_dim)
        )
        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(d_model=self.model_dim)
        # Transformer
        self.transformer = torch.nn.TransformerEncoderLayer(
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

    def forward(self, x, add_positional_encoding: bool = True) -> torch.Tensor:
        """
        Inputs:
            x - Input features of shape [Batch, SeqLen, input_dim]
            add_positional_encoding - If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        Returns:
            Output features of shape [Batch, SeqLen, input_dim]
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x)
        x = self.output_net(x)
        return x


if __name__ == "__main__":
    tf = TransformerPredictor(input_dim=10,
                              model_dim=32,
                              num_heads=1,
                              num_classes=10,
                              num_layers=1,
                              dropout=0.0,
                              lr=5e-4,
                              warmup=50)
    input = torch.randn([2, 17, 10])
    print(tf(input).shape)
