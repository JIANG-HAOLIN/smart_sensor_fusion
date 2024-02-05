import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ClassificationHead(nn.Module):
    """Head for downstream classification task"""

    def __init__(self,
                 model_dim: int = 256,
                 num_classes: int = 10,
                 dropout: float = 0.):
        """
        Args:
            model_dim: the model dim for linear layer
            num_classes: dim of output linear layer
            dropout: dropout rate, normally should be 0
        """
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(model_dim, 2 * model_dim),
                                        nn.GELU(),
                                        nn.LayerNorm(2 * model_dim),
                                        nn.Dropout(dropout),
                                        nn.Linear(2 * model_dim, num_classes))

    def forward(self, x):
        """

        Args:
            x: with shape [..., model_dim]

        Returns: shape [..., num_classes]

        """
        return self.classifier(x)


class MLPHead(nn.Module):
    """Head based on 2 layer mlp"""

    def __init__(self,
                 in_dim: int = 256,
                 hidden_dim: Optional[int] = None,
                 out_dim: int = 10,
                 dropout: float = 0.,
                 norm: str = "layer"):
        """
        Args:
            in_dim: the model dim for linear layer
            out_dim: dim of output linear layer
            dropout: dropout rate, normally should be 0
        """
        super().__init__()
        hidden_dim = 2 * in_dim if hidden_dim is None else hidden_dim
        if norm == "layer":
            norm = nn.LayerNorm(hidden_dim)
        elif norm == "batch":
            from src.models.utils.helpers import MyPermute
            norm = nn.Sequential(MyPermute([0, 2, 1]),
                                 nn.BatchNorm1d(momentum=0.9, eps=1e-5, num_features=hidden_dim),
                                 MyPermute([0, 2, 1]), )

        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                 nn.GELU(),
                                 norm,
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        """

        Args:
            x: with shape [..., model_dim]

        Returns: shape [..., num_classes]

        """
        return self.mlp(x)
