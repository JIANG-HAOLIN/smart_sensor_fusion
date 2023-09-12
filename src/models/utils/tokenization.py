import torch
import torch.nn as nn
from src.models.utils.to_patches import Img2Patches
from typing import Optional


class VanillaTokenization(nn.Module):
    """The vanilla way of tokenization for 2D input tensor"""

    def __init__(self, channel_size: int = 3, model_dim: int = 32,
                 patch_size: tuple = (4, 4), input_size: Optional[tuple] = None,
                 **kwargs):
        """
        Inputs:
            channel_size - Hidden dimensionality of the input
            model_dim - Hidden dimensionality to use inside the Transformer
            num_classes - Number of classes to predict per sequence element
        """
        super().__init__()
        self.model_dim = model_dim

        # convert the input image tensor to patches
        self.to_patches = Img2Patches(input_size, patch_size)
        # Input dim -> Model dim
        patch_dim = patch_size[0] * patch_size[1] * channel_size
        self.input_emb = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

    def forward(self, x: torch.Tensor) -> [torch.Tensor, list]:
        """
        Inputs:
            x - Input img tensor of shape [batch size, channel size, height, width]
        Returns:
            x - tokenized input of shape [Batch, SeqLen, model_dim]
        """
        x = self.to_patches(x)
        x = self.input_emb(x)
        return x
