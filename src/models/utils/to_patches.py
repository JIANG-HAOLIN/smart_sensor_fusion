import torch
from einops.layers.torch import Rearrange


class Img2Patches(torch.nn.Module):
    """Convert an image tensor to patches"""

    def __init__(self, patch_size: tuple = (4, 4)):
        """
        Args:
            patch_size: should have shape [patch_h, patch_w]
        """
        super().__init__()
        self.patch_h, self.patch_w = patch_size
        self.to_patches = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_h, p2=self.patch_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input img tensor of shape [batch size, channel size, height, width]

        Returns: patches of shape [batch size, num of patches, patch dim]

        """
        assert x.shape[2] % self.patch_h == 0 and x.shape[3] % self.patch_w == 0
        return self.to_patches(x)


def img_2_patches(x: torch.Tensor, patch_size: tuple = (4, 4)) -> torch.Tensor:
    """
    Args:
        x: input img tensor of shape [batch size, channel size, height, width]
        patch_size: should have shape [patch_h, patch_w]

    Returns:patches of shape [batch size, num of patches, patch dim]

    """
    patch_h, patch_w = patch_size
    assert x.shape[2] % patch_h == 0 and x.shape[3] % patch_w == 0
    return x.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
