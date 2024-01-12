import torch
import torchvision
from einops.layers.torch import Rearrange
from einops import rearrange
from typing import Optional
import logging


class Seq2Patches(torch.nn.Module):
    """convert an 1d sequence to patches"""
    def __init__(self, patch_size: int = 4, step_size: Optional[int] = None):
        super().__init__()
        self.patch_size = patch_size
        self.step_size = step_size

    def forward(self, x: torch.Tensor):
        l = x.shape[2]
        if not ((l % self.patch_size) == 0):
            crop_l = (l // self.patch_size) * self.patch_size
            # print(f"input can not be exactly divided! inputs are linearly resized to length {crop_l}")
            x = torch.nn.functional.interpolate(x, crop_l, mode='linear')
        assert x.shape[2] % self.patch_size == 0
        x = x.unfold(2, self.patch_size, self.patch_size if self.step_size is None else self.step_size).permute(0, 2, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        return x


class Img2Patches(torch.nn.Module):
    """Convert an image tensor to patches"""

    def __init__(self, input_size: Optional[tuple] = None,
                 patch_size: tuple = (4, 4), ):
        """
        Args:
            patch_size: should have shape [patch_h, patch_w]
        """
        super().__init__()
        self.patch_h, self.patch_w = patch_size
        self.to_patches = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_h, p2=self.patch_w)
        self.rand_crop = None
        self.num_patches = None
        if input_size is not None:
            if not (input_size[0] % self.patch_h == 0 and input_size[1] % self.patch_w == 0):
                crop_h = int(input_size[0] / self.patch_h) * self.patch_h
                crop_w = int(input_size[1] / self.patch_w) * self.patch_w
                print(f"input can not be exactly patchified! input {input_size} "
                      f"has to be cropped to size ({crop_h, crop_w})")
                self.rand_crop = torchvision.transforms.RandomCrop(size=[crop_h, crop_w])
                self.num_patches = (crop_h // self.patch_h) * (crop_w // self.patch_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input img tensor of shape [batch size, channel size, height, width]

        Returns: patches of shape [batch size, num of patches, patch dim]

        """
        if self.rand_crop is not None:
            x = self.rand_crop(x)
        return self.to_patches(x)


class Video2Patches(torch.nn.Module):
    """Convert a video tensor with shape [B, N, C, H, W] to patches"""

    def __init__(self, input_size: Optional[tuple] = None,
                 patch_size: tuple = (1, 4, 4), ):
        """
        Args:
            patch_size: should have shape [patch_h, patch_w]
        """
        super().__init__()
        n, h, w = input_size
        patch_n, patch_h, patch_w = patch_size
        self.to_patches = Rearrange('b (n p3) c (h p1) (w p2) -> b n (h w) (p1 p2 p3 c)',
                                    p1=patch_h,
                                    p2=patch_w,
                                    p3=patch_n,
                                    )
        self.rand_crop = None
        self.num_patches = None
        if input_size is not None:
            assert n % patch_n == 0, "make sure the video can be exactly divided along time axis"
            if not (h % patch_h == 0 and w % patch_w == 0):
                crop_h = int(h / patch_h) * patch_h
                crop_w = int(w / patch_w) * patch_w
                print(f"input can not be exactly patchified! input {input_size}"
                      f" has to be cropped to size ({n, crop_h, crop_w})")
                self.rand_crop = torchvision.transforms.RandomCrop(size=[crop_h, crop_w])
                self.num_patches = (crop_h // patch_h) * (crop_w // patch_w) * (n // patch_n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input img tensor of shape [batch size, channel size, height, width]

        Returns: patches of shape [batch size, num of patches, patch dim]

        """
        if self.rand_crop is not None:
            x = self.rand_crop(x)
        return self.to_patches(x)


def seq_2_patches(x: torch.Tensor, patch_size: int = 4, step_size: Optional[int] = None) -> torch.Tensor:
    """
    Args:
        x: input tensor of shape [batch size, channel size, length]
        patch_size: window size
        step_size: step size of moving window

    Returns:patches of shape [batch size, num of patches, patch_h, patch_w]

    """
    l = x.shape[2]
    if not ((l % patch_size) == 0):
        crop_l = (l // patch_size) * patch_size
        # print(f"input can not be exactly divided! inputs are linearly resized to length {crop_l}")
        x = torch.nn.functional.interpolate(x, crop_l, mode='linear')
    assert x.shape[2] % patch_size == 0
    x = x.unfold(2, patch_size, patch_size if step_size is None else step_size).permute(0, 2, 1, 3)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    return x


def img_2_patches(x: torch.Tensor, patch_size: tuple = (4, 4)) -> torch.Tensor:
    """
    Args:
        x: input img tensor of shape [batch size, channel size, height, width]
        patch_size: should have shape [patch_h, patch_w]

    Returns:patches of shape [batch size, num of patches, dim]

    """
    h, w = x.shape[2], x.shape[3]
    patch_h, patch_w = patch_size
    if not ((h % patch_h) == 0 and (w % patch_w) == 0):
        crop_h = (h // patch_h) * patch_h
        crop_w = (w // patch_w) * patch_w
        print(f"input can not be exactly divided! inputs are bilinearly resized to ({crop_h, crop_w})")
        x = torch.nn.functional.interpolate(x, [crop_h, crop_w], mode='bilinear')
    assert x.shape[2] % patch_h == 0 and x.shape[3] % patch_w == 0
    x = rearrange(x, 'b c (h h1) (w w1) -> b (h w) (h1 w1 c)', h1=patch_h, w1=patch_w)
    return x


def img_2_patches_unfold(x: torch.Tensor, patch_size: tuple = (4, 4)) -> torch.Tensor:
    """
    Args:
        x: input img tensor of shape [batch size, channel size, height, width]
        patch_size: should have shape [patch_h, patch_w]

    Returns:patches of shape [batch size, num of patches along h, num of patches along w, patch_h, patch_w]

    """
    h, w = x.shape[2], x.shape[3]
    patch_h, patch_w = patch_size
    if not ((h % patch_h) == 0 and (w % patch_w) == 0):
        crop_h = (h // patch_h) * patch_h
        crop_w = (w // patch_w) * patch_w
        print(f"input can not be exactly divided! inputs are bilinearly resized to ({crop_h, crop_w})")
        x = torch.nn.functional.interpolate(x, [crop_h, crop_w], mode='bilinear')
    assert x.shape[2] % patch_h == 0 and x.shape[3] % patch_w == 0
    x = x.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
    return x


class Mel2Patches_Time_Axis(torch.nn.Module):
    """Convert a Mel Spectrogram to patches"""

    def __init__(self):
        super().__init__()
        self.to_patches = Rearrange('b c h w -> b w (h c)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input img tensor of shape [batch size, channel size, height, width]

        Returns: patches of shape [batch size, num of patches, patch dim]

        """
        return self.to_patches(x)
