"""https://github.com/JunzheJosephZhu/see_hear_feel/tree/master/src/models"""
from torchvision.models import resnet18
import torch.nn as nn
import torch
from torchvision.models.feature_extraction import (
    create_feature_extractor,
)


class CoordConv(nn.Module):
    """Add coordinates in [0,1] to an image, like CoordConv paper."""
    def forward(self, x):
        # needs N,C,H,W inputs
        assert x.ndim == 4
        h, w = x.shape[2:]
        ones_h = x.new_ones((h, 1))
        type_dev = dict(dtype=x.dtype, device=x.device)
        lin_h = torch.linspace(-1, 1, h, **type_dev)[:, None]
        ones_w = x.new_ones((1, w))
        lin_w = torch.linspace(-1, 1, w, **type_dev)[None, :]
        new_maps_2d = torch.stack((lin_h * ones_w, lin_w * ones_h), dim=0)
        new_maps_4d = new_maps_2d[None]  # this line add new dimension, just like what unsqueeze does
        assert new_maps_4d.shape == (1, 2, h, w), (x.shape, new_maps_4d.shape)
        batch_size = x.size(0)
        new_maps_4d_batch = new_maps_4d.repeat(batch_size, 1, 1, 1)
        result = torch.cat((x, new_maps_4d_batch), dim=1)
        return result


class Encoder(nn.Module):
    def __init__(self, feature_extractor, out_dim=None):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.coord_conv = CoordConv()
        if out_dim is not None:
            self.fc = nn.Conv2d(256, out_dim, kernel_size=1)

    def forward(self, x):
        x = self.coord_conv(x)
        x = self.feature_extractor(x)
        assert len(x.values()) == 1
        x = list(x.values())[0]
        if self.fc is not None:
            x = self.fc(x)
        return x


def make_audio_encoder(out_dim=None):
    audio_extractor = resnet18()
    audio_extractor.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
    audio_extractor = create_feature_extractor(audio_extractor, ["layer3.1.relu_1"])

    return Encoder(audio_extractor, out_dim)
