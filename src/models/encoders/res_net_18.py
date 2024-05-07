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
    """Feature Extractor using Resnet-18"""

    def __init__(self, feature_extractor, in_dim=256, out_dim=None):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.coord_conv = CoordConv()
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        if out_dim is not None:
            self.fc = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: input tensor of shape [Bx(num_stacks), C, H, W]
        Return: tensor of shape [Bx(num_stacks), 1, out_dim]
        """
        if len(x.shape) == 5:
            seq = True
            _, _, c, h, w = x.shape
            x = x.reshape(-1, c, h, w)
        elif len(x.shape) == 4:
            seq = False
            b, c, h, w = x.shape
        else:
            raise RuntimeError("input size wrong")
        x = self.coord_conv(x)
        x = self.feature_extractor(x)
        assert len(x.values()) == 1
        x = list(x.values())[0]
        x = self.maxpool(x)
        if self.fc is not None:
            x = self.fc(x)
        x = torch.flatten(x, start_dim=1).unsqueeze(1)
        return x


class Encoder(nn.Module):
    """Feature Extractor using Resnet-18"""

    def __init__(self, feature_extractor, in_dim=256, out_dim=None):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.coord_conv = CoordConv()
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        if out_dim is not None:
            self.fc = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: input tensor of shape [Bx(num_stacks), C, H, W]
        Return: tensor of shape [Bx(num_stacks), 1, out_dim]
        """
        if len(x.shape) == 5:
            seq = True
            _, _, c, h, w = x.shape
            x = x.reshape(-1, c, h, w)
        elif len(x.shape) == 4:
            seq = False
            b, c, h, w = x.shape
        else:
            raise RuntimeError("input size wrong")
        x = self.coord_conv(x)
        x = self.feature_extractor(x)
        assert len(x.values()) == 1
        x = list(x.values())[0]
        x = self.maxpool(x)
        if self.fc is not None:
            x = self.fc(x)
        x = torch.flatten(x, start_dim=1).unsqueeze(1)
        return x


def make_audio_encoder(out_dim=None, out_layer="layer4.1.relu_1", **kwargs):
    audio_extractor = resnet18()
    audio_extractor.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
    audio_extractor = create_feature_extractor(audio_extractor, [out_layer])
    out_dim_dict = {
        "layer3.1.relu_1": 256,
        "layer4.1.relu_1": 512,
    }
    return Encoder(audio_extractor, in_dim=out_dim_dict[out_layer], out_dim=out_dim)


def make_vision_encoder(out_dim=None, out_layer="layer4.1.relu_1", weight=None, **kwargs, ):
    vision_extractor = resnet18(weight)
    vision_extractor.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=1, padding=3, bias=False)
    vision_extractor = create_feature_extractor(vision_extractor, [out_layer])
    out_dim_dict = {
        "layer3.1.relu_1": 256,
        "layer4.1.relu_1": 512,
    }
    return Encoder(vision_extractor, in_dim=out_dim_dict[out_layer], out_dim=out_dim)


def make_tactile_encoder(out_dim=None, out_layer="layer4.1.relu_1", **kwargs):
    tactile_extractor = resnet18(weights='DEFAULT')
    tactile_extractor.conv1 = nn.Conv2d(
        5, 64, kernel_size=7, stride=1, padding=3, bias=False
    )
    tactile_extractor = create_feature_extractor(tactile_extractor, [out_layer])
    out_dim_dict = {
        "layer3.1.relu_1": 256,
        "layer4.1.relu_1": 512,
    }
    return Encoder(tactile_extractor, in_dim=out_dim_dict[out_layer], out_dim=out_dim)


def make_diffusion_policy_encoder():
    from omegaconf import OmegaConf
    from robomimic.config import config_factory
    import robomimic.scripts.generate_paper_configs as gpc
    from robomimic.scripts.generate_paper_configs import (
        modify_config_for_default_image_exp,
        modify_config_for_default_low_dim_exp,
        modify_config_for_dataset,
    )
    from robomimic.algo import algo_factory
    from robomimic.algo.algo import PolicyAlgo
    from typing import Dict, Callable, List
    import robomimic.utils.obs_utils as ObsUtils
    import robomimic.models.base_nets as rmbn
    import src.models.encoders.utils.crop_randomizer as dmvc

    def replace_submodules(
            root_module: nn.Module,
            predicate: Callable[[nn.Module], bool],
            func: Callable[[nn.Module], nn.Module]) -> nn.Module:
        """
        predicate: Return true if the module is to be replaced.
        func: Return new module to use.
        """
        if predicate(root_module):
            return func(root_module)

        bn_list = [k.split('.') for k, m
                   in root_module.named_modules(remove_duplicate=True)
                   if predicate(m)]
        for *parent, k in bn_list:
            parent_module = root_module
            if len(parent) > 0:
                parent_module = root_module.get_submodule('.'.join(parent))
            if isinstance(parent_module, nn.Sequential):
                src_module = parent_module[int(k)]
            else:
                src_module = getattr(parent_module, k)
            tgt_module = func(src_module)
            if isinstance(parent_module, nn.Sequential):
                parent_module[int(k)] = tgt_module
            else:
                setattr(parent_module, k, tgt_module)
        # verify that all BN are replaced
        bn_list = [k.split('.') for k, m
                   in root_module.named_modules(remove_duplicate=True)
                   if predicate(m)]
        assert len(bn_list) == 0
        return root_module

    def get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='tool_hang',
            dataset_type='ph'
    ):
        base_dataset_dir = '/tmp/null'
        filter_key = None

        # decide whether to use low-dim or image training defaults
        modifier_for_obs = modify_config_for_default_image_exp
        if hdf5_type in ["low_dim", "low_dim_sparse", "low_dim_dense"]:
            modifier_for_obs = modify_config_for_default_low_dim_exp

        algo_config_name = "bc" if algo_name == "bc_rnn" else algo_name
        config = config_factory(algo_name=algo_config_name)
        # turn into default config for observation modalities (e.g.: low-dim or rgb)
        config = modifier_for_obs(config)
        # add in config based on the dataset
        config = modify_config_for_dataset(
            config=config,
            task_name=task_name,
            dataset_type=dataset_type,
            hdf5_type=hdf5_type,
            base_dataset_dir=base_dataset_dir,
            filter_key=filter_key,
        )
        # add in algo hypers based on dataset
        algo_config_modifier = getattr(gpc, f'modify_{algo_name}_config_for_dataset')
        config = algo_config_modifier(
            config=config,
            task_name=task_name,
            dataset_type=dataset_type,
            hdf5_type=hdf5_type,
        )
        return config

    config = get_robomimic_config(
        algo_name='bc_rnn',
        hdf5_type='image',
        task_name='tool_hang',
        dataset_type='ph')

    with config.unlocked():
        # set config with shape_meta
        obs_config = {'depth': [], 'low_dim': [], 'rgb': ['v_fix'], 'scan': []}
        config.observation.modalities.obs = obs_config

        crop_shape = [216, 288]
        if crop_shape is None:
            for key, modality in config.observation.encoder.items():
                if modality.obs_randomizer_class == 'CropRandomizer':
                    modality['obs_randomizer_class'] = None
        else:
            # set random crop parameter
            ch, cw = crop_shape
            for key, modality in config.observation.encoder.items():
                if modality.obs_randomizer_class == 'CropRandomizer':
                    modality.obs_randomizer_kwargs.crop_height = ch
                    modality.obs_randomizer_kwargs.crop_width = cw

    # init global state
    ObsUtils.initialize_obs_utils_with_config(config)

    obs_key_shapes = {'v_fix': [3, 240, 320]}
    # load model
    policy: PolicyAlgo = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=obs_key_shapes,
        ac_dim=7,
        device='cpu',
    )

    obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']

    # replace batch norm with group norm
    replace_submodules(
        root_module=obs_encoder,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features // 16,
            num_channels=x.num_features)
    )
    # obs_encoder.obs_nets['agentview_image'].nets[0].nets

    # obs_encoder.obs_randomizers['agentview_image']
    replace_submodules(
        root_module=obs_encoder,
        predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
        func=lambda x: dmvc.CropRandomizer(
            input_shape=x.input_shape,
            crop_height=x.crop_height,
            crop_width=x.crop_width,
            num_crops=x.num_crops,
            pos_enc=x.pos_enc
        )
    )

    return obs_encoder


def make_resnet18_randomcrop_coordconv_groupnorm_maxpool(**kwargs):
    from copy import deepcopy
    from src.models.encoders.utils.base_nets import SpatialSoftmax
    import torchvision.transforms as T
    spatialsoftmax = SpatialSoftmax(num_kp=32,
                                    learnable_temperature=False,
                                    temperature=1.0,
                                    noise_std=0.0,
                                    output_variance=False,
                                    input_shape=[512, 8, 10])

    class Randomcrop_coordconv_resnet_groupnorm_maxpool(nn.Module):

        def __init__(self, feature_extractor):
            super().__init__()
            self.feature_extractor = feature_extractor
            self.coord_conv = CoordConv()

        def forward(self, x):
            """
            Args:
                x: input tensor of shape [Bx(num_stacks), C, H, W]
            Return: tensor of shape [Bx(num_stacks), 1, out_dim]
            """
            if len(x.shape) == 5:
                seq = True
                _, _, c, h, w = x.shape
                x = x.reshape(-1, c, h, w)
            elif len(x.shape) == 4:
                seq = False
                b, c, h, w = x.shape
            else:
                raise RuntimeError("input size wrong")
            x = self.coord_conv(x)

            if self.training:
                i_v, j_v, h_v, w_v = T.RandomCrop.get_params(
                    x, output_size=(int(x.shape[-2] * 0.9), int(x.shape[-1] * 0.9))
                )
            else:
                i_v, h_v = (
                                   x.shape[-2] - int(x.shape[-2] * 0.9)
                           ) // 2, int(x.shape[-2] * 0.9)
                j_v, w_v = (
                                   x.shape[-1] - int(x.shape[-1] * 0.9)
                           ) // 2, int(x.shape[-1] * 0.9)
            x = x[..., i_v: i_v + h_v, j_v: j_v + w_v]
            # print((i_v + i_v + h_v)/2)
            # print((j_v + j_v + w_v)/2)
            x = self.feature_extractor(x)
            return x.unsqueeze(1)

    resnet = resnet18(weights='DEFAULT')
    resnet.conv1 = nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    resnet.bn1 = nn.GroupNorm(
        num_groups=64 // 16,
        num_channels=64)

    for i in range(1, 5):
        num_features = resnet.__getattr__(f"layer{i}")[0].bn1.num_features
        for j in [0, 1]:
            resnet.__getattr__(f"layer{i}")[j].bn1 = nn.GroupNorm(num_groups=num_features // 16,
                                                                  num_channels=num_features)
            resnet.__getattr__(f"layer{i}")[j].bn2 = nn.GroupNorm(num_groups=num_features // 16,
                                                                  num_channels=num_features)
            if resnet.__getattr__(f"layer{i}")[j].downsample is not None:
                # print(resnet.__getattr__(f"layer{i}")[j].downsample[1])
                resnet.__getattr__(f"layer{i}")[j].downsample[1] = nn.GroupNorm(num_groups=num_features // 16,
                                                                                num_channels=num_features)
    resnet.avgpool = nn.AdaptiveMaxPool2d((1, 1))
    resnet.fc = nn.Identity()
    return Randomcrop_coordconv_resnet_groupnorm_maxpool(resnet)
