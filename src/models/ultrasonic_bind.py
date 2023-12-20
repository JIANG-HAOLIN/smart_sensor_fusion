import torch
import torch.nn as nn
from torch.nn.modules.activation import MultiheadAttention
from src.models.trafo_classifier_vit import VitImageBind
from src.models.utils.mel_spec import MelSpec
from omegaconf import DictConfig, OmegaConf
import hydra
from src.models.utils.to_patches import seq_2_patches


class SeparateEncoder(torch.nn.Module):
    """Model for short term drilling progress prediction(Y_corr) using acc cage(ac), acc PTU(ap), Force(F),
    Current(I)"""

    def __init__(self,

                 acc_in_feature: int = 240*6,
                 acc_out_feature: int = 512,
                 acc_vit_dim: int = 512,
                 acc_num_pos_emb: int = 151,
                 acc_patch_size: int = 240,

                 f_in_feature: int = 20*3,
                 f_out_feature: int = 512,
                 f_vit_dim: int = 512,
                 f_num_pos_emb: int = 151,
                 f_patch_size: int = 20,

                 i_in_feature: int = 20*2,
                 i_out_feature: int = 512,
                 i_vit_dim: int = 512,
                 i_num_pos_emb: int = 151,
                 i_patch_size: int = 20,

                 s_in_feature: int = 20*2,
                 s_out_feature: int = 512,
                 s_vit_dim: int = 512,
                 s_num_pos_emb: int = 151,
                 s_patch_size: int = 20,


                 ):
        """

        Args:

        """
        super().__init__()
        self.acc_patch_size =
        self.stem_ac = nn.Sequential(
            nn.Linear(
                in_features=acc_in_feature,
                out_features=acc_out_feature,
                bias=False,
            ),
            nn.LayerNorm(normalized_shape=acc_out_feature),
        )
        self.vit_ac = VitImageBind(
            model_dim=acc_vit_dim,
            num_heads=8,
            dropout=0.0,
            input_dropout=0.0,
            num_layers=4,
            num_pos_emb=acc_num_pos_emb,
        )

        self.stem_f = nn.Sequential(
            nn.Linear(
                in_features=f_in_feature,
                out_features=f_out_feature,
                bias=False,
            ),
            nn.LayerNorm(normalized_shape=f_out_feature),
        )
        self.vit_f = VitImageBind(
            model_dim=f_vit_dim,
            num_heads=8,
            dropout=0.0,
            input_dropout=0.0,
            num_layers=4,
            num_pos_emb=f_num_pos_emb,
        )

        self.stem_i = nn.Sequential(
            nn.Linear(
                in_features=i_in_feature,
                out_features=i_out_feature,
                bias=False,
            ),
            nn.LayerNorm(normalized_shape=i_out_feature),
        )
        self.vit_i = VitImageBind(
            model_dim=i_vit_dim,
            num_heads=8,
            dropout=0.0,
            input_dropout=0.0,
            num_layers=4,
            num_pos_emb=i_num_pos_emb,
        )

        self.stem_s = nn.Sequential(
            nn.Linear(
                in_features=s_in_feature,
                out_features=s_out_feature,
                bias=False,
            ),
            nn.LayerNorm(normalized_shape=s_out_feature),
        )
        self.vit_s = VitImageBind(
            model_dim=s_vit_dim,
            num_heads=8,
            dropout=0.0,
            input_dropout=0.0,
            num_layers=4,
            num_pos_emb=s_num_pos_emb,
        )

    def forward(self,
                acc_cage_x: torch.Tensor,
                acc_cage_y: torch.Tensor,
                acc_cage_z: torch.Tensor,
                acc_ptu_x: torch.Tensor,
                acc_ptu_y: torch.Tensor,
                acc_ptu_z: torch.Tensor,
                f_x: torch.Tensor,
                f_y: torch.Tensor,
                f_z: torch.Tensor,
                i_s: torch.Tensor,
                i_z: torch.Tensor,
                s_1: torch.Tensor,
                s_2: torch.Tensor,
                ):
        batch_size = acc_ptu_x.shape[0]

        acc = torch.stack([acc_cage_x, acc_cage_y, acc_cage_z, acc_ptu_x, acc_ptu_y, acc_ptu_z], dim=1)
        acc = torch.nn.functional.interpolate(acc, 36000, mode='linear')
        acc = seq_2_patches(acc, patch_size=240, step_size=240)
        acc_latent, _ = self.vit_ac(self.stem_ac(acc))

        f = torch.stack([f_x, f_y, f_z, ], dim=1)
        f = seq_2_patches(f, patch_size=20, step_size=20)
        f_latent, _ = self.vit_f(self.stem_f(f))

        i = torch.stack([i_s, i_z], dim=1)
        i = seq_2_patches(i, patch_size=20, step_size=20)
        i_latent, _ = self.vit_i(self.stem_i(i))

        s = torch.stack([s_1, s_2], dim=1)
        s = seq_2_patches(s, patch_size=20, step_size=20)
        s_latent, _ = self.vit_s(self.stem_s(s))

        return s_latent, [acc_latent, f_latent, i_latent]
