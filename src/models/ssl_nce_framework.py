import copy

import copyreg

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention
from src.models.vit_implementations import Vit_Classifier, Vit_Classifier_Mel, LrnEmb_Agg_Trf
from src.models.utils.mel_spec import MelSpec
from src.models.utils.header import ClassificationHead
from src.models.utils.helpers import get_scatter_idx_target, get_mask_sequence1d
from omegaconf import DictConfig, OmegaConf
import hydra
from typing import Optional, Dict, List
from types import SimpleNamespace
import time
from src.models.utils.helpers import cosine_loss_fn, mse_fn


class SslNceFramework_EarlySum(torch.nn.Module):
    """Framework for multi-model self-supervised pretraining"""

    def __init__(self,
                 mod_names: List,
                 main_mod: str,
                 model_dim: int,
                 num_stack: int,

                 nce_args: DictConfig,
                 fom_args: DictConfig,
                 mask_args: DictConfig,

                 audio_args: Optional[DictConfig] = None,
                 vision_args: Optional[DictConfig] = None,
                 tactile_args: Optional[DictConfig] = None,
                 ultrasonic_args: Optional[DictConfig] = None,
                 imu_args: Optional[DictConfig] = None,
                 force_args: Optional[DictConfig] = None,
                 current_args: Optional[DictConfig] = None,
                 thermal_args: Optional[DictConfig] = None,
                 depth_args: Optional[DictConfig] = None,
                 text_args: Optional[DictConfig] = None,

                 fusion_args: Optional[DictConfig] = None,
                 pos_emb_args: Optional[DictConfig] = None,
                 cross_time_trf_args: Optional[DictConfig] = None,

                 **kwargs
                 ):
        """

        Args:
             preprocess_audio_args: arguments for audio prepressing
             tokenization_audio: arguments for audio tokenization
             pe_audio: arguments for positional encoding for audio tokens
             encoder_audio_args: arguments for audio encoder(identity for earlycat/transformer for multi to one)
             preprocess_vision_args: arguments for vision prepressing
             tokenization_vision: arguments for vision tokenization
             pe_vision: arguments for positional encoding for vision tokens
             encoder_vision_args: arguments for vision encoder(identity for earlycat/transformer for multi to one)
             cross_time_trf_args: arguments for cross time transformer
             **kwargs:
        """
        super().__init__()

        self.mod_args = {
            "vision": vision_args,
            "tactile": tactile_args,
            "audio": audio_args,

            "ultrasonic": ultrasonic_args,
            "imu": imu_args,
            "force": force_args,
            "current": current_args,

            "thermal": thermal_args,
            "depth": depth_args,
            "text": text_args,
        }
        self.mod_names = mod_names
        self.main_mod = main_mod
        self.num_stack = num_stack

        self.fom_args = fom_args
        self.mask_args = mask_args
        self.nce_args = nce_args
        self.nce_head = hydra.utils.instantiate(nce_args.nce_proj_head)

        self.mask_fusion_nce_proj = hydra.utils.instantiate(mask_args.mask_fusion_nce.proj_head) if \
            mask_args.mask_fusion_nce is not None else None

        self.mask_cross_time_trf_proj = hydra.utils.instantiate(mask_args.mask_cross_time_trf_nce.proj_head) if \
            mask_args.mask_cross_time_trf_nce is not None else None

        self.mask_latent_predictor = hydra.utils.instantiate(mask_args.mask_latent_prediction.predictor) if \
            mask_args.mask_latent_prediction is not None else None

        for mod_name, mod_args in self.mod_args.items():
            if mod_args is not None:
                self.__setattr__(f"preprocess_{mod_name}",
                                 hydra.utils.instantiate(mod_args[f"preprocess_{mod_name}_args"]))
                self.__setattr__(f"tokenization_{mod_name}",
                                 hydra.utils.instantiate(mod_args[f"tokenization_{mod_name}"]))
                self.__setattr__(f"positional_encoding_{mod_name}",
                                 hydra.utils.instantiate(mod_args[f"pe_{mod_name}"]))
                self.__setattr__(f"encoder_{mod_name}",
                                 hydra.utils.instantiate(mod_args[f"encoder_{mod_name}_args"]))

        self.fusion = hydra.utils.instantiate(fusion_args)

        self.cls = torch.nn.Parameter(torch.randn(1, 1, model_dim))
        self.cross_time_pos_emb = hydra.utils.instantiate(pos_emb_args)
        self.cross_time_trf = hydra.utils.instantiate(cross_time_trf_args)

        self.fom_classifier = ClassificationHead(model_dim=model_dim, num_classes=num_stack)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(model_dim, model_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(model_dim, model_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(model_dim, 3 ** 3),
        )
        self.aux_mlp = torch.nn.Linear(model_dim, 6)

    def forward(self, multimod_inputs: Dict,
                mask: bool = True,
                task: Optional[str] = "imitation",
                mode: str = "train",
                ):
        assert mode in ["train", "val", "inference"]
        start_time = time.time()
        output = {"predict": {}, "ssl_losses": {}, "repr": {}}

        encoded_inputs = self.forward_modality_specific_encoding(multimod_inputs)

        if task == "repr":
            output["repr"]["encoded_inputs"] = encoded_inputs
            encoded_inputs = self.fusion(encoded_inputs)
            output["repr"]['fused_encoded_inputs'] = encoded_inputs
            x, attn_maps = self.forward_cross_time(encoded_inputs)
            output["repr"]['cross_time_repr'] = x
            agg_feat = x[:, 0, :]
            action_logits = self.mlp(agg_feat)
            xyzrpy = self.aux_mlp(agg_feat)
            output["predict"]["action_logits"] = action_logits
            output["predict"]["xyzrpy"] = xyzrpy
            output["time"] = time.time() - start_time
            return output

        if "bind" in task:
            nce_loss_dict = self.forward_nce(encoded_inputs)
            loss = nce_loss_dict['_loss']
            output["ssl_losses"]["cr_m_nce_loss"] = loss

        fused_t_feats = self.fusion(encoded_inputs)

        if mask:
            masked_multimod_inputs = self.random_masking(multimod_inputs=multimod_inputs,
                                                         masked_mod=self.mask_args.masked_mod,
                                                         mask_prob_args=self.mask_args.mask_prob_args,
                                                         mask_length_args=self.mask_args.mask_length_args, )
            encoded_masked_inputs = self.forward_modality_specific_encoding(masked_multimod_inputs)

            mask_pred_target = fused_t_feats.detach()
            fused_t_masked_feats = self.fusion(encoded_masked_inputs)

            if 'order' in task:
                fom_loss = self.forward_order_prediction(fused_t_masked_feats)
                output["ssl_losses"]["masked_fom_loss"] = fom_loss

            if 'fuse_nce' in task and self.mask_fusion_nce_proj is not None:
                temp = self.mask_args.mask_fusion_nce.temp
                mask_fusion_nce_loss = self.mask_cross_time_nce(self.mask_fusion_nce_proj,
                                                                fused_t_feats, fused_t_masked_feats, temp)
                output["ssl_losses"]["mask_fusion_nce_loss"] = mask_fusion_nce_loss['_loss']

            fused_t_feats, _ = self.forward_cross_time(fused_t_feats)
            agg_feat = fused_t_feats[:, 0, :]
            fused_t_feats = fused_t_feats[:, 1:, :]
            fused_t_masked_feats, _ = self.forward_cross_time(fused_t_masked_feats)
            fused_t_masked_feats = fused_t_masked_feats[:, 1:, :]

            if 'cross_time_nce' in task and self.mask_cross_time_trf_proj is not None:
                temp = self.mask_args.mask_cross_time_trf_nce.temp
                mask_cross_time_nce_loss = self.mask_cross_time_nce(self.mask_cross_time_trf_proj,
                                                                    fused_t_feats, fused_t_masked_feats, temp)
                output["ssl_losses"]["mask_cr_t_nce_loss"] = mask_cross_time_nce_loss['_loss']

            if 'recover' in task and self.mask_latent_predictor is not None:
                recover_loss = self.mask_prediction(self.mask_latent_predictor,
                                                    mask_pred_target, fused_t_masked_feats)
                output["ssl_losses"]["recover_loss"] = recover_loss

            if "imitation" in task:
                action_logits = self.mlp(agg_feat)
                xyzrpy = self.aux_mlp(agg_feat)
                output["predict"]["action_logits"] = action_logits
                output["predict"]["xyzrpy"] = xyzrpy

        else:
            if "order" in task:
                fom_loss = self.forward_order_prediction(fused_t_feats)
                output["ssl_losses"]["fom_loss"] = fom_loss
            if "imitation" in task:
                fused_t_feats, _ = self.forward_cross_time(fused_t_feats)
                agg_feat = fused_t_feats[:, 0, :]
                action_logits = self.mlp(agg_feat)
                xyzrpy = self.aux_mlp(agg_feat)
                output["predict"]["action_logits"] = action_logits
                output["predict"]["xyzrpy"] = xyzrpy

        return output

    def forward_cross_time(self, x):
        cls = self.cls.expand(x.shape[0], self.cls.shape[1], self.cls.shape[2])
        x = torch.cat([cls, x], dim=1)
        x = self.cross_time_pos_emb(x)
        x, attn_maps = self.cross_time_trf(x)
        return x, attn_maps

    def forward_order_prediction(self, fused_t_feats: torch.Tensor, compute_loss=True):

        bs, num_frame, dim = fused_t_feats.shape

        target, shuffled_fused_t_feats = self.random_shuffle(fused_t_feats, reorder_prob=self.fom_args.reorder_prob)

        if all(element == -1 for element in target.reshape(bs * num_frame, )):
            return torch.zeros([])

        x, attn_maps = self.forward_cross_time(shuffled_fused_t_feats)
        x = x[:, 1:, :]
        x = self.fom_classifier(x)

        if compute_loss:
            x = x.view(bs * num_frame, num_frame)
            targets = target.view(x.shape[0])
            loss = F.cross_entropy(
                x, targets, ignore_index=-1,
                reduction='mean')
            return loss
        return x

    @staticmethod
    def random_shuffle(fused_t_feats: torch.Tensor, reorder_prob: float):
        bs, num_frame, dim = fused_t_feats.shape
        shuffled_orders = []
        targets = []
        for i in range(bs):
            shuffled_order, target = get_scatter_idx_target(list(range(num_frame)), reorder_prob=reorder_prob)
            shuffled_order = torch.tensor(shuffled_order, device=fused_t_feats.device)
            target = torch.tensor(target, device=fused_t_feats.device)
            shuffled_orders.append(shuffled_order)
            targets.append(target)
        targets = torch.stack(targets, dim=0)
        shuffled_orders = torch.stack(shuffled_orders, dim=0).unsqueeze(-1)  # [b, num_frame, 1]
        shuffled_orders = shuffled_orders.expand(-1, -1, dim)
        shuffled_fused_t_feats = torch.zeros_like(fused_t_feats,
                                                  dtype=fused_t_feats.dtype, device=fused_t_feats.device)
        shuffled_fused_t_feats = shuffled_fused_t_feats.scatter_(1, shuffled_orders, fused_t_feats)
        # a = fused_t_feats[:, :, 10]
        # b = shuffled_fused_t_feats[:, :, 10]
        return targets, shuffled_fused_t_feats

    def forward_nce(self, encoded_inputs, compute_loss=True):
        main_feats = self.nce_head(encoded_inputs[self.nce_args.main_mod])
        bs, num_frame, dim = main_feats.shape
        main_feats = main_feats.reshape(bs, num_frame * dim)
        main_feats = main_feats[:, None, :]

        # Find positive example -> batch_size//2 away from the original example
        pos_mask = torch.eye(bs, bs, dtype=torch.bool, device=main_feats.device)
        # InfoNCE loss
        nll_sum = torch.zeros([])
        for name, feat in encoded_inputs.items():
            if name != self.nce_args.main_mod:
                side_feat = self.nce_head(feat)
                side_feat = side_feat.reshape(bs, num_frame * dim)
                cos_sim = F.cosine_similarity(main_feats, side_feat[None, :, :], dim=-1)
                cos_sim = cos_sim / self.nce_args.temp
                nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
                nll_ = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=0)
                nll = nll.mean()
                nll_ = nll_.mean()
                nll_sum = nll_sum + nll + nll_
        nce_loss_dict = {
            '_loss': nll_sum,
        }

        return nce_loss_dict

    @staticmethod
    def mask_cross_time_nce(proj: nn.Module,
                            fused_t_feats: torch.Tensor, fused_t_masked_feats: torch.Tensor,
                            temp: float, compute_loss=True):
        proj_fused_t_feats = proj(fused_t_feats)
        proj_fused_t_masked_feats = proj(fused_t_masked_feats)

        bs, num_frame, dim = proj_fused_t_feats.shape
        proj_fused_t_feats = proj_fused_t_feats.reshape(bs, num_frame * dim)
        proj_fused_t_feats = proj_fused_t_feats[:, None, :]

        bs, num_frame, dim = proj_fused_t_masked_feats.shape
        proj_fused_t_masked_feats = proj_fused_t_masked_feats.reshape(bs, num_frame * dim)
        proj_fused_t_masked_feats = proj_fused_t_masked_feats[None, :, :]

        pos_mask = torch.eye(bs, bs, dtype=torch.bool, device=proj_fused_t_feats.device)
        # InfoNCE loss
        nll_sum = torch.zeros([])
        cos_sim = F.cosine_similarity(proj_fused_t_feats, proj_fused_t_masked_feats, dim=-1)
        cos_sim = cos_sim / temp
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll_ = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=0)
        nll = nll.mean()
        nll_ = nll_.mean()
        nll_sum = nll_sum + nll + nll_

        nce_loss_dict = {
            '_loss': nll_sum,
        }

        return nce_loss_dict

    def forward_modality_specific_encoding(self, multimod_inputs: Dict, ):
        encoded_inputs = {}

        # process of vision signal
        if "vision" in multimod_inputs.keys():
            vision_signal = multimod_inputs["vision"]
            batch_size, num_frame, c_v, h_v, w_v = vision_signal.shape
            short_window_len = self.mod_args["vision"].short_window_len
            assert num_frame % short_window_len == 0
            num_stack = num_frame // short_window_len
            assert num_stack == self.num_stack
            vision_signal = torch.reshape(vision_signal, (-1, c_v, h_v, w_v))
            vision_signal = self.preprocess_vision(vision_signal)
            vision_signal = torch.reshape(vision_signal, (-1, short_window_len, c_v, h_v, w_v))
            vision_signal = self.tokenization_vision(vision_signal)  # vit
            if type(vision_signal) == tuple:
                vision_signal, attn_map = vision_signal
            if len(vision_signal.shape) == 2:
                vision_signal = vision_signal.reshape(vision_signal.shape[0], 1, vision_signal.shape[1])
                # [B*N, 256] -> [B*N, 1, 256]
            vision_signal = vision_signal.view(batch_size, num_stack * vision_signal.shape[-2], vision_signal.shape[-1])
            # [B*N, 1, 256] -> [B, N, 256]
            vision_signal = self.positional_encoding_vision(vision_signal)
            vision_signal = self.encoder_vision(vision_signal)
            if type(vision_signal) == tuple:
                vision_signal, attn_vision = vision_signal
            encoded_inputs["vision"] = vision_signal

        # process of audio signal
        if "audio" in multimod_inputs.keys():
            audio_signal = multimod_inputs["audio"]
            audio_signal = self.preprocess_audio(audio_signal)
            # [B, 1, 40K] -> [B, 64, 251]
            audio_signal = self.tokenization_audio(audio_signal)
            # [B*N, 1, 256] -> [B, N, 256]
            audio_signal = self.positional_encoding_audio(audio_signal)
            audio_signal = self.encoder_audio(audio_signal)
            if type(audio_signal) == tuple:
                audio_signal, attn_audio = audio_signal
            encoded_inputs["audio"] = audio_signal

        # process of tactile signal(similar to vision signal)
        if "tactile" in multimod_inputs.keys():
            tactile_signal = multimod_inputs["tactile"]
            batch_size, num_frame, c_v, h_v, w_v = tactile_signal.shape
            short_window_len = self.mod_args["tactile"].short_window_len
            assert num_frame % short_window_len == 0
            num_stack = num_frame // short_window_len
            assert num_stack == self.num_stack
            tactile_signal = torch.reshape(tactile_signal, (-1, c_v, h_v, w_v))
            tactile_signal = self.preprocess_tactile(tactile_signal)
            tactile_signal = torch.reshape(tactile_signal, (-1, short_window_len, c_v, h_v, w_v))
            tactile_signal = self.tokenization_tactile(tactile_signal)  # vit
            if type(tactile_signal) == tuple:
                tactile_signal, attn_map = tactile_signal
            if len(tactile_signal.shape) == 2:
                tactile_signal = tactile_signal.reshape(tactile_signal.shape[0], 1, tactile_signal.shape[1])
                # [B*N, 256] -> [B*N, 1, 256]
            tactile_signal = tactile_signal.view(batch_size, num_stack * tactile_signal.shape[-2],
                                                 tactile_signal.shape[-1])
            # [B*N, 1, 256] -> [B, N, 256]
            tactile_signal = self.positional_encoding_tactile(tactile_signal)
            tactile_signal = self.encoder_tactile(tactile_signal)
            if type(tactile_signal) == tuple:
                tactile_signal, attn_tactile = tactile_signal
            encoded_inputs["tactile"] = tactile_signal

        return encoded_inputs

    @staticmethod
    def random_masking(masked_mod: List,
                       mask_prob_args: DictConfig,
                       mask_length_args: DictConfig,
                       multimod_inputs: Dict):
        mask_multimod_inputs = {}
        for mod_name, mod_feat in multimod_inputs.items():
            if mod_name in masked_mod:
                mask_porb = mask_prob_args[mod_name]
                mask_length = mask_length_args[mod_name]
                if mod_name in ["vision", "tactile"]:
                    # mask whole image
                    bs, num_frame, c, h, w = mod_feat.shape
                    mask = torch.tensor(get_mask_sequence1d(seq_len=bs * num_frame,
                                                            mask_prob=mask_porb,
                                                            mask_length=mask_length,
                                                            mask_mark=0,
                                                            unmask_mark=1, ), device=mod_feat.device)
                    mod_feat = mod_feat * mask.reshape(bs, num_frame, 1, 1, 1)
                elif mod_name in ["audio", "ultrasonic", "imu", "force", "current", ]:
                    # channel-wise variant
                    bs, c, l = mod_feat.shape
                    mask = torch.tensor(get_mask_sequence1d(seq_len=bs * c * l,
                                                            mask_prob=mask_porb,
                                                            mask_length=mask_length,
                                                            mask_mark=0,
                                                            unmask_mark=1, ), device=mod_feat.device)
                    mod_feat = mod_feat * mask.reshape(bs, c, l)
            mask_multimod_inputs[mod_name] = mod_feat
        return multimod_inputs

    @staticmethod
    def mask_prediction(predictor: nn.Module,
                        target: torch.Tensor,
                        fused_t_mask_feats: torch.Tensor, ):
        pred_out = predictor(fused_t_mask_feats)
        pred_loss = F.mse_loss(pred_out, target)
        return pred_loss


class SslNceFramework_EarlySum_DifHd(torch.nn.Module):
    """Framework for multi-model self-supervised pretraining"""

    def __init__(self,
                 mod_names: List,
                 main_mod: str,
                 model_dim: int,
                 num_stack: int,

                 nce_args: DictConfig,
                 fom_args: DictConfig,
                 mask_args: DictConfig,

                 audio_args: Optional[DictConfig] = None,
                 vision_args: Optional[DictConfig] = None,
                 tactile_args: Optional[DictConfig] = None,
                 ultrasonic_args: Optional[DictConfig] = None,
                 imu_args: Optional[DictConfig] = None,
                 force_args: Optional[DictConfig] = None,
                 current_args: Optional[DictConfig] = None,
                 thermal_args: Optional[DictConfig] = None,
                 depth_args: Optional[DictConfig] = None,
                 text_args: Optional[DictConfig] = None,

                 fusion_args: Optional[DictConfig] = None,
                 pos_emb_args: Optional[DictConfig] = None,
                 cross_time_trf_args: Optional[DictConfig] = None,

                 **kwargs
                 ):
        """

        Args:
             preprocess_audio_args: arguments for audio prepressing
             tokenization_audio: arguments for audio tokenization
             pe_audio: arguments for positional encoding for audio tokens
             encoder_audio_args: arguments for audio encoder(identity for earlycat/transformer for multi to one)
             preprocess_vision_args: arguments for vision prepressing
             tokenization_vision: arguments for vision tokenization
             pe_vision: arguments for positional encoding for vision tokens
             encoder_vision_args: arguments for vision encoder(identity for earlycat/transformer for multi to one)
             cross_time_trf_args: arguments for cross time transformer
             **kwargs:
        """
        super().__init__()

        self.mod_args = {
            "vision": vision_args,
            "tactile": tactile_args,
            "audio": audio_args,

            "ultrasonic": ultrasonic_args,
            "imu": imu_args,
            "force": force_args,
            "current": current_args,

            "thermal": thermal_args,
            "depth": depth_args,
            "text": text_args,
        }
        self.mod_names = mod_names
        self.main_mod = main_mod
        self.num_stack = num_stack

        self.fom_args = fom_args
        self.mask_args = mask_args
        self.nce_args = nce_args

        self.nce_head_dict = torch.nn.ModuleDict({mod_name: hydra.utils.instantiate(nce_args.nce_proj_head)
                                                  for mod_name, v in self.mod_args.items() if v is not None})

        self.mask_fusion_nce_proj = hydra.utils.instantiate(mask_args.mask_fusion_nce.proj_head) if \
            mask_args.mask_fusion_nce is not None else None

        self.mask_cross_time_trf_proj = hydra.utils.instantiate(mask_args.mask_cross_time_trf_nce.proj_head) if \
            mask_args.mask_cross_time_trf_nce is not None else None

        self.mask_latent_predictor = hydra.utils.instantiate(mask_args.mask_latent_prediction.predictor) if \
            mask_args.mask_latent_prediction is not None else None

        for mod_name, mod_args in self.mod_args.items():
            if mod_args is not None:
                self.__setattr__(f"preprocess_{mod_name}",
                                 hydra.utils.instantiate(mod_args[f"preprocess_{mod_name}_args"]))
                self.__setattr__(f"tokenization_{mod_name}",
                                 hydra.utils.instantiate(mod_args[f"tokenization_{mod_name}"]))
                self.__setattr__(f"positional_encoding_{mod_name}",
                                 hydra.utils.instantiate(mod_args[f"pe_{mod_name}"]))
                self.__setattr__(f"encoder_{mod_name}",
                                 hydra.utils.instantiate(mod_args[f"encoder_{mod_name}_args"]))

        self.fusion = hydra.utils.instantiate(fusion_args)

        self.cls = torch.nn.Parameter(torch.randn(1, 1, model_dim))
        self.cross_time_pos_emb = hydra.utils.instantiate(pos_emb_args)
        self.cross_time_trf = hydra.utils.instantiate(cross_time_trf_args)

        self.fom_classifier = ClassificationHead(model_dim=model_dim, num_classes=num_stack)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(model_dim, model_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(model_dim, model_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(model_dim, 3 ** 3),
        )
        self.aux_mlp = torch.nn.Linear(model_dim, 6)

    def forward(self, multimod_inputs: Dict,
                mask: bool = True,
                task: Optional[str] = "imitation",
                mode: str = "train",
                ):
        assert mode in ["train", "val", "inference"]
        start_time = time.time()
        output = {"predict": {}, "ssl_losses": {}, "repr": {}}

        encoded_inputs = self.forward_modality_specific_encoding(multimod_inputs)

        if task == "repr":
            output["repr"]["encoded_inputs"] = encoded_inputs
            encoded_inputs = self.fusion(encoded_inputs)
            output["repr"]['fused_encoded_inputs'] = encoded_inputs
            x, attn_maps = self.forward_cross_time(encoded_inputs)
            output["repr"]['cross_time_repr'] = x
            agg_feat = x[:, 0, :]
            action_logits = self.mlp(agg_feat)
            xyzrpy = self.aux_mlp(agg_feat)
            output["predict"]["action_logits"] = action_logits
            output["predict"]["xyzrpy"] = xyzrpy
            output["time"] = time.time() - start_time
            return output

        if "bind" in task:
            nce_loss_dict = self.forward_nce(encoded_inputs)
            loss = nce_loss_dict['_loss']
            output["ssl_losses"]["cr_m_nce_loss"] = loss

        fused_t_feats = self.fusion(encoded_inputs)

        if mask:
            masked_multimod_inputs = self.random_masking(multimod_inputs=multimod_inputs,
                                                         masked_mod=self.mask_args.masked_mod,
                                                         mask_prob_args=self.mask_args.mask_prob_args,
                                                         mask_length_args=self.mask_args.mask_length_args, )
            encoded_masked_inputs = self.forward_modality_specific_encoding(masked_multimod_inputs)

            mask_pred_target = fused_t_feats.detach()
            fused_t_masked_feats = self.fusion(encoded_masked_inputs)

            if 'order' in task:
                fom_loss = self.forward_order_prediction(fused_t_masked_feats)
                output["ssl_losses"]["masked_fom_loss"] = fom_loss

            if 'fuse_nce' in task and self.mask_fusion_nce_proj is not None:
                temp = self.mask_args.mask_fusion_nce.temp
                mask_fusion_nce_loss = self.mask_cross_time_nce(self.mask_fusion_nce_proj,
                                                                fused_t_feats, fused_t_masked_feats, temp)
                output["ssl_losses"]["mask_fusion_nce_loss"] = mask_fusion_nce_loss['_loss']

            fused_t_feats, _ = self.forward_cross_time(fused_t_feats)
            agg_feat = fused_t_feats[:, 0, :]
            fused_t_feats = fused_t_feats[:, 1:, :]
            fused_t_masked_feats, _ = self.forward_cross_time(fused_t_masked_feats)
            fused_t_masked_feats = fused_t_masked_feats[:, 1:, :]

            if 'cross_time_nce' in task and self.mask_cross_time_trf_proj is not None:
                temp = self.mask_args.mask_cross_time_trf_nce.temp
                mask_cross_time_nce_loss = self.mask_cross_time_nce(self.mask_cross_time_trf_proj,
                                                                    fused_t_feats, fused_t_masked_feats, temp)
                output["ssl_losses"]["mask_cr_t_nce_loss"] = mask_cross_time_nce_loss['_loss']

            if 'recover' in task and self.mask_latent_predictor is not None:
                recover_loss = self.mask_prediction(self.mask_latent_predictor,
                                                    mask_pred_target, fused_t_masked_feats)
                output["ssl_losses"]["recover_loss"] = recover_loss

            if "imitation" in task:
                action_logits = self.mlp(agg_feat)
                xyzrpy = self.aux_mlp(agg_feat)
                output["predict"]["action_logits"] = action_logits
                output["predict"]["xyzrpy"] = xyzrpy

        else:
            if "order" in task:
                fom_loss = self.forward_order_prediction(fused_t_feats)
                output["ssl_losses"]["fom_loss"] = fom_loss
            if "imitation" in task:
                fused_t_feats, _ = self.forward_cross_time(fused_t_feats)
                agg_feat = fused_t_feats[:, 0, :]
                action_logits = self.mlp(agg_feat)
                xyzrpy = self.aux_mlp(agg_feat)
                output["predict"]["action_logits"] = action_logits
                output["predict"]["xyzrpy"] = xyzrpy

        return output

    def forward_cross_time(self, x):
        cls = self.cls.expand(x.shape[0], self.cls.shape[1], self.cls.shape[2])
        x = torch.cat([cls, x], dim=1)
        x = self.cross_time_pos_emb(x)
        x, attn_maps = self.cross_time_trf(x)
        return x, attn_maps

    def forward_order_prediction(self, fused_t_feats: torch.Tensor, compute_loss=True):

        bs, num_frame, dim = fused_t_feats.shape

        target, shuffled_fused_t_feats = self.random_shuffle(fused_t_feats, reorder_prob=self.fom_args.reorder_prob)

        if all(element == -1 for element in target.reshape(bs * num_frame, )):
            return torch.zeros([])

        x, attn_maps = self.forward_cross_time(shuffled_fused_t_feats)
        x = x[:, 1:, :]
        x = self.fom_classifier(x)

        if compute_loss:
            x = x.view(bs * num_frame, num_frame)
            targets = target.view(x.shape[0])
            loss = F.cross_entropy(
                x, targets, ignore_index=-1,
                reduction='mean')
            return loss
        return x

    @staticmethod
    def random_shuffle(fused_t_feats: torch.Tensor, reorder_prob: float):
        bs, num_frame, dim = fused_t_feats.shape
        shuffled_orders = []
        targets = []
        for i in range(bs):
            shuffled_order, target = get_scatter_idx_target(list(range(num_frame)), reorder_prob=reorder_prob)
            shuffled_order = torch.tensor(shuffled_order, device=fused_t_feats.device)
            target = torch.tensor(target, device=fused_t_feats.device)
            shuffled_orders.append(shuffled_order)
            targets.append(target)
        targets = torch.stack(targets, dim=0)
        shuffled_orders = torch.stack(shuffled_orders, dim=0).unsqueeze(-1)  # [b, num_frame, 1]
        shuffled_orders = shuffled_orders.expand(-1, -1, dim)
        shuffled_fused_t_feats = torch.zeros_like(fused_t_feats,
                                                  dtype=fused_t_feats.dtype, device=fused_t_feats.device)
        shuffled_fused_t_feats = shuffled_fused_t_feats.scatter_(1, shuffled_orders, fused_t_feats)
        # a = fused_t_feats[:, :, 10]
        # b = shuffled_fused_t_feats[:, :, 10]
        return targets, shuffled_fused_t_feats

    def forward_nce(self, encoded_inputs, compute_loss=True):
        main_feats = self.nce_head_dict[self.nce_args.main_mod](encoded_inputs[self.nce_args.main_mod])
        bs, num_frame, dim = main_feats.shape
        main_feats = main_feats.reshape(bs, num_frame * dim)
        main_feats = main_feats[:, None, :]

        # Find positive example -> batch_size//2 away from the original example
        pos_mask = torch.eye(bs, bs, dtype=torch.bool, device=main_feats.device)
        # InfoNCE loss
        nll_sum = torch.zeros([])
        for name, feat in encoded_inputs.items():
            if name != self.nce_args.main_mod:
                side_feat = self.nce_head_dict[name](feat)
                side_feat = side_feat.reshape(bs, num_frame * dim)
                cos_sim = F.cosine_similarity(main_feats, side_feat[None, :, :], dim=-1)
                cos_sim = cos_sim / self.nce_args.temp
                nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
                nll_ = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=0)
                nll = nll.mean()
                nll_ = nll_.mean()
                nll_sum = nll_sum + nll + nll_
        nce_loss_dict = {
            '_loss': nll_sum,
        }

        return nce_loss_dict

    @staticmethod
    def mask_cross_time_nce(proj: nn.Module,
                            fused_t_feats: torch.Tensor, fused_t_masked_feats: torch.Tensor,
                            temp: float, compute_loss=True):
        proj_fused_t_feats = proj(fused_t_feats)
        proj_fused_t_masked_feats = proj(fused_t_masked_feats)

        bs, num_frame, dim = proj_fused_t_feats.shape
        proj_fused_t_feats = proj_fused_t_feats.reshape(bs, num_frame * dim)
        proj_fused_t_feats = proj_fused_t_feats[:, None, :]

        bs, num_frame, dim = proj_fused_t_masked_feats.shape
        proj_fused_t_masked_feats = proj_fused_t_masked_feats.reshape(bs, num_frame * dim)
        proj_fused_t_masked_feats = proj_fused_t_masked_feats[None, :, :]

        pos_mask = torch.eye(bs, bs, dtype=torch.bool, device=proj_fused_t_feats.device)
        # InfoNCE loss
        nll_sum = torch.zeros([])
        cos_sim = F.cosine_similarity(proj_fused_t_feats, proj_fused_t_masked_feats, dim=-1)
        cos_sim = cos_sim / temp
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll_ = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=0)
        nll = nll.mean()
        nll_ = nll_.mean()
        nll_sum = nll_sum + nll + nll_

        nce_loss_dict = {
            '_loss': nll_sum,
        }

        return nce_loss_dict

    def forward_modality_specific_encoding(self, multimod_inputs: Dict, ):
        encoded_inputs = {}

        # process of vision signal
        if "vision" in multimod_inputs.keys():
            vision_signal = multimod_inputs["vision"]
            batch_size, num_frame, c_v, h_v, w_v = vision_signal.shape
            short_window_len = self.mod_args["vision"].short_window_len
            assert num_frame % short_window_len == 0
            num_stack = num_frame // short_window_len
            assert num_stack == self.num_stack
            vision_signal = torch.reshape(vision_signal, (-1, c_v, h_v, w_v))
            vision_signal = self.preprocess_vision(vision_signal)
            vision_signal = torch.reshape(vision_signal, (-1, short_window_len, c_v, h_v, w_v))
            vision_signal = self.tokenization_vision(vision_signal)  # vit
            if type(vision_signal) == tuple:
                vision_signal, attn_map = vision_signal
            if len(vision_signal.shape) == 2:
                vision_signal = vision_signal.reshape(vision_signal.shape[0], 1, vision_signal.shape[1])
                # [B*N, 256] -> [B*N, 1, 256]
            vision_signal = vision_signal.view(batch_size, num_stack * vision_signal.shape[-2], vision_signal.shape[-1])
            # [B*N, 1, 256] -> [B, N, 256]
            vision_signal = self.positional_encoding_vision(vision_signal)
            vision_signal = self.encoder_vision(vision_signal)
            if type(vision_signal) == tuple:
                vision_signal, attn_vision = vision_signal
            encoded_inputs["vision"] = vision_signal

        # process of audio signal
        if "audio" in multimod_inputs.keys():
            audio_signal = multimod_inputs["audio"]
            audio_signal = self.preprocess_audio(audio_signal)
            # [B, 1, 40K] -> [B, 64, 251]
            audio_signal = self.tokenization_audio(audio_signal)
            # [B*N, 1, 256] -> [B, N, 256]
            audio_signal = self.positional_encoding_audio(audio_signal)
            audio_signal = self.encoder_audio(audio_signal)
            if type(audio_signal) == tuple:
                audio_signal, attn_audio = audio_signal
            encoded_inputs["audio"] = audio_signal

        # process of tactile signal(similar to vision signal)
        if "tactile" in multimod_inputs.keys():
            tactile_signal = multimod_inputs["tactile"]
            batch_size, num_frame, c_v, h_v, w_v = tactile_signal.shape
            short_window_len = self.mod_args["tactile"].short_window_len
            assert num_frame % short_window_len == 0
            num_stack = num_frame // short_window_len
            assert num_stack == self.num_stack
            tactile_signal = torch.reshape(tactile_signal, (-1, c_v, h_v, w_v))
            tactile_signal = self.preprocess_tactile(tactile_signal)
            tactile_signal = torch.reshape(tactile_signal, (-1, short_window_len, c_v, h_v, w_v))
            tactile_signal = self.tokenization_tactile(tactile_signal)  # vit
            if type(tactile_signal) == tuple:
                tactile_signal, attn_map = tactile_signal
            if len(tactile_signal.shape) == 2:
                tactile_signal = tactile_signal.reshape(tactile_signal.shape[0], 1, tactile_signal.shape[1])
                # [B*N, 256] -> [B*N, 1, 256]
            tactile_signal = tactile_signal.view(batch_size, num_stack * tactile_signal.shape[-2],
                                                 tactile_signal.shape[-1])
            # [B*N, 1, 256] -> [B, N, 256]
            tactile_signal = self.positional_encoding_tactile(tactile_signal)
            tactile_signal = self.encoder_tactile(tactile_signal)
            if type(tactile_signal) == tuple:
                tactile_signal, attn_tactile = tactile_signal
            encoded_inputs["tactile"] = tactile_signal

        return encoded_inputs

    @staticmethod
    def random_masking(masked_mod: List,
                       mask_prob_args: DictConfig,
                       mask_length_args: DictConfig,
                       multimod_inputs: Dict):
        mask_multimod_inputs = {}
        for mod_name, mod_feat in multimod_inputs.items():
            if mod_name in masked_mod:
                mask_porb = mask_prob_args[mod_name]
                mask_length = mask_length_args[mod_name]
                if mod_name in ["vision", "tactile"]:
                    # mask whole image
                    bs, num_frame, c, h, w = mod_feat.shape
                    mask = torch.tensor(get_mask_sequence1d(seq_len=bs * num_frame,
                                                            mask_prob=mask_porb,
                                                            mask_length=mask_length,
                                                            mask_mark=0,
                                                            unmask_mark=1, ), device=mod_feat.device)
                    mod_feat = mod_feat * mask.reshape(bs, num_frame, 1, 1, 1)
                elif mod_name in ["audio", "ultrasonic", "imu", "force", "current", ]:
                    # channel-wise variant
                    bs, c, l = mod_feat.shape
                    mask = torch.tensor(get_mask_sequence1d(seq_len=bs * c * l,
                                                            mask_prob=mask_porb,
                                                            mask_length=mask_length,
                                                            mask_mark=0,
                                                            unmask_mark=1, ), device=mod_feat.device)
                    mod_feat = mod_feat * mask.reshape(bs, c, l)
            mask_multimod_inputs[mod_name] = mod_feat
        return multimod_inputs

    @staticmethod
    def mask_prediction(predictor: nn.Module,
                        target: torch.Tensor,
                        fused_t_mask_feats: torch.Tensor, ):
        pred_out = predictor(fused_t_mask_feats)
        pred_loss = F.mse_loss(pred_out, target)
        return pred_loss


class SslNceFramework_EarlySum_VATT(torch.nn.Module):
    """Framework for multi-model self-supervised pretraining"""

    def __init__(self,
                 mod_names: List,
                 main_mod: str,
                 model_dim: int,
                 num_stack: int,

                 nce_args: DictConfig,
                 fom_args: DictConfig,
                 mask_args: DictConfig,
                 mask_latent_args: DictConfig,

                 audio_args: Optional[DictConfig] = None,
                 vision_args: Optional[DictConfig] = None,
                 tactile_args: Optional[DictConfig] = None,
                 ultrasonic_args: Optional[DictConfig] = None,
                 imu_args: Optional[DictConfig] = None,
                 force_args: Optional[DictConfig] = None,
                 current_args: Optional[DictConfig] = None,
                 thermal_args: Optional[DictConfig] = None,
                 depth_args: Optional[DictConfig] = None,
                 text_args: Optional[DictConfig] = None,

                 fusion_args: Optional[DictConfig] = None,
                 pos_emb_args: Optional[DictConfig] = None,
                 cross_time_trf_args: Optional[DictConfig] = None,

                 **kwargs
                 ):
        """

        Args:
             preprocess_audio_args: arguments for audio prepressing
             tokenization_audio: arguments for audio tokenization
             pe_audio: arguments for positional encoding for audio tokens
             encoder_audio_args: arguments for audio encoder(identity for earlycat/transformer for multi to one)
             preprocess_vision_args: arguments for vision prepressing
             tokenization_vision: arguments for vision tokenization
             pe_vision: arguments for positional encoding for vision tokens
             encoder_vision_args: arguments for vision encoder(identity for earlycat/transformer for multi to one)
             cross_time_trf_args: arguments for cross time transformer
             **kwargs:
        """
        super().__init__()

        self.mod_args = {
            "vision": vision_args,
            "tactile": tactile_args,
            "audio": audio_args,

            "ultrasonic": ultrasonic_args,
            "imu": imu_args,
            "force": force_args,
            "current": current_args,

            "thermal": thermal_args,
            "depth": depth_args,
            "text": text_args,
        }
        self.mod_names = mod_names
        self.main_mod = main_mod
        self.num_stack = num_stack

        self.fom_args = fom_args
        self.mask_args = mask_args
        self.mask_latent_args = mask_latent_args
        self.nce_args = nce_args

        if self.nce_args.norm == "batch":
            from src.models.utils.helpers import MyPermute
            self.nce_head_v_va = nn.Sequential(
                nn.Linear(model_dim, model_dim, bias=False),
                MyPermute([0, 2, 1]),
                nn.BatchNorm1d(momentum=0.9, eps=1e-5, num_features=model_dim),
                MyPermute([0, 2, 1]),
                nn.ReLU(),
                nn.Linear(model_dim, model_dim, bias=False),
                MyPermute([0, 2, 1]),
                nn.BatchNorm1d(momentum=0.9, eps=1e-5, num_features=model_dim),
                MyPermute([0, 2, 1]),
            )
            self.nce_head_a_va = nn.Linear(model_dim, model_dim)
            self.nce_head_va_vat = nn.Sequential(nn.ReLU(),
                                                 nn.Linear(model_dim, model_dim, bias=False),
                                                 MyPermute([0, 2, 1]),
                                                 nn.BatchNorm1d(momentum=0.9, eps=1e-5, num_features=model_dim),
                                                 MyPermute([0, 2, 1]),
                                                 )
            self.nce_head_t_vat = nn.Sequential(
                nn.Linear(model_dim, model_dim, bias=False),
                MyPermute([0, 2, 1]),
                nn.BatchNorm1d(momentum=0.9, eps=1e-5, num_features=model_dim),
                MyPermute([0, 2, 1]),
                nn.ReLU(),
                nn.Linear(model_dim, model_dim, bias=False),
                MyPermute([0, 2, 1]),
                nn.BatchNorm1d(momentum=0.9, eps=1e-5, num_features=model_dim),
                MyPermute([0, 2, 1]),
            )
        elif self.nce_args.norm == "layer":
            self.nce_head_v_va = nn.Sequential(
                nn.Linear(model_dim, model_dim, bias=False),
                nn.LayerNorm(model_dim),
                nn.ReLU(),
                nn.Linear(model_dim, model_dim, bias=False),
                nn.LayerNorm(model_dim),
            )
            self.nce_head_a_va = nn.Linear(model_dim, model_dim)
            self.nce_head_va_vat = nn.Sequential(nn.ReLU(),
                                                 nn.Linear(model_dim, model_dim, bias=False),
                                                 nn.LayerNorm(model_dim),
                                                 )
            self.nce_head_t_vat = nn.Sequential(
                nn.Linear(model_dim, model_dim, bias=False),
                nn.LayerNorm(model_dim),
                nn.ReLU(),
                nn.Linear(model_dim, model_dim, bias=False),
                nn.LayerNorm(model_dim),
            )

        self.mask_fusion_nce_proj = hydra.utils.instantiate(mask_args.mask_fusion_nce.proj_head) if \
            mask_args.mask_fusion_nce is not None else None

        self.mask_cross_time_trf_proj = hydra.utils.instantiate(mask_args.mask_cross_time_trf_nce.proj_head) if \
            mask_args.mask_cross_time_trf_nce is not None else None

        self.mask_latent_predictor = hydra.utils.instantiate(mask_args.mask_latent_prediction.predictor) if \
            mask_args.mask_latent_prediction is not None else None

        for mod_name, mod_args in self.mod_args.items():
            if mod_args is not None:
                self.__setattr__(f"preprocess_{mod_name}",
                                 hydra.utils.instantiate(mod_args[f"preprocess_{mod_name}_args"]))
                self.__setattr__(f"tokenization_{mod_name}",
                                 hydra.utils.instantiate(mod_args[f"tokenization_{mod_name}"]))
                self.__setattr__(f"positional_encoding_{mod_name}",
                                 hydra.utils.instantiate(mod_args[f"pe_{mod_name}"]))
                self.__setattr__(f"encoder_{mod_name}",
                                 hydra.utils.instantiate(mod_args[f"encoder_{mod_name}_args"]))

        self.fusion = hydra.utils.instantiate(fusion_args)

        self.cls = torch.nn.Parameter(torch.randn(1, 1, model_dim))
        self.cross_time_pos_emb = hydra.utils.instantiate(pos_emb_args)
        self.cross_time_trf = hydra.utils.instantiate(cross_time_trf_args)

        self.fom_classifier = ClassificationHead(model_dim=model_dim, num_classes=num_stack)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(model_dim, model_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(model_dim, model_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(model_dim, 3 ** 3),
        )
        self.aux_mlp = torch.nn.Linear(model_dim, 6)

        self.latent_mask_token = nn.ParameterDict({mod_name: torch.nn.Parameter(torch.randn([1, 1, model_dim]))
                                                   for mod_name in self.mod_names})

    def forward(self, multimod_inputs: Dict,
                mask: Optional[str] = "input_mask",
                task: Optional[str] = "imitation",
                mode: str = "train",
                ):
        assert mode in ["train", "val", "inference"]

        start_time = time.time()
        output = {"predict": {}, "ssl_losses": {}, "repr": {}}

        encoded_inputs = self.forward_modality_specific_encoding(multimod_inputs)

        if task == "repr":
            output["repr"]["encoded_inputs"] = encoded_inputs
            encoded_inputs = self.fusion(encoded_inputs)
            output["repr"]['fused_encoded_inputs'] = encoded_inputs
            x, attn_maps = self.forward_cross_time(encoded_inputs)
            output["repr"]['cross_time_repr'] = x
            agg_feat = x[:, 0, :]
            action_logits = self.mlp(agg_feat)
            xyzrpy = self.aux_mlp(agg_feat)
            output["predict"]["action_logits"] = action_logits
            output["predict"]["xyzrpy"] = xyzrpy
            output["time"] = time.time() - start_time
            return output

        if "bind" in task:
            nce_loss_dict = self.forward_nce(encoded_inputs)
            loss = nce_loss_dict['_loss']
            output["ssl_losses"]["cr_m_nce_loss"] = loss

        fused_t_feats = self.fusion(encoded_inputs)

        if mask == "input_mask":
            masked_multimod_inputs = self.random_masking(multimod_inputs=multimod_inputs,
                                                         masked_mod=self.mask_args.masked_mod,
                                                         mask_prob_args=self.mask_args.mask_prob_args,
                                                         mask_length_args=self.mask_args.mask_length_args, )
            encoded_masked_inputs = self.forward_modality_specific_encoding(masked_multimod_inputs)

            mask_pred_target = fused_t_feats.detach()
            fused_t_masked_feats = self.fusion(encoded_masked_inputs)

            if 'order' in task:
                fom_loss = self.forward_order_prediction(fused_t_masked_feats)
                output["ssl_losses"]["masked_fom_loss"] = fom_loss

            if 'fuse_nce' in task and self.mask_fusion_nce_proj is not None:
                temp = self.mask_args.mask_fusion_nce.temp
                mask_fusion_nce_loss = self.mask_cross_time_nce(self.mask_fusion_nce_proj,
                                                                fused_t_feats, fused_t_masked_feats, temp)
                output["ssl_losses"]["mask_fusion_nce_loss"] = mask_fusion_nce_loss['_loss']

            fused_t_feats, _ = self.forward_cross_time(fused_t_feats)
            agg_feat = fused_t_feats[:, 0, :]
            fused_t_feats = fused_t_feats[:, 1:, :]
            fused_t_masked_feats, _ = self.forward_cross_time(fused_t_masked_feats)
            fused_t_masked_feats = fused_t_masked_feats[:, 1:, :]

            if 'cross_time_nce' in task and self.mask_cross_time_trf_proj is not None:
                temp = self.mask_args.mask_cross_time_trf_nce.temp
                mask_cross_time_nce_loss = self.mask_cross_time_nce(self.mask_cross_time_trf_proj,
                                                                    fused_t_feats, fused_t_masked_feats, temp)
                output["ssl_losses"]["mask_cr_t_nce_loss"] = mask_cross_time_nce_loss['_loss']

            if 'recover' in task and self.mask_latent_predictor is not None:
                recover_loss = self.mask_prediction(self.mask_latent_predictor,
                                                    mask_pred_target, fused_t_masked_feats)
                output["ssl_losses"]["recover_loss"] = recover_loss

            if "imitation" in task:
                action_logits = self.mlp(agg_feat)
                xyzrpy = self.aux_mlp(agg_feat)
                output["predict"]["action_logits"] = action_logits
                output["predict"]["xyzrpy"] = xyzrpy

        elif mask == "latent_mask":
            encoded_masked_inputs, masked_index = self.random_latent_masking(multimod_inputs=encoded_inputs,
                                                                             mask_tokens=self.latent_mask_token,
                                                                             mask_prob=self.mask_latent_args.mask_prob,
                                                                             mask_length=self.mask_latent_args.mask_length, )

            mask_pred_target = fused_t_feats.detach()
            fused_t_masked_feats = self.fusion(encoded_masked_inputs)

            if 'order' in task:
                fom_loss = self.forward_order_prediction(fused_t_masked_feats)
                output["ssl_losses"]["masked_fom_loss"] = fom_loss

            if 'fuse_nce' in task and self.mask_fusion_nce_proj is not None:
                temp = self.mask_args.mask_fusion_nce.temp
                mask_fusion_nce_loss = self.mask_cross_time_nce(self.mask_fusion_nce_proj,
                                                                fused_t_feats, fused_t_masked_feats, temp)
                output["ssl_losses"]["mask_fusion_nce_loss"] = mask_fusion_nce_loss['_loss']

            fused_t_feats, _ = self.forward_cross_time(fused_t_feats)
            agg_feat = fused_t_feats[:, 0, :]
            fused_t_feats = fused_t_feats[:, 1:, :]
            fused_t_masked_feats, _ = self.forward_cross_time(fused_t_masked_feats)
            fused_t_masked_feats = fused_t_masked_feats[:, 1:, :]

            if 'cross_time_nce' in task and self.mask_cross_time_trf_proj is not None:
                temp = self.mask_args.mask_cross_time_trf_nce.temp
                mask_cross_time_nce_loss = self.mask_cross_time_nce(self.mask_cross_time_trf_proj,
                                                                    fused_t_feats, fused_t_masked_feats, temp)
                output["ssl_losses"]["mask_cr_t_nce_loss"] = mask_cross_time_nce_loss['_loss']

            if 'recover' in task and self.mask_latent_predictor is not None:
                recover_loss = self.mask_prediction(self.mask_latent_predictor,
                                                    mask_pred_target,
                                                    fused_t_masked_feats,
                                                    masks=masked_index,
                                                    loss=self.mask_latent_args.mask_latent_prediction.loss)
                output["ssl_losses"]["recover_loss"] = recover_loss

            if "imitation" in task:
                action_logits = self.mlp(agg_feat)
                xyzrpy = self.aux_mlp(agg_feat)
                output["predict"]["action_logits"] = action_logits
                output["predict"]["xyzrpy"] = xyzrpy

        elif mask is None:
            if "order" in task:
                fom_loss = self.forward_order_prediction(fused_t_feats)
                output["ssl_losses"]["fom_loss"] = fom_loss
            if "imitation" in task:
                fused_t_feats, _ = self.forward_cross_time(fused_t_feats)
                agg_feat = fused_t_feats[:, 0, :]
                action_logits = self.mlp(agg_feat)
                xyzrpy = self.aux_mlp(agg_feat)
                output["predict"]["action_logits"] = action_logits
                output["predict"]["xyzrpy"] = xyzrpy

        return output

    def forward_cross_time(self, x):
        cls = self.cls.expand(x.shape[0], self.cls.shape[1], self.cls.shape[2])
        x = torch.cat([cls, x], dim=1)
        x = self.cross_time_pos_emb(x)
        x, attn_maps = self.cross_time_trf(x)
        return x, attn_maps

    def forward_order_prediction(self, fused_t_feats: torch.Tensor, compute_loss=True):

        bs, num_frame, dim = fused_t_feats.shape

        target, shuffled_fused_t_feats = self.random_shuffle(fused_t_feats, reorder_prob=self.fom_args.reorder_prob)

        if all(element == -1 for element in target.reshape(bs * num_frame, )):
            return torch.zeros([])

        x, attn_maps = self.forward_cross_time(shuffled_fused_t_feats)
        x = x[:, 1:, :]
        x = self.fom_classifier(x)

        if compute_loss:
            x = x.view(bs * num_frame, num_frame)
            targets = target.view(x.shape[0])
            loss = F.cross_entropy(
                x, targets, ignore_index=-1,
                reduction='mean')
            return loss
        return x

    @staticmethod
    def random_shuffle(fused_t_feats: torch.Tensor, reorder_prob: float):
        bs, num_frame, dim = fused_t_feats.shape
        shuffled_orders = []
        targets = []
        for i in range(bs):
            shuffled_order, target = get_scatter_idx_target(list(range(num_frame)), reorder_prob=reorder_prob)
            shuffled_order = torch.tensor(shuffled_order, device=fused_t_feats.device)
            target = torch.tensor(target, device=fused_t_feats.device)
            shuffled_orders.append(shuffled_order)
            targets.append(target)
        targets = torch.stack(targets, dim=0)
        shuffled_orders = torch.stack(shuffled_orders, dim=0).unsqueeze(-1)  # [b, num_frame, 1]
        shuffled_orders = shuffled_orders.expand(-1, -1, dim)
        shuffled_fused_t_feats = torch.zeros_like(fused_t_feats,
                                                  dtype=fused_t_feats.dtype, device=fused_t_feats.device)
        shuffled_fused_t_feats = shuffled_fused_t_feats.scatter_(1, shuffled_orders, fused_t_feats)
        # a = fused_t_feats[:, :, 10]
        # b = shuffled_fused_t_feats[:, :, 10]
        return targets, shuffled_fused_t_feats

    def forward_nce(self, encoded_inputs, compute_loss=True):
        v_va_feats = self.nce_head_v_va(encoded_inputs["vision"])
        v_vat_feats = self.nce_head_va_vat(v_va_feats)

        bs, num_frame, dim = v_va_feats.shape
        v_va_feats = v_va_feats.reshape(bs, num_frame * dim)
        v_va_feats = v_va_feats[:, None, :]

        bs, num_frame, dim = v_vat_feats.shape
        v_vat_feats = v_vat_feats.reshape(bs, num_frame * dim)
        v_vat_feats = v_vat_feats[:, None, :]

        # Find positive example -> batch_size//2 away from the original example
        pos_mask = torch.eye(bs, bs, dtype=torch.bool, device=v_vat_feats.device)
        # InfoNCE loss
        nll_sum = torch.zeros([])
        for name, feat in encoded_inputs.items():
            if name == "audio":
                side_feat = self.nce_head_a_va(feat)
                side_feat = side_feat.reshape(bs, num_frame * dim)
                cos_sim = F.cosine_similarity(v_va_feats, side_feat[None, :, :], dim=-1)
                cos_sim = cos_sim / self.nce_args.temp
                nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
                nll_ = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=0)
                nll = nll.mean()
                nll_ = nll_.mean()
                nll_sum = nll_sum + nll + nll_
            elif name == "tactile":
                side_feat = self.nce_head_t_vat(feat)
                side_feat = side_feat.reshape(bs, num_frame * dim)
                cos_sim = F.cosine_similarity(v_vat_feats, side_feat[None, :, :], dim=-1)
                cos_sim = cos_sim / self.nce_args.temp
                nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
                nll_ = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=0)
                nll = nll.mean()
                nll_ = nll_.mean()
                nll_sum = nll_sum + nll + nll_
        nce_loss_dict = {
            '_loss': nll_sum,
        }

        return nce_loss_dict

    @staticmethod
    def mask_cross_time_nce(proj: nn.Module,
                            fused_t_feats: torch.Tensor, fused_t_masked_feats: torch.Tensor,
                            temp: float, compute_loss=True):
        proj_fused_t_feats = proj(fused_t_feats)
        proj_fused_t_masked_feats = proj(fused_t_masked_feats)

        bs, num_frame, dim = proj_fused_t_feats.shape
        proj_fused_t_feats = proj_fused_t_feats.reshape(bs, num_frame * dim)
        proj_fused_t_feats = proj_fused_t_feats[:, None, :]

        bs, num_frame, dim = proj_fused_t_masked_feats.shape
        proj_fused_t_masked_feats = proj_fused_t_masked_feats.reshape(bs, num_frame * dim)
        proj_fused_t_masked_feats = proj_fused_t_masked_feats[None, :, :]

        pos_mask = torch.eye(bs, bs, dtype=torch.bool, device=proj_fused_t_feats.device)
        # InfoNCE loss
        nll_sum = torch.zeros([])
        cos_sim = F.cosine_similarity(proj_fused_t_feats, proj_fused_t_masked_feats, dim=-1)
        cos_sim = cos_sim / temp
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll_ = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=0)
        nll = nll.mean()
        nll_ = nll_.mean()
        nll_sum = nll_sum + nll + nll_

        nce_loss_dict = {
            '_loss': nll_sum,
        }

        return nce_loss_dict

    def forward_modality_specific_encoding(self, multimod_inputs: Dict, ):
        encoded_inputs = {}

        # process of vision signal
        if "vision" in multimod_inputs.keys():
            vision_signal = multimod_inputs["vision"]
            batch_size, num_frame, c_v, h_v, w_v = vision_signal.shape
            short_window_len = self.mod_args["vision"].short_window_len
            assert num_frame % short_window_len == 0
            num_stack = num_frame // short_window_len
            assert num_stack == self.num_stack
            vision_signal = torch.reshape(vision_signal, (-1, c_v, h_v, w_v))
            vision_signal = self.preprocess_vision(vision_signal)
            vision_signal = torch.reshape(vision_signal, (-1, short_window_len, c_v, h_v, w_v))
            vision_signal = self.tokenization_vision(vision_signal)  # vit
            if type(vision_signal) == tuple:
                vision_signal, attn_map = vision_signal
            if len(vision_signal.shape) == 2:
                vision_signal = vision_signal.reshape(vision_signal.shape[0], 1, vision_signal.shape[1])
                # [B*N, 256] -> [B*N, 1, 256]
            vision_signal = vision_signal.view(batch_size, num_stack * vision_signal.shape[-2], vision_signal.shape[-1])
            # [B*N, 1, 256] -> [B, N, 256]
            vision_signal = self.positional_encoding_vision(vision_signal)
            vision_signal = self.encoder_vision(vision_signal)
            if type(vision_signal) == tuple:
                vision_signal, attn_vision = vision_signal
            encoded_inputs["vision"] = vision_signal

        # process of audio signal
        if "audio" in multimod_inputs.keys():
            audio_signal = multimod_inputs["audio"]
            audio_signal = self.preprocess_audio(audio_signal)
            # [B, 1, 40K] -> [B, 64, 251]
            audio_signal = self.tokenization_audio(audio_signal)
            # [B*N, 1, 256] -> [B, N, 256]
            audio_signal = self.positional_encoding_audio(audio_signal)
            audio_signal = self.encoder_audio(audio_signal)
            if type(audio_signal) == tuple:
                audio_signal, attn_audio = audio_signal
            encoded_inputs["audio"] = audio_signal

        # process of tactile signal(similar to vision signal)
        if "tactile" in multimod_inputs.keys():
            tactile_signal = multimod_inputs["tactile"]
            batch_size, num_frame, c_v, h_v, w_v = tactile_signal.shape
            short_window_len = self.mod_args["tactile"].short_window_len
            assert num_frame % short_window_len == 0
            num_stack = num_frame // short_window_len
            assert num_stack == self.num_stack
            tactile_signal = torch.reshape(tactile_signal, (-1, c_v, h_v, w_v))
            tactile_signal = self.preprocess_tactile(tactile_signal)
            tactile_signal = torch.reshape(tactile_signal, (-1, short_window_len, c_v, h_v, w_v))
            tactile_signal = self.tokenization_tactile(tactile_signal)  # vit
            if type(tactile_signal) == tuple:
                tactile_signal, attn_map = tactile_signal
            if len(tactile_signal.shape) == 2:
                tactile_signal = tactile_signal.reshape(tactile_signal.shape[0], 1, tactile_signal.shape[1])
                # [B*N, 256] -> [B*N, 1, 256]
            tactile_signal = tactile_signal.view(batch_size, num_stack * tactile_signal.shape[-2],
                                                 tactile_signal.shape[-1])
            # [B*N, 1, 256] -> [B, N, 256]
            tactile_signal = self.positional_encoding_tactile(tactile_signal)
            tactile_signal = self.encoder_tactile(tactile_signal)
            if type(tactile_signal) == tuple:
                tactile_signal, attn_tactile = tactile_signal
            encoded_inputs["tactile"] = tactile_signal

        return encoded_inputs

    @staticmethod
    def random_masking(masked_mod: List,
                       mask_prob_args: DictConfig,
                       mask_length_args: DictConfig,
                       multimod_inputs: Dict):
        mask_multimod_inputs = {}
        for mod_name, mod_feat in multimod_inputs.items():
            if mod_name in masked_mod:
                mask_porb = mask_prob_args[mod_name]
                mask_length = mask_length_args[mod_name]
                if mod_name in ["vision", "tactile"]:
                    # mask whole image
                    bs, num_frame, c, h, w = mod_feat.shape
                    mask = torch.tensor(get_mask_sequence1d(seq_len=bs * num_frame,
                                                            mask_prob=mask_porb,
                                                            mask_length=mask_length,
                                                            mask_mark=0,
                                                            unmask_mark=1, ), device=mod_feat.device)
                    mod_feat = mod_feat * mask.reshape(bs, num_frame, 1, 1, 1)
                elif mod_name in ["audio", "ultrasonic", "imu", "force", "current", ]:
                    # channel-wise variant
                    bs, c, l = mod_feat.shape
                    mask = torch.tensor(get_mask_sequence1d(seq_len=bs * c * l,
                                                            mask_prob=mask_porb,
                                                            mask_length=mask_length,
                                                            mask_mark=0,
                                                            unmask_mark=1, ), device=mod_feat.device)
                    mod_feat = mod_feat * mask.reshape(bs, c, l)
            mask_multimod_inputs[mod_name] = mod_feat
        return multimod_inputs

    @staticmethod
    def random_latent_masking(multimod_inputs: Dict,
                              mask_tokens: nn.ParameterDict,
                              mask_prob: float,
                              mask_length: int, ):
        mask_multimod_inputs = {}
        masked_index = {}
        for mod_name, mod_feat in multimod_inputs.items():
            bs, num_frame, dim = mod_feat.shape
            mask = torch.tensor(get_mask_sequence1d(seq_len=bs * num_frame,
                                                    mask_prob=mask_prob,
                                                    mask_length=mask_length,
                                                    mask_mark=0,
                                                    unmask_mark=1, ), device=mod_feat.device).reshape(bs, num_frame, 1)
            mod_feat = mod_feat * mask + mask_tokens[mod_name].reshape(1, 1, -1) * (1 - mask)
            mask_multimod_inputs[mod_name] = mod_feat
            masked_index[mod_name] = mask
        return multimod_inputs, masked_index

    @staticmethod
    def mask_prediction(predictor: nn.Module,
                        target: torch.Tensor,
                        fused_t_mask_feats: torch.Tensor,
                        masks: Optional[dict] = None,
                        loss: str = "mse",):
        """

        Args:
            predictor:
            target:
            fused_t_mask_feats:
            masks: shape [B, S ,1]
            loss:

        Returns:

        """
        pred_out = predictor(fused_t_mask_feats)
        if loss == "mse":
            loss_fn = mse_fn
        elif loss == "cosine":
            loss_fn = cosine_loss_fn

        if masks is None:
            pred_loss = loss_fn(pred_out, target)
        else:
            # can not filter the zero value as number of masked ele varies alone dif dim
            mask = 1
            for mod_name, mode_mask in masks.items():
                mask = mask * mode_mask
            mask = 1 - mask
            pred_loss = loss_fn(pred_out, target, mask)

        return pred_loss
