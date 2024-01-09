import copy

import copyreg

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention
from src.models.vit_implementations import Vit_Classifier, Vit_Classifier_Mel, LrnEmb_Agg_Trf
from src.models.utils.mel_spec import MelSpec
from src.models.utils.header import ClassificationHead
from src.models.utils.helpers import shuffle_sequence, get_mask_sequence1d
from omegaconf import DictConfig, OmegaConf
import hydra
from typing import Optional, Dict, List
from types import SimpleNamespace


class VisionAudioFusion_EarlySum(torch.nn.Module):
    """Vision audio fusion model for vision and audio signal from see_hear_feel using Early Summation"""

    def __init__(self,
                 mod_names: List,
                 model_dim: int,
                 num_stack: int,

                 nce_args: DictConfig,
                 fom_args: DictConfig,
                 mask_args: DictConfig,

                 audio_args: Optional[DictConfig] = None,
                 vision_args: Optional[DictConfig] = None,
                 tactile_args: Optional[DictConfig] = None,

                 fusion_args: Optional[DictConfig] = None,
                 pos_emb_args: Optional[DictConfig] = None,
                 cross_time_trf_args: Optional[DictConfig] = None,

                 # preprocess_audio_args: DictConfig,
                 # tokenization_audio: DictConfig,
                 # pe_audio: DictConfig,
                 # encoder_audio_args: DictConfig,
                 #
                 # preprocess_vision_args: DictConfig,
                 # tokenization_vision: DictConfig,
                 # pe_vision: DictConfig,
                 # encoder_vision_args: DictConfig,
                 #
                 # preprocess_tactile_args: DictConfig,
                 # tokenization_tactile: DictConfig,
                 # pe_tactile: DictConfig,
                 # encoder_tactile_args: DictConfig,

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

        self.ModalityType = [
            "vision",
            "tactile",
            "audio",

            "ultrasonic",
            "imu",
            "force",
            "current",

            "thermal",
            "depth",
            "text",
        ]
        self.mod_names = mod_names

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

        if audio_args is not None:
            self.preprocess_audio = hydra.utils.instantiate(audio_args.preprocess_audio_args)
            self.tokenization_audio = hydra.utils.instantiate(audio_args.tokenization_audio)
            self.positional_encoding_audio = hydra.utils.instantiate(audio_args.pe_audio)
            self.encoder_audio = hydra.utils.instantiate(audio_args.encoder_audio_args)

        if vision_args is not None:
            self.preprocess_vision = hydra.utils.instantiate(vision_args.preprocess_vision_args)
            self.tokenization_vision = hydra.utils.instantiate(vision_args.tokenization_vision)
            self.positional_encoding_vision = hydra.utils.instantiate(vision_args.pe_vision)
            self.encoder_vision = hydra.utils.instantiate(vision_args.encoder_vision_args)

        if tactile_args is not None:
            self.preprocess_tactile = hydra.utils.instantiate(tactile_args.preprocess_tactile_args)
            self.tokenization_tactile = hydra.utils.instantiate(tactile_args.tokenization_tactile)
            self.positional_encoding_tactile = hydra.utils.instantiate(tactile_args.pe_tactile)
            self.encoder_tactile = hydra.utils.instantiate(tactile_args.encoder_tactile_args)

        self.cls = torch.nn.Parameter(torch.randn(1, 1, model_dim))

        self.fusion = hydra.utils.instantiate(fusion_args)
        # self.register_parameter('vision_gamma', torch.nn.Parameter(torch.randn((1, 1, model_dim))))
        # self.register_parameter('audio_gamma', torch.nn.Parameter(torch.randn((1, 1, model_dim))))
        # self.register_parameter('tactile_gamma', torch.nn.Parameter(torch.randn((1, 1, model_dim))))

        self.pos_emb = hydra.utils.instantiate(pos_emb_args)
        self.cross_time_trf = hydra.utils.instantiate(cross_time_trf_args)

        self.fom_classifier = ClassificationHead(model_dim=model_dim, num_classes=num_stack)

    def forward(self, multimod_inputs: Dict,
                mask: bool = True,
                task: tuple = ("repr", 'order', 'fuse_nce', 'cross_time_nce', 'predict'),
                main_mod: Optional[str] = "vision",
                ):

        encoded_inputs = self.forward_modality_specific_encoding(multimod_inputs)
        nce_loss_dict = self.forward_nce(encoded_inputs)
        loss = nce_loss_dict['_loss']

        if task == "repr":
            repr = {'encoded_inputs': encoded_inputs}
            encoded_inputs = self.fusion(encoded_inputs)
            repr['fused_encoded_inputs'] = encoded_inputs
            x, attn_maps = self.forward_cross_time(encoded_inputs)
            repr['cross_time_repr'] = x
            return repr

        if mask:
            mask_loss = loss
            masked_multimod_inputs = self.random_masking(multimod_inputs=multimod_inputs,
                                                         masked_mod=self.mask_args.masked_mod,
                                                         mask_prob_args=self.mask_args.mask_prob_args,
                                                         mask_length_args=self.mask_args.mask_length_args,)
            encoded_masked_inputs = self.forward_modality_specific_encoding(masked_multimod_inputs)

            fused_t_feats = self.fusion(encoded_inputs)
            mask_pred_target = fused_t_feats.detach()
            fused_t_masked_feats = self.fusion(encoded_masked_inputs)

            if task == 'order':
                fom_loss = self.forward_order_prediction(fused_t_masked_feats)
                mask_loss += fom_loss

            if self.mask_fusion_nce_proj is not None and 'fuse_nce' in task:
                temp = self.mask_args.mask_fusion_nce.temp
                mask_fusion_nce_loss = self.mask_cross_time_nce(self.mask_fusion_nce_proj,
                                                                fused_t_feats, fused_t_masked_feats, temp)
                mask_loss += mask_fusion_nce_loss['_loss']

            fused_t_feats, _ = self.forward_cross_time(fused_t_feats)
            fused_t_feats = fused_t_feats[:, 1:, :]
            fused_t_masked_feats, _ = self.forward_cross_time(fused_t_masked_feats)
            fused_t_masked_feats = fused_t_masked_feats[:, 1:, :]

            if self.mask_cross_time_trf_proj is not None and 'cross_time_nce' in task:
                temp = self.mask_args.mask_cross_time_trf_nce.temp
                mask_cross_time_nce_loss = self.mask_cross_time_nce(self.mask_cross_time_trf_proj,
                                                                    fused_t_feats, fused_t_masked_feats, temp)
                mask_loss += mask_cross_time_nce_loss['_loss']
            if self.mask_latent_predictor is not None and 'predict' in task:
                pred_loss = self.mask_prediction(self.mask_latent_predictor,
                                                 mask_pred_target, fused_t_masked_feats)
                mask_loss += pred_loss
            return mask_loss
        else:
            fused_t_feats = self.fusion(encoded_inputs)
            return loss + self.forward_order_prediction(fused_t_feats)

    def forward_cross_time(self, x):
        cls = self.cls.expand(x.shape[0], self.cls.shape[1], self.cls.shape[2])
        x = torch.cat([cls, x], dim=1)
        x = self.pos_emb(x)
        x, attn_maps = self.cross_time_trf(x)
        return x, attn_maps

    def forward_order_prediction(self, fused_t_feats: torch.Tensor, compute_loss=True):

        bs, num_stack, dim = fused_t_feats.shape

        target, shuffled_fused_t_feats = self.ranom_shuffle(fused_t_feats, reorder_prob=self.fom_args.reorder_prob)

        x, attn_maps = self.forward_cross_time(shuffled_fused_t_feats)
        x = x[:, 1:, :]
        x = self.fom_classifier(x)

        if compute_loss:
            x = x.view(bs * num_stack, num_stack)
            targets = target.view(x.shape[0])
            loss = F.cross_entropy(
                x, targets, ignore_index=-1,
                reduction='mean')
            return loss
        return x

    @staticmethod
    def ranom_shuffle(fused_t_feats: torch.Tensor, reorder_prob: float):
        bs, num_stack, dim = fused_t_feats.shape
        order = list(range(num_stack))
        shuffled_order, target = shuffle_sequence(order, reorder_prob=reorder_prob)
        shuffled_order = torch.tensor(shuffled_order, device=fused_t_feats.device)
        target = torch.tensor(target, device=fused_t_feats.device)
        shuffled_fused_t_feats = fused_t_feats.index_select(1, shuffled_order)
        return target, shuffled_fused_t_feats

    def forward_nce(self, encoded_inputs, compute_loss=True):
        main_feats = self.nce_head(encoded_inputs[self.nce_args.main_mod])
        bs, num_stack, dim = main_feats.shape
        main_feats = main_feats.reshape(bs, num_stack * dim)
        main_feats = main_feats[:, None, :]

        # Find positive example -> batch_size//2 away from the original example
        pos_mask = torch.eye(bs, bs, dtype=torch.bool, device=main_feats.device)
        # InfoNCE loss
        nll_sum = torch.zeros([])
        for name, feat in encoded_inputs.items():
            if name != self.nce_args.main_mod:
                side_feat = self.nce_head(feat)
                side_feat = side_feat.reshape(bs, num_stack * dim)
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

        bs, num_stack, dim = proj_fused_t_feats.shape
        proj_fused_t_feats = proj_fused_t_feats.reshape(bs, num_stack * dim)
        proj_fused_t_feats = proj_fused_t_feats[:, None, :]

        bs, num_stack, dim = proj_fused_t_masked_feats.shape
        proj_fused_t_masked_feats = proj_fused_t_masked_feats.reshape(bs, num_stack * dim)
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
            batch_size, num_stack, c_v, h_v, w_v = vision_signal.shape
            vision_signal = torch.reshape(vision_signal, (-1, c_v, h_v, w_v))
            vision_signal = self.preprocess_vision(vision_signal)
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
            batch_size, num_stack, c_v, h_v, w_v = tactile_signal.shape
            tactile_signal = torch.reshape(tactile_signal, (-1, c_v, h_v, w_v))
            tactile_signal = self.preprocess_tactile(tactile_signal)
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
                    bs, num_stack, c, h, w = mod_feat.shape
                    mask = torch.tensor(get_mask_sequence1d(seq_len=bs * num_stack,
                                                            mask_prob=mask_porb,
                                                            mask_length=mask_length,
                                                            mask_mark=0,
                                                            unmask_mark=1, ), device=mod_feat.device)
                    mod_feat = mod_feat * mask.reshape(bs, num_stack, 1, 1, 1)
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
