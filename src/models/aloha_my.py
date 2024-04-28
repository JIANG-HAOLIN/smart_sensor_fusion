# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from omegaconf import DictConfig, OmegaConf
import hydra
from typing import Optional, Dict, List

from src.models.utils.positional_encoding import StandardPositionalEncoding
from src.models.ssl_nce_framework import SslNceFramework_EarlySum_VATT

import torch
from torch import nn
from torch.autograd import Variable

import numpy as np

import time

from utils.quaternion import q_exp_map, q_log_map, recover_pose_from_quat_real_delta, exp_map_seq, log_map_seq, exp_map

import copy

import copyreg

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention
from src.models.vit_implementations import Vit_Classifier, Vit_Classifier_Mel, LrnEmb_Agg_Trf
from src.models.utils.mel_spec import MelSpec
from src.models.utils.header import ClassificationHead, MLPHead
from src.models.utils.helpers import get_scatter_idx_target, get_mask_sequence1d
from omegaconf import DictConfig, OmegaConf
import hydra
from typing import Optional, Dict, List
from types import SimpleNamespace
import time
from src.models.utils.helpers import cosine_loss_fn, mse_fn
from src.models.utils.embeddings import LearnablePosEmb


class Transformer(nn.Module):

    def __init__(self, encoder,
                 d_model=512):
        super().__init__()

        self.encoder = encoder

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=4 * d_model,
            dropout=0.,
            activation='gelu',
            batch_first=True,
            norm_first=True  # important for stability
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=8
        )

        self.d_model = d_model


    def forward(self, multi_mod_input, mask, query_embed, latent_input=None, proprio_input=None,
                additional_pos_embed=None,
                mask_type: Optional[str] = "input_mask",
                task: Optional[str] = "imitation",
                mode: str = "train",
                ):
        """
        multi_mod_input should begin with batch size
        all embed should have size: [1, S, D]
        """

        bs = latent_input.shape[0]
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # input for the transformer-decoder !!
        # mask = mask.flatten(1)

        additional_pos_embed = additional_pos_embed.unsqueeze(1).repeat(1, bs, 1)[:1]  # seq, bs, dim
        # seq+2, bs, dim(positional embedding for input(multi_mod_input) for trm-encoder!!)

        addition_input = torch.stack([latent_input], dim=0)
        addition_input = addition_input + additional_pos_embed

        tgt = torch.zeros_like(query_embed)

        trm_enc_out = self.encoder(multi_mod_input,
                                   multi_mod_input_key_padding_mask=mask,
                                   mask=mask_type,
                                   task=task,
                                   mode=mode,
                                   additional_input=addition_input.permute(1, 0, 2))
        memory = trm_enc_out["repr"]['cross_time_repr'].permute(1, 0, 2)
        pos_embed = trm_enc_out["pos_emb"]["cross_time_emb"].permute(1, 0, 2).repeat(1, bs, 1)
        pos_embed = torch.cat([pos_embed, additional_pos_embed], dim=0)

        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        hs = hs.transpose(1, 2)
        return {"action_decoder_out": hs,
                "obs_encoder_out": trm_enc_out,
                }




def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """

    def __init__(self, style_encoder, transformer, action_dim, num_queries, pose_dim):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            action_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()

        self.num_queries = num_queries
        self.transformer = transformer
        self.style_encoder = style_encoder
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.input_proj_robot_state = nn.Linear(pose_dim, hidden_dim)

        # encoder extra parameters
        self.latent_dim = 32  # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim)  # extra cls token embedding
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)  # project action to embedding
        self.encoder_joint_proj = nn.Linear(pose_dim, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)  # project hidden state to latent std, var
        self.pos_table = get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim) # [CLS], qpos, a_seq
        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)  # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)  # learned position embedding for proprio and latent

    def forward(self, qpos, multi_mod_input, actions=None, is_pad=None,
                mask=None,
                mask_type="input_mask",
                task="imitation",
                mode="train",
                env_state=None,
                ):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None  # train or val
        bs, _ = qpos.shape
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, dim=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight  # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, dim=0).repeat(bs, 1, 1)  # (bs, 1, hidden_dim)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], dim=1)  # (bs, seq+2, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device)  # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], dim=1)  # (bs, seq+2)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+2, 1, hidden_dim)
            # query model
            encoder_output = self.style_encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0]  # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        # Image observation features and position embeddings

        # proprioception features
        proprio_input = self.input_proj_robot_state(qpos)
        # fold camera dimension into width dimension
        multi_mod_input["vision"] = multi_mod_input["vision"]["v_fix"]
        output = self.transformer(
            multi_mod_input=multi_mod_input,
            mask=mask,

            query_embed=self.query_embed.weight,
            latent_input=latent_input,
            proprio_input=proprio_input,
            additional_pos_embed=self.additional_pos_embed.weight,

            mask_type=mask_type,
            task=task,
            mode=mode,
        )
        hs = output["action_decoder_out"][0]
        a_hat = self.action_head(hs)  # bs, seq_len, action_dim
        is_pad_hat = self.is_pad_head(hs)
        return {"vae_output": [a_hat, is_pad_hat, [mu, logvar]],
                "obs_encoder_out": output["obs_encoder_out"],
                }

    def rollout(self, qpos, multimod_inputs, env_state, actions=None, is_pad=None,
                all_time_position=None,
                all_time_orientation=None,
                t=None,
                args=None,
                v_scale=None,
                inference_type=None,
                num_queries=None,
                normalizer=None,
                **kwargs):
        output = self.forward(qpos,
                              multimod_inputs,
                              actions=actions,
                              is_pad=is_pad,
                              mask=None,
                              mask_type="None",
                              task="repr",
                              mode="val",
                              env_state=None, )
        a_hat, is_pad_hat, (mu, logvar) = output["vae_output"]
        og_a_hat = a_hat
        a_hat, gripper = a_hat[:, :, :-1], a_hat[:, :, -1:]
        qpos = normalizer.denormalize(qpos[:, :-1], "target_glb_pos_ori").float()
        qpos_gripper = normalizer.denormalize(qpos[:, -1:], "gripper").float()
        qpos = torch.cat([qpos, qpos_gripper], dim=-1)
        gripper = normalizer.denormalize(gripper[:, :num_queries, :], "gripper")

        gripper = gripper.squeeze(0).detach().cpu().numpy()
        if a_hat.shape[-1] == 6:
            if inference_type == "real_delta_target":
                a_hat = normalizer.denormalize(a_hat[:, :num_queries, :], "target_real_delta")
                base = exp_map(qpos[:, :-1].squeeze(0).detach().cpu().numpy(), np.array([0, 0, 0, 0, 1, 0, 0]))
                v = a_hat.squeeze(0).detach().cpu().numpy() * v_scale
                out_chunk = recover_pose_from_quat_real_delta(v, base)

            elif inference_type == "real_delta_source":
                a_hat = normalizer.denormalize(a_hat[:, :num_queries, :], "source_real_delta")
                base = exp_map(qpos[:, :-1].squeeze(0).detach().cpu().numpy(), np.array([0, 0, 0, 0, 1, 0, 0]))
                v = a_hat.squeeze(0).detach().cpu().numpy() * v_scale
                out_chunk = recover_pose_from_quat_real_delta(v, base)

            elif inference_type == "direct_vel":
                a_hat = normalizer.denormalize(a_hat[:, :num_queries, :], "direct_vel")
                base = exp_map(qpos[:, :-1].squeeze(0).detach().cpu().numpy(), np.array([0, 0, 0, 0, 1, 0, 0]))
                v = a_hat.squeeze(0).detach().cpu().numpy() * v_scale
                out_chunk = recover_pose_from_quat_real_delta(v, base)

            elif inference_type == "position":
                a_hat = normalizer.denormalize(a_hat[:, :num_queries, :], "target_glb_pos_ori")
                v = a_hat.squeeze(0).detach().cpu().numpy()
                out_chunk = exp_map_seq(v, np.array([0, 0, 0, 0, 1, 0, 0]))

        else:
            raise RuntimeError("action has only 6 dim, no gripper dim!")

        out_chunk = np.concatenate([gripper, out_chunk, ], axis=-1)
        out_position = torch.from_numpy(out_chunk[:, :4])
        out_orientation = torch.from_numpy(out_chunk[:, 4:])
        all_time_orientation[[t], t:t + num_queries] = out_orientation.float().to(args.device)
        orientation_for_curr_step = all_time_orientation[:, t]
        actions_populated = torch.all(orientation_for_curr_step != 0, axis=1)
        orientation_for_curr_step = orientation_for_curr_step[actions_populated]

        k = 0.01
        exp_weights = np.exp(-k * np.arange(len(orientation_for_curr_step)))
        exp_weights = exp_weights / exp_weights.sum()
        exp_weights = (exp_weights[::-1]).copy()  # [::-1] could lead to negative strides

        weights = np.expand_dims(exp_weights, axis=0)
        raw_orientation = orientation_for_curr_step[0].detach().cpu().numpy()
        orientation = orientation_for_curr_step.permute(1, 0).detach().cpu().numpy()
        for i in range(5):
            tangent_space_vector = q_log_map(orientation, raw_orientation)
            tangent_space_vector = np.sum(tangent_space_vector * weights, axis=1, keepdims=True)
            raw_orientation = q_exp_map(tangent_space_vector, raw_orientation)[:, 0]

        all_time_position[[t], t:t + num_queries] = out_position.float().to(args.device)
        position_for_curr_step = all_time_position[:, t]
        actions_populated = torch.all(position_for_curr_step != 0, axis=1)
        position_for_curr_step = position_for_curr_step[actions_populated]
        weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
        raw_action = (position_for_curr_step * weights).sum(dim=0, keepdim=True)
        raw_position = raw_action.squeeze(0).cpu().numpy()
        return out_chunk, np.concatenate(
            [raw_position, raw_orientation]), all_time_position, all_time_orientation, og_a_hat


def build_encoder(args):
    d_model = args.hidden_dim  # 256
    dropout = args.dropout  # 0.1
    nhead = args.nheads  # 8
    dim_feedforward = args.dim_feedforward  # 2048
    num_encoder_layers = args.enc_layers  # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm  # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build_detrvae(style_encoder: DictConfig,
                  action_decoder: DictConfig,
                  obs_encoder: DictConfig,
                  action_dim: int,
                  pose_dim: int, ):
    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image

    transformer = Transformer(
        encoder=hydra.utils.instantiate(obs_encoder, _recursive_=False),
        d_model=action_decoder.hidden_dim,
        dropout=action_decoder.dropout,
        nhead=action_decoder.nheads,
        dim_feedforward=action_decoder.dim_feedforward,
        num_encoder_layers=action_decoder.enc_layers,
        num_decoder_layers=action_decoder.dec_layers,
        normalize_before=action_decoder.pre_norm,
        return_intermediate_dec=True,
    )

    style_encoder = build_encoder(style_encoder)

    model = DETRVAE(
        style_encoder,
        transformer,
        action_dim=action_dim,
        pose_dim=pose_dim,
        num_queries=action_decoder.num_queries,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    return model


class MyAloha(torch.nn.Module):

    def __init__(self,
                 mod_names: List,
                 main_mod: str,
                 model_dim: int,
                 num_stack: int,

                 nce_args: DictConfig,
                 fom_args: DictConfig,
                 mask_args: DictConfig,
                 mask_fusion_nce: DictConfig,
                 mask_cross_time_trf_nce: DictConfig,
                 mask_latent_prediction: DictConfig,

                 audio_args: Optional[DictConfig] = None,
                 vision_args: Optional[DictConfig] = None,
                 tactile_args: Optional[DictConfig] = None,
                 qpos_args: Optional[DictConfig] = None,
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
            "qpos": qpos_args,

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
        self.mask_fusion_nce = mask_fusion_nce
        self.mask_cross_time_trf_nce = mask_cross_time_trf_nce
        self.mask_latent_prediction = mask_latent_prediction
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
            self.nce_head_qpos_va = nn.Linear(model_dim, model_dim)
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
            self.nce_head_qpos_va = nn.Linear(model_dim, model_dim)
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

        self.mask_fusion_nce_proj = hydra.utils.instantiate(mask_fusion_nce.proj_head) if \
            mask_fusion_nce is not None else None

        self.mask_cross_time_trf_proj = hydra.utils.instantiate(mask_cross_time_trf_nce.proj_head) if \
            mask_cross_time_trf_nce is not None else None

        self.mask_latent_predictor = hydra.utils.instantiate(mask_latent_prediction.predictor) if \
            mask_latent_prediction is not None else None

        self.fom_classifier = hydra.utils.instantiate(fom_args.predictor) if \
            fom_args is not None else None

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

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(model_dim, model_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(model_dim, model_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(model_dim, 3 ** 3),
        )
        self.aux_mlp = nn.Sequential(nn.Linear(model_dim, 6))

        self.latent_mask_token = nn.ParameterDict({mod_name: torch.nn.Parameter(torch.randn([1, 1, model_dim]))
                                                   for mod_name in self.mod_names})

        self.output_dim = model_dim

    def forward(self, multimod_inputs: Dict,
                mask: Optional[str] = "input_mask",
                task: Optional[str] = "imitation",
                mode: str = "train",
                additional_input: Optional[torch.Tensor] = None,
                multi_mod_input_key_padding_mask: Optional[torch.Tensor] = None,
                ):
        assert mode in ["train", "val", "inference"]

        start_time = time.time()
        output = {"predict": {}, "ssl_losses": {}, "repr": {}, "pos_emb": {}}

        encoded_inputs = self.forward_modality_specific_encoding(multimod_inputs)

        if task == "repr":
            output["repr"]["encoded_inputs"] = encoded_inputs
            encoded_inputs = self.fusion(encoded_inputs)
            output["repr"]['fused_encoded_inputs'] = encoded_inputs
            x, attn_maps = self.forward_cross_time(encoded_inputs, additional_input)
            output["repr"]['cross_time_repr'] = x
            agg_feat = x[:, 0, :]
            action_logits = self.mlp(agg_feat)
            xyzrpy = self.aux_mlp(agg_feat)
            output["predict"]["action_logits"] = action_logits
            output["predict"]["xyzrpy"] = xyzrpy
            output["time"] = time.time() - start_time
            output["pos_emb"]["cross_time_emb"] = self.cross_time_pos_emb.pe[:, :self.num_stack + 1, :]
            return output

        if "bind" in task:
            nce_loss_dict = self.forward_nce(encoded_inputs)
            loss = nce_loss_dict['_loss']
            output["ssl_losses"]["cr_m_nce_loss"] = loss

        fused_t_feats = self.fusion(encoded_inputs)

        if mask == "latent_mask":
            encoded_masked_inputs, masked_index = self.random_latent_masking(multimod_inputs=encoded_inputs,
                                                                             mask_tokens=self.latent_mask_token,
                                                                             mask_prob=self.mask_args.mask_prob.latent,
                                                                             mask_length=self.mask_args.mask_length.latent, )

            mask_pred_target = multimod_inputs["qpos"]
            fused_t_masked_feats = self.fusion(encoded_masked_inputs)

            if 'order' in task:
                fom_loss = self.forward_order_prediction(fused_t_masked_feats,
                                                         compute_loss=True,
                                                         additional_input=additional_input)
                output["ssl_losses"]["masked_fom_loss"] = fom_loss

            fused_t_feats, _ = self.forward_cross_time(fused_t_feats, additional_input)
            output["repr"]['cross_time_repr'] = fused_t_feats
            agg_feat = fused_t_feats[:, 0, :]
            fused_t_feats = fused_t_feats[:, 1: 1 + self.num_stack, :]
            fused_t_masked_feats, _ = self.forward_cross_time(fused_t_masked_feats, additional_input)
            fused_t_masked_feats = fused_t_masked_feats[:, 1: 1 + self.num_stack, :]

            if 'recover' in task and self.mask_latent_predictor is not None:
                recover_loss = self.mask_prediction(self.mask_latent_predictor,
                                                    mask_pred_target,
                                                    fused_t_masked_feats,
                                                    masks=masked_index,
                                                    loss=self.mask_latent_prediction.loss)
                output["ssl_losses"]["recover_loss"] = recover_loss

            if "imitation" in task:
                action_logits = self.mlp(agg_feat)
                xyzrpy = self.aux_mlp(agg_feat)
                output["predict"]["action_logits"] = action_logits
                output["predict"]["xyzrpy"] = xyzrpy

        elif mask == "no_mask":
            if "order" in task:
                fom_loss = self.forward_order_prediction(fused_t_feats,
                                                         compute_loss=True,
                                                         additional_input=additional_input)
                output["ssl_losses"]["fom_loss"] = fom_loss
            if "imitation" in task:
                fused_t_feats, _ = self.forward_cross_time(fused_t_feats, additional_input)
                output["repr"]['cross_time_repr'] = fused_t_feats
                agg_feat = fused_t_feats[:, 0, :]
                action_logits = self.mlp(agg_feat)
                xyzrpy = self.aux_mlp(agg_feat)
                output["predict"]["action_logits"] = action_logits
                output["predict"]["xyzrpy"] = xyzrpy

        output["pos_emb"]["cross_time_emb"] = self.cross_time_pos_emb.pe[:, :self.num_stack + 1, :]
        return output

    def forward_cross_time(self, x, additional_input=None):
        cls = self.cls.expand(x.shape[0], self.cls.shape[1], self.cls.shape[2])
        x = torch.cat([cls, x], dim=1)
        x = self.cross_time_pos_emb(x)
        if additional_input is not None:
            x = torch.cat([x, additional_input], dim=1)
        x, attn_maps = self.cross_time_trf(x)
        return x, attn_maps

    def forward_order_prediction(self, fused_t_feats: torch.Tensor, compute_loss=True, additional_input=None):

        bs, num_frame, dim = fused_t_feats.shape

        target, shuffled_fused_t_feats = self.random_shuffle(fused_t_feats, reorder_prob=self.fom_args.reorder_prob)

        if all(element == -1 for element in target.reshape(bs * num_frame, )):
            return torch.zeros([])

        x, attn_maps = self.forward_cross_time(shuffled_fused_t_feats, additional_input)
        x = x[:, 1: 1 + self.num_stack, :]
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
        nll_sum = []
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
                nll = (nll_ + nll) / 2
                nll_sum.append(nll)
            elif name == "tactile":
                side_feat = self.nce_head_t_vat(feat)
                side_feat = side_feat.reshape(bs, num_frame * dim)
                cos_sim = F.cosine_similarity(v_vat_feats, side_feat[None, :, :], dim=-1)
                cos_sim = cos_sim / self.nce_args.temp
                nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
                nll_ = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=0)
                nll = nll.mean()
                nll_ = nll_.mean()
                nll = (nll_ + nll) / 2
                nll_sum.append(nll)
            elif name == "qpos":
                side_feat = self.nce_head_qpos_va(feat)
                side_feat = side_feat.reshape(bs, num_frame * dim)
                cos_sim = F.cosine_similarity(v_va_feats, side_feat[None, :, :], dim=-1)
                cos_sim = cos_sim / self.nce_args.temp
                nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
                nll_ = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=0)
                nll = nll.mean()
                nll_ = nll_.mean()
                nll = (nll_ + nll) / 2
                nll_sum.append(nll)
        nce_loss_dict = {
            '_loss': sum(nll_sum) / len(nll_sum),
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

        # process of qpos signal
        if "qpos" in multimod_inputs.keys():
            qpos_signal = multimod_inputs["qpos"]
            qpos_signal = self.preprocess_qpos(qpos_signal)
            # [B, 1, 40K] -> [B, 64, 251]
            qpos_signal = self.tokenization_qpos(qpos_signal)
            # [B*N, 1, 256] -> [B, N, 256]
            qpos_signal = self.positional_encoding_qpos(qpos_signal)
            qpos_signal = self.encoder_qpos(qpos_signal)
            encoded_inputs["qpos"] = qpos_signal

        return encoded_inputs

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
                        loss: str = "mse", ):
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
            mask = masks["qpos"]
            mask = 1 - mask
            pred_loss = loss_fn(pred_out, target, mask)

        return pred_loss

