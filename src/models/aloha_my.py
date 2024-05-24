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
import math


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

        pe = torch.zeros(1000, d_model)
        position = torch.arange(0, 1000, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.cross_time_pe = pe

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
        query_embed = query_embed.unsqueeze(0).repeat(bs, 1, 1)  # input for the transformer-decoder !!
        # mask = mask.flatten(1)

        additional_pos_embed = additional_pos_embed.unsqueeze(0).repeat(bs, 1, 1)[:, :1]
        chunk_len = query_embed.shape[1]

        # seq+2, bs, dim(positional embedding for input(multi_mod_input) for trm-encoder!!)

        addition_input = torch.stack([latent_input], dim=1)
        addition_input = addition_input + additional_pos_embed

        tgt = torch.zeros_like(query_embed)

        trm_enc_out = self.encoder(multi_mod_input,
                                   multi_mod_input_key_padding_mask=mask,
                                   mask=mask_type,
                                   task=task,
                                   mode=mode,
                                   additional_input=addition_input.to(query_embed.device))
        memory = trm_enc_out["repr"]['cross_time_repr'][:, 1:, :]
        pos_emb = self.cross_time_pe[:, memory.shape[1] + 1: memory.shape[1] + 1 + chunk_len, :].repeat(bs, 1, 1).to(query_embed.device)

        mask = (torch.triu(torch.ones(chunk_len, chunk_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(query_embed.device)

        t, s = torch.meshgrid(
            torch.arange(chunk_len),
            torch.arange(memory.shape[1]),
            indexing='ij'
        )
        memory_mask = t >= (s - 1)  # add one dimension since time is the first token in cond
        memory_mask = memory_mask.float().masked_fill(memory_mask == 0, float('-inf')).masked_fill(memory_mask == 1,
                                                                                                   float(0.0))

        hs = self.decoder(
            tgt=pos_emb,
            memory=memory,
            tgt_mask=mask,
            memory_mask=None
        )
        # hs = torch.nn.functional.tanh(hs)  # 697 nan
        return {"action_decoder_out": hs,
                "obs_encoder_out": trm_enc_out,
                }


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()  # pop the last element, but why??
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


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
        self.pos_table = get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim)  # [CLS], qpos, a_seq
        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)  # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)  # learned position embedding for proprio and latent
        # self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
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
            pos_embed = self.pos_table.clone().detach().to(encoder_input.device)
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
        # proprio_input = self.input_proj_robot_state(qpos)
        # fold camera dimension into width dimension
        multi_mod_input["vision"] = multi_mod_input["vision"]["v_fix"]
        output = self.transformer(
            multi_mod_input=multi_mod_input,
            mask=mask,

            query_embed=self.query_embed.weight,
            latent_input=latent_input,
            proprio_input=None,
            additional_pos_embed=self.additional_pos_embed.weight,

            mask_type=mask_type,
            task=task,
            mode=mode,
        )
        hs = output["action_decoder_out"]
        a_hat = self.action_head(hs)  # bs, seq_len, action_dim
        # a_hat = F.tanh(a_hat)
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
        if a_hat.shape[-1] == 3:
            og_a_hat = a_hat

            return None, None, all_time_position, all_time_orientation, og_a_hat




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

