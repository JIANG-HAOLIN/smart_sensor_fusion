import torch
import torch.nn as nn
from torch.nn.modules.activation import MultiheadAttention
from src.models.vit_implementations import Vit_Classifier, Vit_Classifier_Mel, LrnEmb_Agg_Trf
from src.models.utils.mel_spec import MelSpec
from omegaconf import DictConfig, OmegaConf
import hydra
from typing import Optional, Dict, List
from types import SimpleNamespace


class VisionAudioFusion_EarlySum(torch.nn.Module):
    """Vision audio fusion model for vision and audio signal from see_hear_feel using Early Summation"""

    def __init__(self,

                 model_dim: int,

                 preprocess_audio_args: DictConfig,
                 tokenization_audio: DictConfig,
                 pe_audio: DictConfig,
                 encoder_audio_args: DictConfig,

                 preprocess_vision_args: DictConfig,
                 tokenization_vision: DictConfig,
                 pe_vision: DictConfig,
                 encoder_vision_args: DictConfig,

                 preprocess_tactile_args: DictConfig,
                 tokenization_tactile: DictConfig,
                 pe_tactile: DictConfig,
                 encoder_tactile_args: DictConfig,

                 fusion_args: DictConfig,

                 pos_emb_args: DictConfig,
                 transformer_classifier_args: DictConfig,
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
             transformer_classifier_args: arguments for transformer classifier
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

        self.preprocess_audio = hydra.utils.instantiate(preprocess_audio_args)
        self.tokenization_audio = hydra.utils.instantiate(tokenization_audio)
        self.positional_encoding_audio = hydra.utils.instantiate(pe_audio)
        self.encoder_audio = hydra.utils.instantiate(encoder_audio_args)

        self.preprocess_vision = hydra.utils.instantiate(preprocess_vision_args)
        self.tokenization_vision = hydra.utils.instantiate(tokenization_vision)
        self.positional_encoding_vision = hydra.utils.instantiate(pe_vision)
        self.encoder_vision = hydra.utils.instantiate(encoder_vision_args)

        self.preprocess_tactile = hydra.utils.instantiate(preprocess_tactile_args)
        self.tokenization_tactile = hydra.utils.instantiate(tokenization_tactile)
        self.positional_encoding_tactile = hydra.utils.instantiate(pe_tactile)
        self.encoder_tactile = hydra.utils.instantiate(encoder_tactile_args)

        self.cls = torch.nn.Parameter(torch.randn(1, 1, transformer_classifier_args.model_dim))

        self.fusion = hydra.utils.instantiate(fusion_args)
        self.register_parameter('vision_gamma', torch.nn.Parameter(torch.randn((1, 1, model_dim))))
        self.register_parameter('audio_gamma', torch.nn.Parameter(torch.randn((1, 1, model_dim))))
        self.register_parameter('tactile_gamma', torch.nn.Parameter(torch.randn((1, 1, model_dim))))

        self.pos_emb = hydra.utils.instantiate(pos_emb_args)
        self.transformer_classifier = hydra.utils.instantiate(transformer_classifier_args)

    def forward(self, multimod_inputs: Dict,
                task: Optional[str]):

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

        if task == 'repr':
            return encoded_inputs
        audio_signal = self.audio_gamma * audio_signal
        vision_signal = self.vision_gamma * vision_signal
        tactile_signal = self.tactile_gamma * tactile_signal
        x = audio_signal + vision_signal + tactile_signal
        cls = self.cls.expand(x.shape[0], self.cls.shape[1], self.cls.shape[2])

        x = torch.cat([cls, x], dim=1)
        x = self.pos_emb(x)

        return self.transformer_classifier(x)

    def forward_fom(self, batch, compute_loss=True):
        shuffled_orders = torch.randperm(batch)
        transformed_c_v_feats = self.forward_repr(batch, encode_clip=False)

        # Reshuffle c_v_feats according to targets
        shuffled_orders_expanded = shuffled_orders.unsqueeze(-1).expand_as(
            transformed_c_v_feats)
        c_v_feats_shuffled = torch.zeros_like(
            transformed_c_v_feats, dtype=transformed_c_v_feats.dtype,
            device=transformed_c_v_feats.device)
        c_v_feats_shuffled = c_v_feats_shuffled.scatter_(
            1, shuffled_orders_expanded, transformed_c_v_feats)

        # compute pos_ids in embedding layer
        encoded_clip = self.c_encoder(
            clip_level_pos_ids=None,
            clip_level_frame_feat=c_v_feats_shuffled,
            attention_mask=batch["c_attn_masks"])

        bs, seq_len, hid_size = encoded_clip.size()
        encoded_clip = encoded_clip.view(bs * seq_len, hid_size)

        frame_reorder_outputs = self.fom_output(encoded_clip)

        if compute_loss:
            targets = batch['targets'].view(frame_reorder_outputs.shape[0])
            loss = F.cross_entropy(
                frame_reorder_outputs, targets, ignore_index=-1,
                reduction='mean')
            return loss
        return frame_reorder_outputs
