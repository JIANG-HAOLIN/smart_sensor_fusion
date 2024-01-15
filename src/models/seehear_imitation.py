import torch
import torch.nn as nn
from torch.nn.modules.activation import MultiheadAttention
from src.models.encoders.res_net_18 import make_tactile_encoder
from omegaconf import DictConfig, OmegaConf
import hydra


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
                 pos_emb_args: DictConfig,
                 transformer_args: DictConfig,
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
        self.preprocess_audio = hydra.utils.instantiate(preprocess_audio_args)
        self.tokenization_audio = hydra.utils.instantiate(tokenization_audio)
        self.positional_encoding_audio = hydra.utils.instantiate(pe_audio)
        self.encoder_audio = hydra.utils.instantiate(encoder_audio_args)

        self.preprocess_vision = hydra.utils.instantiate(preprocess_vision_args)
        self.tokenization_vision = hydra.utils.instantiate(tokenization_vision)
        self.positional_encoding_vision = hydra.utils.instantiate(pe_vision)
        self.encoder_vision = hydra.utils.instantiate(encoder_vision_args)

        self.register_parameter('vision_gamma', torch.nn.Parameter(torch.randn((1, 1, model_dim))))
        self.register_parameter('audio_gamma', torch.nn.Parameter(torch.randn((1, 1, model_dim))))
        # self.vision_gamma = torch.nn.Linear(model_dim, model_dim, bias=False)
        # self.audio_gamma = torch.nn.Linear(model_dim, model_dim, bias=False)

        self.cls = torch.nn.Parameter(torch.randn(1, 1, transformer_args.token_dim))
        self.pos_emb = hydra.utils.instantiate(pos_emb_args)
        self.transformer = hydra.utils.instantiate(transformer_args)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(model_dim, model_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(model_dim, model_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(model_dim, 3 ** 3),
        )
        self.aux_mlp = torch.nn.Linear(model_dim, 6)

    def forward(self, vision_signal: torch.Tensor, audio_signal: torch.Tensor):
        audio_signal = self.preprocess_audio(audio_signal)
        audio_signal = self.tokenization_audio(audio_signal)
        audio_signal = self.positional_encoding_audio(audio_signal)
        audio_signal = self.encoder_audio(audio_signal)

        # bs, _, audio_len = audio_signal.shape
        # num_frame = vision_signal.shape[1]
        # audio_signal = audio_signal.reshape(bs * num_frame, 1, -1)
        # audio_signal = self.tokenization_audio(audio_signal)
        # audio_signal = self.positional_encoding_audio(audio_signal)
        # audio_signal = self.encoder_audio(audio_signal)
        # if type(audio_signal) == tuple:
        #     audio_signal, attn_map = audio_signal
        # if len(audio_signal.shape) == 3:
        #     audio_signal = audio_signal[:, 0]
        # audio_signal = audio_signal.reshape(bs, num_frame, audio_signal.shape[-1])

        batch_size, num_stack, c_v, h_v, w_v = vision_signal.shape
        vision_signal = torch.reshape(vision_signal, (-1, c_v, h_v, w_v))
        vision_signal = self.preprocess_vision(vision_signal)
        vision_signal = self.tokenization_vision(vision_signal)
        if type(vision_signal) == tuple:
            vision_signal, attn_map = vision_signal
        if len(vision_signal.shape) == 2:
            vision_signal = vision_signal.reshape(vision_signal.shape[0], 1, vision_signal.shape[1])
            # [B*N, 256] -> [B*N, 1, 256]
        vision_signal = vision_signal.view(batch_size, num_stack * vision_signal.shape[-2], vision_signal.shape[-1])
        vision_signal = self.positional_encoding_vision(vision_signal)
        vision_signal = self.encoder_vision(vision_signal)

        if type(audio_signal) == tuple:
            audio_signal, attn_audio = audio_signal
        if type(vision_signal) == tuple:
            vision_signal, attn_vision = vision_signal

        audio_signal = self.audio_gamma * audio_signal
        vision_signal = self.vision_gamma * vision_signal
        # audio_signal = self.audio_gamma(audio_signal)
        # vision_signal = self.vision_gamma(vision_signal)
        x = audio_signal + vision_signal

        cls = self.cls.expand(x.shape[0], self.cls.shape[1], self.cls.shape[2])
        x = torch.cat([cls, x], dim=1)
        x = self.pos_emb(x)
        x, attn_maps = self.transformer(x)
        x = x[:, 0]
        action_logits = self.mlp(x)
        xyzrpy = self.aux_mlp(x)
        return action_logits, xyzrpy, attn_maps


class VisionAudioFusion_Supervised(torch.nn.Module):
    """Vision audio fusion model for vision and audio signal from see_hear_feel using Earlycat /multiple to one"""

    def __init__(self,
                 preprocess_audio_args: DictConfig,
                 tokenization_audio: DictConfig,
                 pe_audio: DictConfig,
                 encoder_audio_args: DictConfig,
                 preprocess_vision_args: DictConfig,
                 tokenization_vision: DictConfig,
                 pe_vision: DictConfig,
                 encoder_vision_args: DictConfig,
                 last_pos_emb_args: DictConfig,
                 transformer_args: DictConfig,
                 model_dim: int,
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
        self.preprocess_audio = hydra.utils.instantiate(preprocess_audio_args)
        self.tokenization_audio = hydra.utils.instantiate(tokenization_audio)
        self.positional_encoding_audio = hydra.utils.instantiate(pe_audio)
        self.encoder_audio = hydra.utils.instantiate(encoder_audio_args)

        self.preprocess_vision = hydra.utils.instantiate(preprocess_vision_args)
        self.tokenization_vision = hydra.utils.instantiate(tokenization_vision)
        self.positional_encoding_vision = hydra.utils.instantiate(pe_vision)
        self.encoder_vision = hydra.utils.instantiate(encoder_vision_args)

        self.cls = torch.nn.Parameter(torch.randn(1, 1, model_dim))
        self.last_pos_emb = hydra.utils.instantiate(last_pos_emb_args)
        self.transformer = hydra.utils.instantiate(transformer_args)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(model_dim, model_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(model_dim, model_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(model_dim, 3 ** 3),
        )
        self.aux_mlp = torch.nn.Linear(model_dim, 6)

    def forward(self, vision_signal: torch.Tensor, audio_signal: torch.Tensor):
        audio_signal = self.preprocess_audio(audio_signal)
        audio_signal = self.tokenization_audio(audio_signal)
        audio_signal = self.positional_encoding_audio(audio_signal)
        audio_signal = self.encoder_audio(audio_signal)

        batch_size, num_stack, c_v, h_v, w_v = vision_signal.shape
        vision_signal = torch.reshape(vision_signal, (-1, c_v, h_v, w_v))
        vision_signal = self.preprocess_vision(vision_signal)
        vision_signal = self.tokenization_vision(vision_signal)
        vision_signal = vision_signal.view(batch_size, num_stack * vision_signal.shape[-2], vision_signal.shape[-1])
        vision_signal = self.positional_encoding_vision(vision_signal)
        vision_signal = self.encoder_vision(vision_signal)

        if type(audio_signal) == tuple:
            audio_signal, attn_audio = audio_signal
        if type(vision_signal) == tuple:
            vision_signal, attn_vision = vision_signal

        cls = self.cls.expand(audio_signal.shape[0], self.cls.shape[1], self.cls.shape[2])
        cls = self.last_pos_emb(cls, index=0)
        audio_signal = self.last_pos_emb(audio_signal, index=1)
        vision_signal = self.last_pos_emb(vision_signal, index=2)

        x = torch.cat([cls, audio_signal, vision_signal], dim=1)
        x, attn_maps = self.transformer(x)
        x = x[:, 0]
        action_logits = self.mlp(x)
        xyzrpy = self.aux_mlp(x)
        return action_logits, xyzrpy, attn_maps


class Seehearfeel_Vanilla(torch.nn.Module):
    """Vision audio fusion model for vision and audio signal from see_hear_feel using see_hear_feel vg_ah ablation"""

    def __init__(self,
                 a_encoder: DictConfig,
                 v_encoder: DictConfig,
                 preprocess_audio_args: DictConfig,
                 args: DictConfig,
                 **kwargs
                 ):
        """

        Args:

             **kwargs:
        """
        super().__init__()
        self.v_encoder = hydra.utils.instantiate(v_encoder)
        self.a_encoder = hydra.utils.instantiate(a_encoder)
        self.t_encoder = make_tactile_encoder(args.encoder_dim)
        self.a_mel = hydra.utils.instantiate(preprocess_audio_args)
        self.mlp = None
        self.layernorm_embed_shape = args.encoder_dim * args.num_stack  ### 256x5 ### why 5 ????
        self.encoder_dim = args.encoder_dim
        self.ablation = args.ablation
        self.use_vision = False
        self.use_tactile = False
        self.use_audio = False
        self.use_mha = args.use_mha
        self.query = nn.Parameter(torch.randn(1, 1, self.layernorm_embed_shape))

        ## load models
        self.modalities = self.ablation.split("_")  #### vg=gripper_cam t=tactile ah=fixed_mic
        print(f"Using modalities: {self.modalities}")
        self.embed_dim = self.layernorm_embed_shape * len(self.modalities)  ## 256 5 3 all emb concat together!
        self.layernorm = nn.LayerNorm(self.layernorm_embed_shape)
        self.mha = MultiheadAttention(self.layernorm_embed_shape, args.num_heads)
        self.bottleneck = nn.Linear(
            self.embed_dim, self.layernorm_embed_shape
        )  # if we dont use mha

        # action_dim = 3 ** task2actiondim[args.task]

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.layernorm_embed_shape, 1024),  ## 256x6->1024
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 3 ** args.action_dim),
        )
        self.aux_mlp = torch.nn.Linear(self.layernorm_embed_shape, 6)

    def forward(self, vision_signal: torch.Tensor, audio_signal: torch.Tensor):
        """
                Args:
                    cam_fixed_framestack, cam_gripper_framestack, tactile_framestack, audio_clip_g, audio_clip_h
                    vf_inp: [batch, num_stack, 3, H, W]
                    vg_inp: [batch, num_stack, 3, H, W]
                    t_inp: [batch, num_stack, 3, H, W]
                    a_inp: [batch, 1, T]

                """

        vg_inp, audio_h = vision_signal, audio_signal
        embeds = []

        if "vg" in self.modalities:
            batch, num_stack, _, Hv, Wv = vg_inp.shape
            vg_inp = vg_inp.view(batch * num_stack, 3, Hv, Wv)
            vg_embeds = self.v_encoder(vg_inp)  # [batch * num_stack, encoder_dim]
            vg_embeds = vg_embeds.view(
                -1, self.layernorm_embed_shape
            )  # [batch, encoder_dim * num_stack]
            embeds.append(vg_embeds)
        if "ah" in self.modalities:
            batch, _, _ = audio_h.shape
            ah_mel = self.a_mel(audio_h)
            ah_embeds = self.a_encoder(ah_mel)
            ah_embeds = ah_embeds.view(-1, self.layernorm_embed_shape)
            embeds.append(ah_embeds)

        if self.use_mha:  ## stack or concate
            mlp_inp = torch.stack(embeds, dim=0)  # create a new dimension !!! [3, batch, D]
            # batch first=False, (L, N, E)
            # query = self.query.repeat(1, batch, 1) # [1, 1, D] -> [1, batch, D]
            # change back to 3*3
            mha_out, weights = self.mha(mlp_inp, mlp_inp, mlp_inp)  # [3, batch, D]
            mha_out += mlp_inp
            mlp_inp = torch.concat([mha_out[i] for i in range(mha_out.shape[0])], 1)
            mlp_inp = self.bottleneck(mlp_inp)
            # mlp_inp = mha_out.squeeze(0) # [batch, D]
        else:
            mlp_inp = torch.cat(embeds, dim=-1)
            mlp_inp = self.bottleneck(mlp_inp)
            weights = None

        action_logits = self.mlp(mlp_inp)
        xyzrpy = self.aux_mlp(mlp_inp)
        return action_logits, xyzrpy, weights


class EarlyCat_Extractor(torch.nn.Module):
    """Vision audio fusion model for vision and audio signal from see_hear_feel using Early Summation/multiple to one"""

    def __init__(self,
                 preprocess_audio_args: DictConfig,
                 tokenization_audio: DictConfig,
                 pe_audio: DictConfig,
                 encoder_audio_args: DictConfig,
                 preprocess_vision_args: DictConfig,
                 tokenization_vision: DictConfig,
                 pe_vision: DictConfig,
                 encoder_vision_args: DictConfig,
                 last_pos_emb_args: DictConfig,
                 transformer_args: DictConfig,
                 model_dim: int,
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
        self.preprocess_audio = hydra.utils.instantiate(preprocess_audio_args)
        self.tokenization_audio = hydra.utils.instantiate(tokenization_audio)
        self.positional_encoding_audio = hydra.utils.instantiate(pe_audio)
        self.encoder_audio = hydra.utils.instantiate(encoder_audio_args)

        self.preprocess_vision = hydra.utils.instantiate(preprocess_vision_args)
        self.tokenization_vision = hydra.utils.instantiate(tokenization_vision)
        self.positional_encoding_vision = hydra.utils.instantiate(pe_vision)
        self.encoder_vision = hydra.utils.instantiate(encoder_vision_args)

        self.cls = torch.nn.Parameter(torch.randn(1, 1, model_dim))
        self.last_pos_emb = hydra.utils.instantiate(last_pos_emb_args)
        self.transformer = hydra.utils.instantiate(transformer_args)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(model_dim, 1024),  ## 256x6->1024
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 3 ** 3),
        )
        self.aux_mlp = torch.nn.Linear(self.layernorm_embed_shape, 6)

    def forward(self, vision_signal: torch.Tensor, audio_signal: torch.Tensor):
        audio_signal = self.preprocess_audio(audio_signal)
        audio_signal = self.tokenization_audio(audio_signal)
        audio_signal = self.positional_encoding_audio(audio_signal)
        audio_signal = self.encoder_audio(audio_signal)

        batch_size, num_stack, c_v, h_v, w_v = vision_signal.shape
        vision_signal = torch.reshape(vision_signal, (-1, c_v, h_v, w_v))
        vision_signal = self.preprocess_vision(vision_signal)
        vision_signal = self.tokenization_vision(vision_signal)
        vision_signal = vision_signal.view(batch_size, num_stack * vision_signal.shape[-2], vision_signal.shape[-1])
        vision_signal = self.positional_encoding_vision(vision_signal)
        vision_signal = self.encoder_vision(vision_signal)

        if type(audio_signal) == tuple:
            audio_signal, attn_audio = audio_signal
        if type(vision_signal) == tuple:
            vision_signal, attn_vision = vision_signal

        cls = self.cls.expand(audio_signal.shape[0], self.cls.shape[1], self.cls.shape[2])
        cls = self.last_pos_emb(cls, index=0)
        audio_signal = self.last_pos_emb(audio_signal, index=1)
        vision_signal = self.last_pos_emb(vision_signal, index=2)

        x = torch.cat([cls, audio_signal, vision_signal], dim=1)
        x, attn_maps = self.transformer(x)
        x = x[:, 0]
        action_logits = self.mlp(x)
        xyzrpy = self.aux_mlp(x)
        return action_logits, xyzrpy, attn_maps
