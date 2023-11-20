import torch
import torch.nn as nn
from torch.nn.modules.activation import MultiheadAttention
from src.models.trafo_classifier_vit import TransformerClassifierVit, TransformerClassifierVit_Mel
from src.models.utils.mel_spec import MelSpec
from omegaconf import DictConfig, OmegaConf
import hydra


class classification_model(torch.nn.Module):
    """Classification model for progress prediction for audio signal from see_hear_feel using vanilla ViT"""

    def __init__(self,
                 preprocess_args: DictConfig,
                 encoder_args: DictConfig,
                 transformer_args: DictConfig,
                 **kwargs):
        super().__init__()
        self.preprocess = hydra.utils.instantiate(preprocess_args)
        self.encoder = hydra.utils.instantiate(encoder_args)
        self.transformer = hydra.utils.instantiate(transformer_args)

    def forward(self, x):
        x = self.preprocess(x)
        x = self.encoder(x)
        return self.transformer(x)


class time_patch_model(torch.nn.Module):
    """Classification model for progress prediction for audio signal from see_hear_feel using columns from mel spec"""

    def __init__(self,
                 preprocess_args: DictConfig,
                 transformer_args: DictConfig,
                 **kwargs):
        super().__init__()
        self.preprocess = MelSpec(**preprocess_args)
        out_size = self.preprocess.out_size
        self.transformer = TransformerClassifierVit_Mel(**transformer_args, input_size=out_size)

    def forward(self, x):
        x = self.preprocess(x)
        return self.transformer(x)


class VisionAudioFusion(torch.nn.Module):
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
        self.preprocess_audio = hydra.utils.instantiate(preprocess_audio_args)
        self.tokenization_audio = hydra.utils.instantiate(tokenization_audio)
        self.positional_encoding_audio = hydra.utils.instantiate(pe_audio)
        self.encoder_audio = hydra.utils.instantiate(encoder_audio_args)

        self.preprocess_vision = hydra.utils.instantiate(preprocess_vision_args)
        self.tokenization_vision = hydra.utils.instantiate(tokenization_vision)
        self.positional_encoding_vision = hydra.utils.instantiate(pe_vision)
        self.encoder_vision = hydra.utils.instantiate(encoder_vision_args)

        self.cls = torch.nn.Parameter(torch.randn(1, 1, last_pos_emb_args.emb_dim))
        self.last_pos_emb = hydra.utils.instantiate(last_pos_emb_args)
        self.transformer_classifier = hydra.utils.instantiate(transformer_classifier_args)

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
        return self.transformer_classifier(x)


class VisionAudioFusionTimeEmb(torch.nn.Module):
    """Vision audio fusion model for vision and audio signal from see_hear_feel using Early Summation/multiple to one"""

    def __init__(self,
                 preprocess_audio_args: DictConfig,
                 tokenization_audio: DictConfig,
                 pe_audio: DictConfig,
                 encoder_audio_args: DictConfig,
                 preprocess_vision_args: DictConfig,
                 tokenization_vision: DictConfig,
                 pe_vision_spatial: DictConfig,
                 pe_vision_temporal: DictConfig,
                 encoder_vision_args: DictConfig,
                 last_pos_emb_args: DictConfig,
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
        self.preprocess_audio = hydra.utils.instantiate(preprocess_audio_args)
        self.tokenization_audio = hydra.utils.instantiate(tokenization_audio)
        self.positional_encoding_audio = hydra.utils.instantiate(pe_audio)
        self.encoder_audio = hydra.utils.instantiate(encoder_audio_args)

        self.preprocess_vision = hydra.utils.instantiate(preprocess_vision_args)
        self.tokenization_vision = hydra.utils.instantiate(tokenization_vision)
        self.positional_encoding_vision_temporal = hydra.utils.instantiate(pe_vision_temporal)
        self.positional_encoding_vision_spatial = hydra.utils.instantiate(pe_vision_spatial)
        self.encoder_vision = hydra.utils.instantiate(encoder_vision_args)

        self.cls = torch.nn.Parameter(torch.randn(1, 1, last_pos_emb_args.emb_dim))
        self.last_pos_emb = hydra.utils.instantiate(last_pos_emb_args)
        self.transformer_classifier = hydra.utils.instantiate(transformer_classifier_args)

    def forward(self, vision_signal: torch.Tensor, audio_signal: torch.Tensor):
        audio_signal = self.preprocess_audio(audio_signal)
        audio_signal = self.tokenization_audio(audio_signal)
        audio_signal = self.positional_encoding_audio(audio_signal)
        audio_signal = self.encoder_audio(audio_signal)

        batch_size, num_stack, c_v, h_v, w_v = vision_signal.shape
        vision_signal = torch.reshape(vision_signal, (-1, c_v, h_v, w_v))
        vision_signal = self.preprocess_vision(vision_signal)
        vision_signal = self.tokenization_vision(vision_signal)
        vision_signal = vision_signal.view(batch_size, num_stack, vision_signal.shape[-2], vision_signal.shape[-1])
        vision_signal = self.positional_encoding_vision_temporal(vision_signal, which_dim=1)
        vision_signal = self.positional_encoding_vision_spatial(vision_signal, which_dim=2)
        vision_signal = vision_signal.view(batch_size, num_stack*vision_signal.shape[-2], vision_signal.shape[-1])
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
        return self.transformer_classifier(x)


class VisionAudioFusion_seehearfeel(torch.nn.Module):
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
            torch.nn.Linear(1024, 10),
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
        return [action_logits, xyzrpy]
