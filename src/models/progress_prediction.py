import torch
import torch.nn as nn
from torch.nn.modules.activation import MultiheadAttention
from src.models.vit_implementations import Vit_Classifier, Vit_Classifier_Mel, LrnEmb_Agg_Trf
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
        self.transformer = Vit_Classifier_Mel(**transformer_args, input_size=out_size)

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


class VisionAudioFusion_Extractor(torch.nn.Module):
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

        # self.register_parameter('vision_gamma', torch.nn.Parameter(torch.randn((1, 1, model_dim))))
        # self.register_parameter('audio_gamma', torch.nn.Parameter(torch.randn((1, 1, model_dim))))
        self.vision_gamma = torch.nn.Linear(model_dim, model_dim, bias=False)
        self.audio_gamma = torch.nn.Linear(model_dim, model_dim, bias=False)

        self.cls = torch.nn.Parameter(torch.randn(1, 1, transformer_classifier_args.model_dim))
        self.pos_emb = hydra.utils.instantiate(pos_emb_args)
        self.transformer_classifier = hydra.utils.instantiate(transformer_classifier_args)

    def forward(self, vision_signal: torch.Tensor, audio_signal: torch.Tensor):
        # audio_signal = self.preprocess_audio(audio_signal)
        # audio_signal = self.tokenization_audio(audio_signal)
        # audio_signal = self.positional_encoding_audio(audio_signal)
        # audio_signal = self.encoder_audio(audio_signal)

        bs, _, audio_len = audio_signal.shape
        num_frame = vision_signal.shape[1]
        audio_signal = audio_signal.reshape(bs * num_frame, 1, -1)
        audio_signal = self.tokenization_audio(audio_signal)
        audio_signal = self.positional_encoding_audio(audio_signal)
        audio_signal = self.encoder_audio(audio_signal)
        if type(audio_signal) == tuple:
            audio_signal, attn_map = audio_signal
        if len(audio_signal.shape) == 3:
            audio_signal = audio_signal[:, 0]
        audio_signal = audio_signal.reshape(bs, num_frame, audio_signal.shape[-1])

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

        # audio_signal = self.audio_gamma * audio_signal
        # vision_signal = self.vision_gamma * vision_signal
        audio_signal = self.audio_gamma(audio_signal)
        vision_signal = self.vision_gamma(vision_signal)
        x = audio_signal + vision_signal

        cls = self.cls.expand(x.shape[0], self.cls.shape[1], self.cls.shape[2])
        x = torch.cat([cls, x], dim=1)
        x = self.pos_emb(x)

        return self.transformer_classifier(x)


class VisionAudioFusion_EarlySum2Fuse(torch.nn.Module):
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

        self.cls = torch.nn.Parameter(torch.randn(1, 1, transformer_classifier_args.model_dim))
        self.fuse = LrnEmb_Agg_Trf(model_dim=model_dim,
                                   num_heads=1,
                                   num_layers=1,
                                   num_emb=15,
                                   )
        self.pos_emb = hydra.utils.instantiate(pos_emb_args)
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

        vision_signal = vision_signal.reshape(batch_size * num_stack, 1, vision_signal.shape[-1])
        audio_signal = audio_signal.reshape(batch_size * num_stack, 1, audio_signal.shape[-1])

        x = torch.cat([vision_signal, audio_signal], dim=1)
        x, attn_map = self.fuse(x)  # [B, 256] not [B, 1, 256]
        x = x.reshape(batch_size, num_stack, x.shape[-1])
        cls = self.cls.expand(batch_size, 1, self.cls.shape[2])

        x = torch.cat([cls, x], dim=1)
        x = self.pos_emb(x)

        return self.transformer_classifier(x)


class VisionAudioFusion_EarlyFuse(torch.nn.Module):
    """Vision audio fusion model for vision and audio signal from see_hear_feel using Early Summation/multiple to one"""

    def __init__(self,
                 preprocess_audio_args: DictConfig,
                 tokenization_audio: DictConfig,
                 pe_audio: DictConfig,
                 preprocess_vision_args: DictConfig,
                 tokenization_vision: DictConfig,
                 pe_vision: DictConfig,
                 early_fuse: DictConfig,
                 mdl_type_emb: DictConfig,
                 temporal_emb: DictConfig,
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

        self.preprocess_vision = hydra.utils.instantiate(preprocess_vision_args)
        self.tokenization_vision = hydra.utils.instantiate(tokenization_vision)
        self.positional_encoding_vision = hydra.utils.instantiate(pe_vision)

        self.mdl_type_emb = hydra.utils.instantiate(mdl_type_emb)
        self.early_fuse = hydra.utils.instantiate(early_fuse)

        self.cls2 = torch.nn.Parameter(torch.randn(1, 1, mdl_type_emb.emb_dim))
        self.temporal_emb = hydra.utils.instantiate(temporal_emb)
        self.transformer_classifier = hydra.utils.instantiate(transformer_classifier_args)

    def forward(self, vision_signal: torch.Tensor, audio_signal: torch.Tensor):

        batch_size, num_stack, c_v, h_v, w_v = vision_signal.shape
        vision_signal = torch.reshape(vision_signal, (-1, c_v, h_v, w_v))
        vision_signal = self.preprocess_vision(vision_signal)
        vision_signal = self.tokenization_vision(vision_signal)
        vision_signal = self.positional_encoding_vision(vision_signal)

        audio_signal = self.preprocess_audio(audio_signal)
        audio_signal = self.tokenization_audio(audio_signal)
        audio_signal = audio_signal.view(batch_size * num_stack, -1, audio_signal.shape[-1])
        audio_signal = self.positional_encoding_audio(audio_signal)

        if type(audio_signal) == tuple:
            audio_signal, attn_audio = audio_signal
        if type(vision_signal) == tuple:
            vision_signal, attn_vision = vision_signal

        audio_signal = self.mdl_type_emb(audio_signal, index=1)
        vision_signal = self.mdl_type_emb(vision_signal, index=2)

        x = torch.cat([audio_signal, vision_signal], dim=1)
        x = self.early_fuse(x)
        if type(x) == tuple:
            x, attn = x
        x = x.view(batch_size, num_stack, x.shape[-1])
        x = self.temporal_emb(x)

        cls2 = self.cls2.expand(batch_size, self.cls2.shape[1], self.cls2.shape[2])
        x = torch.cat([cls2, x], dim=1)

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
        vision_signal = vision_signal.view(batch_size, num_stack * vision_signal.shape[-2], vision_signal.shape[-1])
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


# class ShortDrillingProgressPredictionVanilla(torch.nn.Module):
#     """Model for short term drilling progress prediction(Y_corr) using acc cage(ac), acc PTU(ap), Force(F), Current(I)"""
#
#     def __init__(self,
#                  preprocess_acc_cage_args: DictConfig,
#                  tokenization_acc_cage: DictConfig,
#                  pe_acc_cage_temporal: DictConfig,
#                  pe_acc_cage_spatial: DictConfig,
#                  preprocess_acc_ptu_args: DictConfig,
#                  tokenization_acc_ptu: DictConfig,
#                  pe_acc_ptu_temporal: DictConfig,
#                  pe_acc_ptu_spatial: DictConfig,
#                  preprocess_force_args: DictConfig,
#                  tokenization_force: DictConfig,
#                  pe_force_temporal: DictConfig,
#                  pe_force_spatial: DictConfig,
#                  preprocess_current_args: DictConfig,
#                  tokenization_current: DictConfig,
#                  pe_current_temporal: DictConfig,
#                  pe_current_spatial: DictConfig,
#                  last_pos_emb_args: DictConfig,
#                  transformer_classifier_args: DictConfig,
#                  **kwargs
#                  ):
#         """
#
#         Args:
#              preprocess_audio_args: arguments for audio prepressing
#              tokenization_audio: arguments for audio tokenization
#              pe_audio: arguments for positional encoding for audio tokens
#              encoder_audio_args: arguments for audio encoder(identity for earlycat/transformer for multi to one)
#              preprocess_vision_args: arguments for vision prepressing
#              tokenization_vision: arguments for vision tokenization
#              pe_vision: arguments for positional encoding for vision tokens
#              encoder_vision_args: arguments for vision encoder(identity for earlycat/transformer for multi to one)
#              transformer_classifier_args: arguments for transformer classifier
#              **kwargs:
#         """
#         super().__init__()
#         self.preprocess_acc_cage = hydra.utils.instantiate(preprocess_acc_cage_args)
#         self.tokenization_acc_cage = hydra.utils.instantiate(tokenization_acc_cage)
#         self.positional_encoding_acc_cage_temporal = hydra.utils.instantiate(pe_acc_cage_temporal)
#         self.positional_encoding_acc_cage_spatial = hydra.utils.instantiate(pe_acc_cage_spatial)
#
#         self.preprocess_acc_ptu = hydra.utils.instantiate(preprocess_acc_ptu_args)
#         self.tokenization_acc_ptu = hydra.utils.instantiate(tokenization_acc_ptu)
#         self.positional_encoding_acc_ptu_temporal = hydra.utils.instantiate(pe_acc_ptu_temporal)
#         self.positional_encoding_acc_ptu_spatial = hydra.utils.instantiate(pe_acc_ptu_spatial)
#
#         self.preprocess_force = hydra.utils.instantiate(preprocess_force_args)
#         self.tokenization_force = hydra.utils.instantiate(tokenization_force)
#         self.positional_encoding_force_temporal = hydra.utils.instantiate(pe_force_temporal)
#         self.positional_encoding_force_spatial = hydra.utils.instantiate(pe_force_spatial)
#
#         self.preprocess_current = hydra.utils.instantiate(preprocess_current_args)
#         self.tokenization_current = hydra.utils.instantiate(tokenization_current)
#         self.positional_encoding_current_temporal = hydra.utils.instantiate(pe_current_temporal)
#         self.positional_encoding_current_spatial = hydra.utils.instantiate(pe_current_spatial)
#
#         self.cls = torch.nn.Parameter(torch.randn(1, 1, last_pos_emb_args.emb_dim))
#         self.last_pos_emb = hydra.utils.instantiate(last_pos_emb_args)
#         self.transformer_classifier = hydra.utils.instantiate(transformer_classifier_args)
#
#     def forward(self,
#                 acc_cage_x: torch.Tensor,
#                 acc_cage_y: torch.Tensor,
#                 acc_cage_z: torch.Tensor,
#                 acc_ptu_x: torch.Tensor,
#                 acc_ptu_y: torch.Tensor,
#                 acc_ptu_z: torch.Tensor,
#                 f_x: torch.Tensor,
#                 f_y: torch.Tensor,
#                 f_z: torch.Tensor,
#                 i_s: torch.Tensor,
#                 i_z: torch.Tensor,
#                 ):
#         batch_size = acc_ptu_x.shape[0]
#
#         acc_cage = torch.cat([acc_cage_x, acc_cage_y, acc_cage_z], dim=0)
#         acc_cage = self.preprocess_acc_cage(acc_cage.unsqueeze(1))
#         _, c_v, h_v, w_v = acc_cage.shape
#         acc_cage = self.tokenization_acc_cage(acc_cage)
#         acc_cage = acc_cage.view(batch_size, 3, acc_cage.shape[-2], acc_cage.shape[-1])
#         acc_cage = self.positional_encoding_acc_cage_temporal(acc_cage, which_dim=1)
#         acc_cage = self.positional_encoding_acc_cage_spatial(acc_cage, which_dim=2)
#         acc_cage = acc_cage.view(batch_size, 3 * acc_cage.shape[-2], acc_cage.shape[-1])
#
#         acc_ptu = torch.cat([acc_ptu_x, acc_ptu_y, acc_ptu_z], dim=0)
#         acc_ptu = self.preprocess_acc_ptu(acc_ptu.unsqueeze(1))
#         _, c_v, h_v, w_v = acc_ptu.shape
#         acc_ptu = self.tokenization_acc_ptu(acc_ptu)
#         acc_ptu = acc_ptu.view(batch_size, 3, acc_ptu.shape[-2], acc_ptu.shape[-1])
#         acc_ptu = self.positional_encoding_acc_ptu_temporal(acc_ptu, which_dim=1)
#         acc_ptu = self.positional_encoding_acc_ptu_spatial(acc_ptu, which_dim=2)
#         acc_ptu = acc_ptu.view(batch_size, 3 * acc_ptu.shape[-2], acc_ptu.shape[-1])
#
#         force = torch.cat([f_x, f_y, f_z], dim=0)
#         force = self.preprocess_force(force.unsqueeze(1))
#         _, c_v, h_v, w_v = force.shape
#         force = self.tokenization_force(force)
#         force = force.view(batch_size, 3, force.shape[-2], force.shape[-1])
#         force = self.positional_encoding_force_temporal(force, which_dim=1)
#         force = self.positional_encoding_force_spatial(force, which_dim=2)
#         force = force.view(batch_size, 3 * force.shape[-2], force.shape[-1])
#
#         current = torch.cat([i_s, i_z], dim=0)
#         current = self.preprocess_current(current.unsqueeze(1))
#         _, c_v, h_v, w_v = current.shape
#         current = self.tokenization_current(current)
#         current = current.view(batch_size, 2, current.shape[-2], current.shape[-1])
#         current = self.positional_encoding_current_temporal(current, which_dim=1)
#         current = self.positional_encoding_current_spatial(current, which_dim=2)
#         current = current.view(batch_size, 2 * current.shape[-2], current.shape[-1])
#
#         cls = self.cls.expand(batch_size, self.cls.shape[1], self.cls.shape[2])
#         cls = self.last_pos_emb(cls, index=0)
#         acc_cage = self.last_pos_emb(acc_cage, index=1)
#         acc_ptu = self.last_pos_emb(acc_ptu, index=2)
#         force = self.last_pos_emb(force, index=3)
#         current = self.last_pos_emb(current, index=4)
#
#         x = torch.cat([cls, acc_cage, acc_ptu, force, current], dim=1)
#         return self.transformer_classifier(x)


class ShortDrillingProgressPredictionVanilla(torch.nn.Module):
    """Model for short term drilling progress prediction(Y_corr) using acc cage(ac), acc PTU(ap), Force(F), Current(I)"""

    def __init__(self,
                 preprocess_acc_cage_args: DictConfig,
                 tokenization_acc_cage: DictConfig,
                 pe_acc_cage_temporal: DictConfig,
                 pe_acc_cage_spatial: DictConfig,
                 preprocess_acc_ptu_args: DictConfig,
                 tokenization_acc_ptu: DictConfig,
                 pe_acc_ptu_temporal: DictConfig,
                 pe_acc_ptu_spatial: DictConfig,
                 preprocess_force_args: DictConfig,
                 tokenization_force: DictConfig,
                 pe_force_temporal: DictConfig,
                 pe_force_spatial: DictConfig,
                 preprocess_current_args: DictConfig,
                 tokenization_current: DictConfig,
                 pe_current_temporal: DictConfig,
                 pe_current_spatial: DictConfig,
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
        self.ac_preprocess = torch.nn.Conv1d(kernel_size=12, stride=12, in_channels=6, out_channels=6)

        # self.fusion0 = torch.nn.LSTM(input_size=3+3+3+2, hidden_size=128, num_layers=1, bias=True, batch_first=True,
        #                             bidirectional=False, device=None, )

        self.fusionlist = torch.nn.Sequential(torch.nn.Conv1d(kernel_size=5, stride=5,
                                                              in_channels=11, out_channels=32),
                                              torch.nn.Tanh(),  # 600 32
                                              torch.nn.Conv1d(kernel_size=5, stride=5,
                                                              in_channels=32, out_channels=64),
                                              torch.nn.Tanh(),  # 120 64
                                              torch.nn.Conv1d(kernel_size=5, stride=5,
                                                              in_channels=64, out_channels=128),
                                              torch.nn.ReLU(),  # 24  128
                                              torch.nn.Conv1d(kernel_size=3, stride=3,
                                                              in_channels=128, out_channels=256),
                                              torch.nn.ReLU(),  # 8   256
                                              torch.nn.Conv1d(kernel_size=2, stride=2,
                                                              in_channels=256, out_channels=512),
                                              torch.nn.ReLU(),  # 4   512
                                              torch.nn.Conv1d(kernel_size=2, stride=2,
                                                              in_channels=512, out_channels=1024),
                                              torch.nn.ReLU(),  # 2   1024
                                              torch.nn.Conv1d(kernel_size=2, stride=2,
                                                              in_channels=1024, out_channels=2048),
                                              torch.nn.Flatten(),
                                              torch.nn.Linear(in_features=2048, out_features=1), )

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
                ):
        batch_size = acc_ptu_x.shape[0]

        acc_cage = torch.stack([acc_cage_x, acc_cage_y, acc_cage_z, acc_ptu_x, acc_ptu_y, acc_ptu_z], dim=1)
        acc_cage = torch.nn.functional.interpolate(acc_cage, [36000], mode='linear')
        acc_cage = self.ac_preprocess(acc_cage)
        x = torch.stack([f_x, f_y, f_z, i_s, i_z], dim=1)
        x = torch.cat([acc_cage, x], dim=1)  # N 11 3000
        # x= self.fusion0(x.permute(0, 2, 1))[0].permute(0, 2, 1)  # N 3000 32
        x = self.fusionlist(x)
        return x, 0
