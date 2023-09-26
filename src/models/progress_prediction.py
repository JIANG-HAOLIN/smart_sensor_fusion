import torch
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
        vision_signal = vision_signal.view(batch_size, num_stack*vision_signal.shape[-2], vision_signal.shape[-1])
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
