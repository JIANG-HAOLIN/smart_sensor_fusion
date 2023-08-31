import torch
from src.models.trafo_classifier_vit import TransformerClassifierVit, TransformerClassifierVit_Mel
from src.models.utils.mel_spec import MelSpec
from src.models.encoders.res_net_18 import make_audio_encoder
from omegaconf import DictConfig, OmegaConf


class classification_model(torch.nn.Module):
    def __init__(self,
                 preprocess_args: DictConfig,
                 encoder_args: DictConfig,
                 transformer_args: DictConfig,
                 **kwargs):
        super().__init__()
        self.preprocess = MelSpec(**preprocess_args)
        self.encoder = make_audio_encoder(**encoder_args)
        self.transformer = TransformerClassifierVit(**transformer_args)

    def forward(self, x):
        x = self.preprocess(x)
        x = self.encoder(x)
        return self.transformer(x)


class time_patch_model(torch.nn.Module):
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
