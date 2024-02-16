import unittest
import torch
import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf
from src.models.trafo_predictor import TransformerPredictor as NewTransformerPredictor
from src.models.vit_implementations import Vit_Classifier, VitImageBind, VitVATT3D, Vit


class Test_diffusion_encoder_decoder(unittest.TestCase):
    """
    If directly use the config file in models folder, model_dim can not be correctly interpolated
    """

    def test_vision_audio_fusion(self):
        """Test diffusion transformer encoder-decoder structure with config """
        with initialize(version_base='1.2', config_path="../../../../configs/"):
            # config is relative to a module
            cfg = compose(config_name="config_diffusion",
                          overrides=['models=diffusion/diffusion_encoder_decoder'])
            mdl = hydra.utils.instantiate(cfg.models.model, _recursive_=False)
            bs = 2
            multi_mod_input = {"vision": torch.randn([bs, 10, 3, 67, 90]),
                               "audio": torch.randn([bs, 1, 80000]),
                               "tactile": torch.randn([bs, 10, 3, 54, 72]), }
            Tp = 100
            act_dim = 2
            actions = torch.randn([bs, Tp, act_dim])
            timesteps = torch.randint(
                0, 100,
                (bs,), device=actions.device
            ).long()
            out = mdl(sample=actions,
                      timestep=timesteps,
                      multimod_inputs=multi_mod_input,
                      mask=None,
                      mask_type="input_mask",
                      task="imitation",
                      mode="train", )

            self.assertEqual(torch.Size([bs, Tp, act_dim]), out.shape, )


if __name__ == '__main__':
    unittest.main()
