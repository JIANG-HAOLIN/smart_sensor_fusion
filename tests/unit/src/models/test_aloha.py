import unittest
import torch
import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf
from src.models.trafo_predictor import TransformerPredictor as NewTransformerPredictor
from src.models.vit_implementations import Vit_Classifier, VitImageBind, VitVATT3D, Vit


class Test_DETRVAE(unittest.TestCase):
    """
    If directly use the config file in models folder, model_dim can not be correctly interpolated
    """

    def test_vae(self):
        """Test DETRVAE(Aloha) with config """
        with initialize(version_base='1.2', config_path="../../../../configs/"):
            # config is relative to a module
            cfg = compose(config_name="config_aloha",
                          overrides=['models=aloha/vae'])
            mdl = hydra.utils.instantiate(cfg.models.model, _recursive_=False)
            bs = 2
            multi_mod_input = {"vision": torch.randn([bs, 10, 3, 67, 90]),
                               "audio": torch.randn([bs, 1, 80000]),
                               "tactile": torch.randn([bs, 10, 3, 54, 72]), }
            qpos = torch.randn([bs, 6])
            actions = torch.randn([bs, 100, 6])
            is_pad = torch.zeros([bs, 100]).bool()
            out = mdl(qpos, multi_mod_input,
                      actions=actions,
                      is_pad=is_pad,
                      mask=None,
                      mask_type="input_mask",
                      task="imitation",
                      mode="train", )
            self.assertEqual(actions.shape, out["vae_output"][0].shape, )


if __name__ == '__main__':
    unittest.main()
