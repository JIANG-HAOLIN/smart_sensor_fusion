import torch
import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf
import unittest


class TestProgressPrediction(unittest.TestCase):
    def test_progress_prediction(self):
        with initialize(version_base=None, config_path="../../../../configs/models/", job_name="test_model"):
            cfg = compose(config_name="vit")
            print(OmegaConf.to_yaml(cfg))
            mdl = hydra.utils.instantiate(cfg.model, _recursive_=False)
            input = torch.randn([2, 1, 40000])
            out = mdl(input)
            self.assertEqual(torch.Size([2, 10]), out[0].shape, )
            self.assertEqual(torch.Size([2, 2, 61, 61]), out[1][0].shape, )


class TestTimePatchModel(unittest.TestCase):
    def test_progress_prediction(self):
        with initialize(version_base=None, config_path="../../../../configs/models/", job_name="test_model"):
            cfg = compose(config_name="vit_time_patch_small_hop")
            print(OmegaConf.to_yaml(cfg))
            mdl = hydra.utils.instantiate(cfg.model, _recursive_=False)
            input = torch.randn([2, 1, 40000])
            out = mdl(input)
            h, w = mdl.preprocess.out_size
            self.assertEqual(torch.Size([2, 10]), out[0].shape, )
            self.assertEqual(torch.Size([2, 2, w + 1, w + 1]), out[1][0].shape)


class TestVisionAudioFusion(unittest.TestCase):
    def test_vision_audio_fusion(self):
        with initialize(version_base='1.2', config_path="../../../../configs/models/progress_vision_audio"):
            # config is relative to a module
            cfg = compose(config_name="mul2one_vit_audio_vision")
            mdl = hydra.utils.instantiate(cfg.model, _recursive_=False)
            input = (torch.randn([2, 1, 40000]), torch.randn([2, 3, 67, 90]))
            out = mdl(*input)
            self.assertEqual(torch.Size([2, 10]), out[0].shape, )


if __name__ == "__main__":
    unittest.main()
