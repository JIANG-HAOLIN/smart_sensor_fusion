import torch
import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf
import unittest


class TestProgressPrediction(unittest.TestCase):
    def test_progress_prediction(self):
        initialize(version_base=None, config_path="../../../../configs/models/", job_name="test_model")
        cfg = compose(config_name="vit_transformer_encoder")
        print(OmegaConf.to_yaml(cfg))
        mdl = hydra.utils.instantiate(cfg.model)
        input = torch.randn([2, 1, 40000])
        out = mdl(input)
        self.assertEqual(out[0].shape, torch.Size([2, 10]))
        self.assertEqual(out[1][0].shape, torch.Size([2, 2, 17, 17]))


class TestTimePatchModel(unittest.TestCase):
    def test_progress_prediction(self):
        initialize(version_base=None, config_path="../../../../configs/models/", job_name="test_model")
        cfg = compose(config_name="vit_time_patch_small_hop.yaml")
        print(OmegaConf.to_yaml(cfg))
        mdl = hydra.utils.instantiate(cfg.model)
        input = torch.randn([2, 1, 40000])
        out = mdl(input)
        h, w = mdl.preprocess.out_size
        self.assertEqual(torch.Size([2, 10]), out[0].shape, )
        self.assertEqual(torch.Size([2, 2, w+1,  w+1]), out[1][0].shape)


if __name__ == "__main__":
    unittest.main()
