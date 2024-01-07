import torch
import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf
import unittest


class TestProgressPrediction(unittest.TestCase):
    def test_progress_prediction(self):
        with initialize(version_base=None, config_path="../../../../configs/models/progress_prediction",
                        job_name="test_model"):
            cfg = compose(config_name="vit_64_51")
            print(OmegaConf.to_yaml(cfg))
            mdl = hydra.utils.instantiate(cfg.model, _recursive_=False)
            input = torch.randn([2, 1, 40000])
            out = mdl(input)
            self.assertEqual(torch.Size([2, 10]), out[0].shape, )
            self.assertEqual(torch.Size([2, 2, 49, 49]), out[1][0].shape, )


class TestTimePatchModel(unittest.TestCase):
    def test_progress_prediction(self):
        with initialize(version_base=None, config_path="../../../../configs/models/progress_prediction",
                        job_name="test_model"):
            cfg = compose(config_name="vit_time_patch_default")
            print(OmegaConf.to_yaml(cfg))
            mdl = hydra.utils.instantiate(cfg.model, _recursive_=False)
            input = torch.randn([2, 1, 40000])
            out = mdl(input)
            h, w = mdl.preprocess.out_size
            self.assertEqual(torch.Size([2, 10]), out[0].shape, )
            self.assertEqual(torch.Size([2, 2, w + 1, w + 1]), out[1][0].shape)


class TestVisionAudioFusion(unittest.TestCase):
    """
    If directly use the config file in models folder, model_dim can not be correctly interpolated
    """

    def test_vision_audio_fusion(self):
        """Test class VisionAudioFusion with config """
        with initialize(version_base='1.2', config_path="../../../../configs/"):
            # config is relative to a module
            cfg = compose(config_name="config_progress_prediction_vision_audio",
                          overrides=['models=progress_vision_audio/earlycat_vit_vgah_timeemb'])
            mdl = hydra.utils.instantiate(cfg.models.model, _recursive_=False)
            input = (torch.randn([2, 5, 3, 67, 90]), torch.randn([2, 1, 40000]),)
            out = mdl(*input)
            self.assertEqual(torch.Size([2, 10]), out[0].shape, )

    def test_vision_audio_fusion_earlysum(self):
        """Test class VisionAudioFusion_EarlySum with config earlysum_vit_nores.yaml"""
        with initialize(version_base='1.2', config_path="../../../../configs/"):
            # config is relative to a module
            cfg = compose(config_name="config_progress_prediction_vision_audio",
                          overrides=['models=progress_vision_audio/earlysum_vit_nores'])
            mdl = hydra.utils.instantiate(cfg.models.model, _recursive_=False)
            input = (torch.randn([2, 5, 3, 67, 90]), torch.randn([2, 1, 40000]),)
            out = mdl(*input)
            self.assertEqual(torch.Size([2, 10]), out[0].shape, )

    def test_vision_audio_fusion_earlysum2fuse(self):
        """Test class VisionAudioFusion_EarlySum2Fuse with config earlysum_vit_nores.yaml"""
        with initialize(version_base='1.2', config_path="../../../../configs/"):
            # config is relative to a module
            cfg = compose(config_name="config_progress_prediction_vision_audio",
                          overrides=['models=progress_vision_audio/earlyfuse_vit_nores'])
            mdl = hydra.utils.instantiate(cfg.models.model, _recursive_=False)
            input = (torch.randn([2, 5, 3, 67, 90]), torch.randn([2, 1, 40000]),)
            out = mdl(*input)
            self.assertEqual(torch.Size([2, 10]), out[0].shape, )

    def test_vision_audio_fusion_early_fuse(self):
        """Test class VisionAudioFusion_EarlyFuse which is a early implementation for early fusion method"""
        with initialize(version_base='1.2', config_path="../../../../configs/"):
            # config is relative to a module
            cfg = compose(config_name="config_progress_prediction_vision_audio",
                          overrides=['models=progress_vision_audio/earlyfuse_vit_vgah'])
            mdl = hydra.utils.instantiate(cfg.models.model, _recursive_=False)
            input = (torch.randn([2, 5, 3, 67, 90]), torch.randn([2, 1, 40000]),)
            out = mdl(*input)
            self.assertEqual(torch.Size([2, 10]), out[0].shape, )



class TestVisionAudioFusion_seehearfeel(unittest.TestCase):
    def test_vision_audio_fusion(self):
        with initialize(version_base='1.2', config_path="../../../../configs/"):
            # config is relative to a module
            cfg = compose(config_name="config_progress_prediction_vision_audio",
                          overrides=['models=progress_vision_audio/audio_vision_vanilla'])
            mdl = hydra.utils.instantiate(cfg.models.model, _recursive_=False)
            input = (torch.randn([2, 5, 3, 67, 90]), torch.randn([2, 1, 40000]),)
            out = mdl(*input)
            self.assertEqual(torch.Size([2, 10]), out[0].shape, )


class TestVisionAudioFusionTimeEmb(unittest.TestCase):
    def test_vision_audio_fusion_time_emb(self):
        with initialize(version_base='1.2', config_path="../../../../configs/"):
            # config is relative to a module
            cfg = compose(config_name="config_progress_prediction_vision_audio",
                          overrides=['models=progress_vision_audio/earlycat_vit_vgah_timeemb'])
            mdl = hydra.utils.instantiate(cfg.models.model, _recursive_=False)
            input = (torch.randn([2, 5, 3, 67, 90]), torch.randn([2, 1, 40000]),)
            out = mdl(*input)
            self.assertEqual(torch.Size([2, 10]), out[0].shape, )


if __name__ == "__main__":
    unittest.main()
