import hydra.utils
from hydra import initialize, compose
import unittest
import torch


class TestVisionAudio(unittest.TestCase):
    def test_vision_audio(self):
        with initialize(version_base='1.2', config_path="../../../../configs/datasets/"):
            # config is relative to a module
            cfg = compose(config_name="see_hear_feel_vision_audio", overrides=["dataloader.batch_size=1",
                                                                               "dataloader.data_folder='/fs/scratch"
                                                                               "/rng_cr_bcai_dl_students/jin4rng/data/'"])
            train_loader, val_loader, _ = hydra.utils.instantiate(cfg.dataloader)
            for idx, data in enumerate(train_loader):
                self.assertEqual(data[0][1].shape[1:], torch.Size([cfg.dataloader.args.num_stack, 3, 67, 90]))
                self.assertEqual(data[0][4].shape[1:], torch.Size([1, 40000]))
            for idx, data in enumerate(val_loader):
                self.assertEqual(data[0][1].shape[1:], torch.Size([cfg.dataloader.args.num_stack, 3, 67, 90]))
                self.assertEqual(data[0][4].shape[1:], torch.Size([1, 40000]))


if __name__ == '__main__':
    unittest.main()
