import hydra.utils
from hydra import initialize, compose
import unittest
import torch


class TestVisionAudio(unittest.TestCase):
    import random
    import numpy as np
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    def test_vision_audio(self):

        with initialize(version_base='1.2', config_path="../../../../configs/datasets/"):
            # config is relative to a module

            cfg = compose(config_name="see_hear_feel", overrides=["dataloader.batch_size=32",
                                                                               "dataloader.data_folder='/fs/scratch"
                                                                               "/rng_cr_bcai_dl_students/jin4rng/data/'",
                                                                               "dataloader.args.len_lb=20"])
            train_loader, val_loader, _ = hydra.utils.instantiate(cfg.dataloader)
            for idx, data in enumerate(train_loader):
                self.assertEqual(torch.Size([32, cfg.dataloader.args.num_stack, 3, 67, 90]),
                                 data["observation"][1].shape)
                self.assertEqual(torch.Size([32, 1, 40000]), data["observation"][4].shape)
                self.assertEqual(torch.Size([32, 20]), data["action_seq"].shape)
                self.assertEqual(torch.Size([32, 20, 6]), data["pose_seq"].shape)
            for idx, data in enumerate(val_loader):
                self.assertEqual(torch.Size([1, cfg.dataloader.args.num_stack, 3, 67, 90]),
                                 data["observation"][1].shape)
                self.assertEqual(torch.Size([1, 1, 40000]), data["observation"][4].shape)
                self.assertEqual(torch.Size([1, 20]), data["action_seq"].shape)
                self.assertEqual(torch.Size([1, 20, 6]), data["pose_seq"].shape)

        with initialize(version_base='1.2', config_path="../../../../configs/datasets/"):
            # config is relative to a module

            cfg = compose(config_name="see_hear_feel", overrides=["dataloader.batch_size=32",
                                                                               "dataloader.data_folder='/fs/scratch"
                                                                               "/rng_cr_bcai_dl_students/jin4rng/data/'"])
            train_loader, val_loader, _ = hydra.utils.instantiate(cfg.dataloader)
            for idx, data in enumerate(train_loader):
                self.assertEqual(torch.Size([32, cfg.dataloader.args.num_stack, 3, 67, 90]),
                                 data["observation"][1].shape)
                self.assertEqual(torch.Size([32, 1, 40000]), data["observation"][4].shape)
                self.assertEqual(torch.Size([32, 1]), data["action_seq"].shape)
                self.assertEqual(torch.Size([32, 1, 6]), data["pose_seq"].shape)
            for idx, data in enumerate(val_loader):
                self.assertEqual(torch.Size([1, cfg.dataloader.args.num_stack, 3, 67, 90]),
                                 data["observation"][1].shape)
                self.assertEqual(torch.Size([1, 1, 40000]), data["observation"][4].shape)
                self.assertEqual(torch.Size([1, 1]), data["action_seq"].shape)
                self.assertEqual(torch.Size([1, 1, 6]), data["pose_seq"].shape)



if __name__ == '__main__':
    unittest.main()
