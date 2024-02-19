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
            len_lb = 20
            seq_len = len_lb + 20
            cfg = compose(config_name="see_hear_feel", overrides=["dataloader.batch_size=32",
                                                                               "dataloader.data_folder='/fs/scratch"
                                                                               "/rng_cr_bcai_dl_students/jin4rng/data/'",
                                                                               f"dataloader.args.len_lb={len_lb}"])
            train_loader, val_loader, _ = hydra.utils.instantiate(cfg.dataloader)
            for idx, data in enumerate(train_loader):
                self.assertEqual(torch.Size([32, cfg.dataloader.args.num_stack, 3, 67, 90]),
                                 data["observation"][1].shape)
                self.assertEqual(torch.Size([32, 1, 40000]), data["observation"][4].shape)
                self.assertEqual(torch.Size([32, seq_len]), data["action_seq"].shape)
                self.assertEqual(torch.Size([32, seq_len, 6]), data["pose_seq"].shape)
                self.assertEqual(torch.equal(data["action"], data["action_seq"][:, -len_lb]), True)
                self.assertEqual(torch.equal(data["pose"], data["pose_seq"][:, -len_lb, :]), True)
            for idx, data in enumerate(val_loader):
                self.assertEqual(torch.Size([1, cfg.dataloader.args.num_stack, 3, 67, 90]),
                                 data["observation"][1].shape)
                self.assertEqual(torch.Size([1, 1, 40000]), data["observation"][4].shape)
                self.assertEqual(torch.Size([1, seq_len]), data["action_seq"].shape)
                self.assertEqual(torch.Size([1, seq_len, 6]), data["pose_seq"].shape)
                self.assertEqual(torch.equal(data["action"], data["action_seq"][:, -len_lb]), True)
                self.assertEqual(torch.equal(data["pose"], data["pose_seq"][:, -len_lb, :]), True)



if __name__ == '__main__':
    unittest.main()
