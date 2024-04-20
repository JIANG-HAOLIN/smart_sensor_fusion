import hydra.utils
from hydra import initialize, compose
import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from src.optims.schedulers import CosineWarmupScheduler


class TestSchedulers(unittest.TestCase):
    import random
    import numpy as np
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    def test_my_cosine_scheduler(self):
        with initialize(version_base='1.2', config_path="../../../../configs/datasets/"):
            # config is relative to a module
            len_lb = 20
            seq_len = len_lb + 20
            cfg = compose(config_name="see_hear_feel", overrides=["dataloader.batch_size=32",
                                                                  "dataloader.data_folder='/fs/scratch"
                                                                  "/rng_cr_bcai_dl_students/jin4rng/data/'",
                                                                  f"dataloader.args.len_lb={len_lb}"])
        p = nn.Parameter(torch.empty(4, 4))
        optimizer = optim.Adam([p], lr=1e-4)
        lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=500, max_iters=10000)

        # Plotting
        epochs = list(range(10000))
        plt.figure(figsize=(8, 3))
        lrs = []
        for e in epochs:
            lrs.append(lr_scheduler.get_last_lr())
            lr_scheduler.step()
        plt.plot(epochs, lrs)
        plt.ylabel("Learning rate factor")
        plt.xlabel("Iterations (in batches)")
        plt.title("step lr Learning Rate Scheduler")
        plt.show()

    def test_steplr_scheduler(self):
        with initialize(version_base='1.2', config_path="../../../../configs/optimizers/"):
            # config is relative to a module
            cfg = compose(config_name="adam_steplr", overrides=[])

            p = nn.Parameter(torch.empty(4, 4))
            optimizer = optim.Adam([p], lr=1e-4)

            lr_scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)
            # Plotting
            epochs = list(range(11400))
            plt.figure(figsize=(8, 3))
            lrs = []
            for e in epochs:
                lrs.append(lr_scheduler.get_last_lr())
                lr_scheduler.step()
            plt.plot(epochs, lrs)
            plt.ylabel("Learning rate factor")
            plt.xlabel("Iterations (in batches)")
            plt.title("step lr Learning Rate Scheduler")
            plt.show()

    def test_huggingface_cosine_scheduler(self):
        with initialize(version_base='1.2', config_path="../../../../configs/optimizers/"):
            # config is relative to a module
            cfg = compose(config_name="adam_cosine", overrides=["scheduler.num_warmup_steps=250",
                                                                "scheduler.num_training_steps=2500"])

            p = nn.Parameter(torch.empty(4, 4))
            optimizer = optim.Adam([p], lr=1e-4)

            lr_scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)
            # Plotting
            epochs = list(range(5000))
            plt.figure(figsize=(8, 3))
            lrs = []
            for e in epochs:
                lrs.append(lr_scheduler.get_last_lr())
                lr_scheduler.step()
            plt.plot(epochs, lrs)
            plt.ylabel("Learning rate factor")
            plt.xlabel("Iterations (in batches)")
            plt.title("step lr Learning Rate Scheduler")
            plt.show()

    def test_huggingface_cosine_restart_scheduler(self):
        with initialize(version_base='1.2', config_path="../../../../configs/optimizers/"):
            # config is relative to a module
            cfg = compose(config_name="adam_cosine_restart", overrides=["scheduler.num_warmup_steps=250",
                                                                        "scheduler.num_training_steps=2500"])

            p = nn.Parameter(torch.empty(4, 4))
            optimizer = optim.Adam([p], lr=1e-4)

            lr_scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)
            # Plotting
            epochs = list(range(5000))
            plt.figure(figsize=(8, 3))
            lrs = []
            for e in epochs:
                lrs.append(lr_scheduler.get_last_lr())
                lr_scheduler.step()
            plt.plot(epochs, lrs)
            plt.ylabel("Learning rate factor")
            plt.xlabel("Iterations (in batches)")
            plt.title("step lr Learning Rate Scheduler")
            plt.show()


if __name__ == '__main__':
    unittest.main()
