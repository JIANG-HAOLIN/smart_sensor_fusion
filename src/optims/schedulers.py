import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """Cosine warmup scheduler"""
    def __init__(self, optimizer, warmup, max_iters, **kwargs):
        """
        Args:
            optimizer: which optimizer to schedule
            warmup: learning rate of [1, warmup] iters will be linear increasing
            max_iters: the maximum iteration of training, which decide the wave length of cosine function
            **kwargs: contains the other arguments
        """
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


if __name__ == '__main__':
    p = nn.Parameter(torch.empty(4, 4))
    optimizer = optim.Adam([p], lr=1e-4)
    lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=500, max_iters=10000)

    # Plotting
    epochs = list(range(4000))
    plt.figure(figsize=(8, 3))
    plt.plot(epochs, [lr_scheduler.get_lr_factor(e) for e in epochs])
    plt.ylabel("Learning rate factor")
    plt.xlabel("Iterations (in batches)")
    plt.title("Cosine Warm-up Learning Rate Scheduler")
    plt.show()
