import torch.optim as optim
import numpy as np

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