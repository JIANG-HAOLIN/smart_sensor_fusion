from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning import Callback
import sys
import time


class MyProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar


class MyEpochTimer(Callback):
    def on_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
        print(self.epoch_start_time)

    def on_epoch_end(self, trainer, pl_module):
        elapsed_time = time.time() - self.epoch_start_time
        print(f"Epoch {trainer.current_epoch + 1} took {elapsed_time:.2f} seconds")
