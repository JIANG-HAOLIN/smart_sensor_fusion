from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks import Callback
from lightning import Trainer
import sys
import time
import os


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


class SaveBestTxt(Callback):
    def __init__(self, out_dir_path: str, label: str):
        super().__init__()
        self.out_dir_path = out_dir_path
        self.label = label

    def on_validation_epoch_end(self, trainer: Trainer, pl_module):
        # Log the current best model score and path to a text file
        if trainer.current_epoch > 0:
            txt_file_path = os.path.join(self.out_dir_path, f'best_{self.label}.text')
            with open(txt_file_path, 'a') as file:
                file.write(f"Epoch: {trainer.current_epoch}, "
                           f"Best Model Score: {trainer.checkpoint_callback.best_model_score:.8f}, "
                           f"Best Model Path: {trainer.checkpoint_callback.best_model_path}\n")

    def on_fit_end(self, trainer: Trainer, pl_module):
        # Log the current best model score and path to a text file
        txt_file_path = os.path.join(self.out_dir_path, f'best_{self.label}.text')
        with open(txt_file_path, 'a') as file:
            file.write(f"Epoch: {trainer.current_epoch}, "
                       f"Best Model Score: {trainer.checkpoint_callback.best_model_score:.8f}, "
                       f"Best Model Path: {trainer.checkpoint_callback.best_model_path}\n")
