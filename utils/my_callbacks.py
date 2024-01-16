from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks import Callback
from lightning import Trainer
import sys
import time
import os
import numpy as np
import matplotlib.pyplot as plt


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
    """this hook is always executed before the main code defined in pl_modules, so at epoch n only the best_model_score
    only contains the best value appeared in epoch 0 --> n-1 """
    def __init__(self, out_dir_path: str, label: str):
        super().__init__()
        self.out_dir_path = out_dir_path
        self.label = label

    def on_validation_epoch_end(self, trainer: Trainer, pl_module):
        # Log the current best model score and path to a text file
        if not trainer.sanity_checking:
            if trainer.current_epoch > 0:
                txt_file_path = os.path.join(self.out_dir_path, f'best_{self.label}.text')
                with open(txt_file_path, 'a') as file:
                    file.write(f"Epoch: {trainer.current_epoch - 1}, "
                               f"Best Model Score: {trainer.checkpoint_callback.best_model_score:.8f}, "
                               f"Best Model Path: {trainer.checkpoint_callback.best_model_path}\n")

    def on_fit_end(self, trainer: Trainer, pl_module):
        # Log the current best model score and path to a text file
        txt_file_path = os.path.join(self.out_dir_path, f'best_{self.label}.text')
        with open(txt_file_path, 'a') as file:
            file.write(f"Epoch: {trainer.current_epoch - 1}, "
                       f"Best Model Score: {trainer.checkpoint_callback.best_model_score:.8f}, "
                       f"Best Model Path: {trainer.checkpoint_callback.best_model_path}\n")


class PlotMetric(Callback):
    """i want to plot the metrics of all the epochs including current epoch that's why i access the metric through log
    instead of checkpoints"""
    def __init__(self, out_dir_path: str,
                 wanted_metrics: tuple = ("loss", "acc"),
                 ):
        super().__init__()
        self.out_dir_path = out_dir_path
        self.wanted_metrics = wanted_metrics
        self.metrics = {}

    def on_validation_epoch_end(self, trainer: Trainer, pl_module):
        # print the validation metrics as curves in png forms and save them
        if not trainer.sanity_checking:
            num_metric = 0
            for metric, value in trainer.callback_metrics.items():
                if "val" in metric and any(element in metric for element in self.wanted_metrics):
                    num_metric += 1
                    value = value.detach().cpu().numpy()
                    if metric not in self.metrics.keys():
                        self.metrics[metric] = []
                    self.metrics[metric].append(value)
            fig, ax = plt.subplots(num_metric, 1, figsize = (5, 2.5*num_metric))
            for idx, (metric, values) in enumerate(self.metrics.items()):
                x = np.arange(len(values))
                y = np.asarray(values)
                ax[idx].plot(x, y, '-', label=f'{metric}', linewidth=0.2)
                ax[idx].set_xlabel('x')
                ax[idx].set_ylabel('y')
                if "loss" in metric:
                    best_pos = y.argmin()
                elif "acc" in metric:
                    best_pos = y.argmax()
                ax[idx].set_title(f'{metric}  = {y[best_pos]} at epoch {best_pos}')
                fig.savefig(os.path.join(self.out_dir_path, f'val_metrics.png'),
                            dpi=300,
                            bbox_inches='tight', )
            plt.clf()
            plt.close('all')
