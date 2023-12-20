import os
import lightning as pl
import numpy as np
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, Timer
from utils.my_callbacks import MyProgressBar, MyEpochTimer, SaveBestTxt
from datetime import datetime
from typing import Optional
from omegaconf import DictConfig
from utils.hydra_utils import extract_sweeper_output_label


def launch_trainer(pl_module: pl.LightningModule,
                   out_dir_path: str,
                   model_name: str, dataset_name: str, task_name: str,
                   max_epochs: int,
                   monitor: str,
                   label: str,
                   resume: Optional[str] = None,
                   mode: str = 'max',
                   save_top_k: int = 1,
                   hydra_conf: Optional[DictConfig] = None,
                   **kwargs) -> None:
    """ Construct the trainer and start the training process.

    Args:
        save_top_k: save top K best model as .ckpt
        mode: is the monitored metrics lower the better or the opposite
        pl_module:  the pytorch lighting module, which contains the network and train/val step and datasets etc.
        out_dir_path: the path of output folder where the tensorboard log, the best epoch, and hyper
                      parameters are stored
        model_name:  the name of the model, can be found in .yaml file inside configs/models
        dataset_name: the name of the dataset, can be found in .yaml file inside configs/datasets
        task_name: the name of the task(transpose etc.), can be found in config_progress_prediction.yaml file in configs folder
        max_epochs: the maximum number of epoch for training
        monitor: which validation metrics to be monitored
        label: label for the output file contains variable names and value for multirun setup
        resume: the path of model parameters for inference
        hydra_conf: the hydra config file for single run
        **kwargs: other keyword arguments
    """
    jobid = os.environ.get("SLURM_JOB_ID", 0)
    exp_time = datetime.now().strftime("%m-%d-%H:%M:%S") + "-jobid=" + str(jobid)
    checkpoints_out_path = os.path.join(out_dir_path, 'checkpoints', )
    checkpoint = ModelCheckpoint(
        dirpath=checkpoints_out_path,
        # filename='best' + exp_time + "-{epoch}-{step}",
        filename='best_' + label + exp_time,
        save_top_k=save_top_k,
        save_last=True,
        monitor=monitor,
        mode=mode,
    )
    tensorboard_logger = TensorBoardLogger(
        save_dir=out_dir_path,
        version=label + exp_time, name="lightning_tensorboard_logs"
    )
    csv_logger = CSVLogger(save_dir=out_dir_path, version=label + exp_time, name="csv_logs")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint, MyProgressBar(), MyEpochTimer(), SaveBestTxt(checkpoints_out_path, label),
                   ],
        default_root_dir=model_name,
        accelerator='gpu',
        devices=-1,
        strategy="auto",
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        logger=[tensorboard_logger, csv_logger],
    )
    trainer.fit(
        pl_module,
        ckpt_path=None
        if resume is None
        else resume,
    )
