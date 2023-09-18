import os
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from datetime import datetime
from typing import Optional


def launch_trainer(pl_module: pl.LightningModule,
                   out_dir_path: str,
                   model_name: str, dataset_name: str, task_name: str,
                   max_epochs: int,
                   monitor: str = "val_acc", resume: Optional[str] = None,
                   **kwargs) -> None:
    """ Construct the trainer and start the training process.

    Args:
        pl_module:  the pytorch lighting module, which contains the network and train/val step and datasets etc.
        out_dir_path: the path of output folder where the tensorboard log, the best epoch, and hyper
                      parameters are stored
        model_name:  the name of the model, can be found in .yaml file inside configs/models
        dataset_name: the name of the dataset, can be found in .yaml file inside configs/datasets
        task_name: the name of the task(transpose etc.), can be found in config_progress_prediction.yaml file in configs folder
        max_epochs: the maximum number of epoch for training
        monitor: which validation metrics to be monitored
        resume: the path of model parameters for inference
        **kwargs: other keyword arguments
    """
    jobid = os.environ.get("SLURM_JOB_ID", 0)
    exp_time = datetime.now().strftime("%m-%d-%H:%M:%S") + "-jobid=" + str(jobid)
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(out_dir_path, 'checkpoints', ),
        filename=exp_time + "-{epoch}-{step}",
        save_top_k=4,
        save_last=True,
        monitor=monitor,
        mode="max",
    )

    logger = TensorBoardLogger(
        save_dir=out_dir_path,
        version=task_name + exp_time, name="lightning_tensorboard_logs"
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint],
        default_root_dir=model_name,
        accelerator='gpu',
        devices=-1,
        strategy="auto",
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        logger=logger,
    )
    trainer.fit(
        pl_module,
        ckpt_path=None
        if resume is None
        else resume,
    )
    print("best_model", checkpoint.best_model_path)


