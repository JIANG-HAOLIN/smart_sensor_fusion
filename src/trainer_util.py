import os
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from datetime import datetime
from typing import Optional


# def train_reverse(**kwargs):
#     # Create a PyTorch Lightning trainer with the generation callback
#     root_dir = os.path.join(CHECKPOINT_PATH, "ReverseTask")
#     os.makedirs(root_dir, exist_ok=True)
#     trainer = pl.Trainer(default_root_dir=root_dir,
#                          callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
#                          accelerator="gpu" if str(device).startswith("cuda") else "cpu",
#                          devices=1,
#                          max_epochs=10,
#                          gradient_clip_val=5)
#     trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need
#
#     # Check whether pretrained model exists. If yes, load it and skip training
#     pretrained_filename = os.path.join(CHECKPOINT_PATH, "ReverseTask.ckpt")
#     if os.path.isfile(pretrained_filename):
#         print("Found pretrained model, loading...")
#         model = ReversePredictor.load_from_checkpoint(pretrained_filename)
#     else:
#         model = ReversePredictor(max_iters=trainer.max_epochs * len(train_loader), **kwargs)
#         trainer.fit(model, train_loader, val_loader)
#
#     # Test best model on validation and test set
#     val_result = trainer.test(model, val_loader, verbose=False)
#     test_result = trainer.test(model, test_loader, verbose=False)
#     result = {"test_acc": test_result[0]["test_acc"], "val_acc": val_result[0]["test_acc"]}
#
#     model = model.to(device)
#     return model, result
#

def launch_trainer(pl_module: pl.LightningModule,
                   out_dir_path: str,
                   model_name: str, dataset_name: str, task_name: str,
                   max_epochs: int,
                   monitor: str = "val_acc", resume: Optional[str] = None,
                   project_path: str = '.', **kwargs) -> None:
    jobid = os.environ.get("SLURM_JOB_ID", 0)
    exp_date = datetime.now().strftime("_%m-%d_")
    exp_time = datetime.now().strftime("%m-%d-%H:%M:%S") + "-jobid=" + str(jobid)
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(out_dir_path, 'checkpoints',),
        filename=exp_time + "-{epoch}-{step}",
        save_top_k=4,
        save_last=True,
        monitor=monitor,
        mode="max",
    )

    logger = TensorBoardLogger(
        save_dir=out_dir_path,
        version=task_name+exp_time, name="lightning_tensorboard_logs"
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
        else os.path.join(os.getcwd(), resume),
    )
    print("best_model", checkpoint.best_model_path)
