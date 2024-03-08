import hydra.utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as pl
import logging
from utils.metrics import top_k_accuracy
import copy
import logging
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


class TransformerPredictorPl(pl.LightningModule):

    def __init__(self, mdl: nn.Module, optimizer, scheduler,
                 train_loader, val_loader, test_loader,
                 train_tasks, masked_train,
                 weight, ema,
                 **kwargs):
        """ The pytorch lighting module that configures the model and its training configuration.

        Inputs:
            mdl: the model to be trained or tested
            optimizer: the optimizer e.g. Adam
            scheduler: scheduler for learning rate schedule
            train_loader: Dataloader for training dataset
            val_loader: Dataloader for validation dataset
            test_loader: Dataloader for test dataset
        """
        super().__init__()
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.mdl = mdl
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_tasks = train_tasks
        self.masked_train = masked_train
        self.weight = weight
        self.ema_mdl = copy.deepcopy(self.mdl) if ema else None
        self.ema = hydra.utils.instantiate(ema, model=self.ema_mdl) if ema else None

    def configure_optimizers(self):
        """ configure the optimizer and scheduler """

        return {"optimizer": self.optimizer,
                "lr_scheduler": {"scheduler": self.scheduler,
                                 "interval": "step"},
                }

    def compute_loss(self, delta, output):
        aux_loss = F.l1_loss(delta, output)
        return aux_loss

    def _calculate_loss(self, batch, mode="train", ema=""):
        """ Calculation of loss and prediction accuracy using output and label"""
        # Fetch data and transform categories to one-hot vectors
        total_loss = 0
        metrics = {}

        inp_data = batch["observation"]
        delta = batch["target_delta_seq"][:, 1].float()
        vf_inp, vg_inp, _, _ = inp_data
        multimod_inputs = {
            "vision": [vf_inp, vg_inp],
        }
        task = self.train_tasks.split("+")

        mdl = self.mdl
        if ema == "EMA" and self.ema is not None and mode != "train":
            mdl = self.ema_mdl

        # Perform prediction and calculate loss and accuracy
        output = mdl(multimod_inputs,
                     mask=self.masked_train,
                     task=task,
                     mode=mode,
                     )

        if "imitation" in task:
            loss = self.compute_loss(delta, output["predict"]["xyzrpy"])
            total_loss += loss
            metrics[f"imitation_loss{ema}"] = loss

        for key, value in output["ssl_losses"].items():
            total_loss += value * self.weight[key]
            metrics[f"{key}{ema}"] = value

        if mode == "train":
            self.log("learning_rate", self.scheduler.get_last_lr()[0], on_step=True, prog_bar=True)
        metrics[f"total_loss{ema}"] = total_loss
        mod_metric = {}
        for key, value in metrics.items():
            mod_metric[f"{mode}_{key}"] = value
        self.log_dict(mod_metric)

        return metrics

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader

    def training_step(self, batch, batch_idx):
        """ Calculate training loss and accuracy after each batch """
        train_metrics = self._calculate_loss(batch, mode="train")
        return train_metrics["total_loss"]

    def validation_step(self, batch, batch_idx):
        """ Calculate validation loss and accuracy after each batch
            Also store the intermediate validation accuracy and prediction results of first sample of the batch
        """
        val_step_output = self._calculate_loss(batch, mode="val")
        if self.ema is not None:
            val_step_output_ema = self._calculate_loss(batch, mode="val", ema="EMA")

    def on_validation_epoch_end(self) -> None:
        """ Calculate the validation accuracy after an entire epoch.

        Returns: validation accuracy of an entire epoch

        """


        for name, value in self.trainer.callback_metrics.items():
            logger.info(f'{name} at epoch {self.current_epoch}:, {float(value.item())}')

    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        if self.ema is not None:
            self.ema.step(self.mdl)
