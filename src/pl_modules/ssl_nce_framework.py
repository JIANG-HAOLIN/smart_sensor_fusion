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
        self.num_stack = kwargs['num_stack']
        self.validation_epoch_outputs = []
        self.validation_preds = []
        self.loss_cce = torch.nn.CrossEntropyLoss()
        self.train_tasks = train_tasks
        self.masked_train = masked_train
        self.weight = weight
        self.ema_mdl = copy.deepcopy(self.mdl) if ema else None
        self.ema = hydra.utils.instantiate(ema, model=self.ema_mdl) if ema else None

    def configure_optimizers(self):
        """ configure the optimizer and scheduler """

        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def compute_loss(self, demo, pred_logits, xyz_gt, xyz_pred):
        immi_loss = self.loss_cce(pred_logits, demo)
        aux_loss = F.mse_loss(xyz_gt, xyz_pred)
        return immi_loss + aux_loss * 1.0, immi_loss, aux_loss

    def _calculate_loss(self, batch, mode="train"):
        """ Calculation of loss and prediction accuracy using output and label"""
        # Fetch data and transform categories to one-hot vectors
        inp_data = batch["observation"]
        demo = batch["action"]
        xyzrpy_gt = batch["pose"]
        vf_inp, vg_inp, t_inp, audio_g, audio_h = inp_data
        multimod_inputs = {
            "vision": vg_inp,
            "tactile": t_inp,
            "audio": audio_h
        }
        task = self.train_tasks.split("+")

        mdl = self.mdl
        if self.ema is not None and mode != "train":
            mdl = self.ema_mdl

        # Perform prediction and calculate loss and accuracy
        output = mdl(multimod_inputs,
                     mask=self.masked_train,
                     task=task,
                     mode=mode,
                     )
        step_output = {}
        total_loss = 0
        if "imitation" in task:
            loss, immi_loss, aux_loss = self.compute_loss(
                demo, output["predict"]["action_logits"], xyzrpy_gt, output["predict"]["xyzrpy"]
            )
            total_loss += loss
            action_logits = output["predict"]["action_logits"]
            action_pred = torch.argmax(output["predict"]["action_logits"], dim=1)
            acc = (action_pred == demo).sum() / action_pred.numel()
            top_1_accu = top_k_accuracy(action_logits, demo, 1)
            top_3_accu = top_k_accuracy(action_logits, demo, 3)
            top_5_accu = top_k_accuracy(action_logits, demo, 5)
            self.log_dict({
                f"{mode}_sup_loss": loss,
                f"{mode}_immi_loss": immi_loss,
                f"{mode}_aux_loss": aux_loss,
                f"{mode}_acc": acc,
                f"{mode}_top_1_acc": top_1_accu,
                f"{mode}_top_3_acc": top_3_accu,
                f"{mode}_top_5_acc": top_5_accu,

            })
            step_output["imitation_acc"] = acc

        ssl_losses_dict = {}
        for key, value in output["ssl_losses"].items():
            total_loss += value * self.weight[key]
            ssl_losses_dict[f"{mode}_{key}"] = value
        self.log_dict(ssl_losses_dict)

        step_output["total_loss"] = total_loss

        return step_output

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader

    def training_step(self, batch, batch_idx):
        """ Calculate training loss and accuracy after each batch """
        train_step_output = self._calculate_loss(batch, mode="train")
        return train_step_output["total_loss"]

    def validation_step(self, batch, batch_idx):
        """ Calculate validation loss and accuracy after each batch
            Also store the intermediate validation accuracy and prediction results of first sample of the batch
        """
        val_step_output = self._calculate_loss(batch, mode="val")
        self.validation_epoch_outputs.append(val_step_output["imitation_acc"])

    def on_validation_epoch_end(self) -> None:
        """ Calculate the validation accuracy after an entire epoch.

        Returns: validation accuracy of an entire epoch

        """
        val_acc = sum(self.validation_epoch_outputs) / len(self.validation_epoch_outputs)
        self.validation_epoch_outputs.clear()
        self.validation_preds.clear()

        for name, value in self.trainer.callback_metrics.items():
            logger.info(f'{name} at epoch {self.current_epoch}:, {float(value.item())}')

    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        if self.ema is not None:
            self.ema.step(self.mdl)
