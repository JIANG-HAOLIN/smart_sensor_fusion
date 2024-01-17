import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as pl
import logging

import logging

logger = logging.getLogger(__name__)


class TransformerPredictorPl(pl.LightningModule):

    def __init__(self, mdl: nn.Module, optimizer, scheduler,
                 train_loader, val_loader, test_loader, **kwargs):
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
        self.num_classes = kwargs['num_classes']
        self.num_stack = kwargs['num_stack']
        self.validation_epoch_outputs = []
        self.validation_preds = []
        self.loss_cce = torch.nn.CrossEntropyLoss()

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
        inp_data, demo, xyzrpy_gt, optical_flow, start, labels = batch
        vf_inp, vg_inp, t_inp, audio_g, audio_h = inp_data
        multimod_inputs = {
            "vision": vg_inp,
            "tactile": t_inp,
            "audio": audio_h
        }
        task = ("bind", 'order', 'fuse_nce', 'cross_time_nce', 'recover', 'imitation')
        # Perform prediction and calculate loss and accuracy
        output = self.mdl.forward(multimod_inputs,
                                  mask=True,
                                  task=task,
                                  mode=mode,
                                  )
        step_output = {}
        total_loss = 0
        if "imitation" in task:
            loss, immi_loss, aux_loss = self.compute_loss(
                demo, output["predict"]["action_logits"], xyzrpy_gt, output["predict"]["xyzrpy"]
            )
            total_loss = immi_loss + aux_loss
            action_pred = torch.argmax(output["predict"]["action_logits"], dim=1)
            acc = (action_pred == demo).sum() / action_pred.numel()
            self.log_dict({
                f"{mode}_immi_loss": immi_loss,
                f"{mode}_aux_loss": aux_loss,
                f"{mode}_acc": acc,
            })
            step_output["imitation_acc"] = acc

        for key, value in output["ssl_losses"].items():
            ssl_losses_dict = {}
            total_loss += value
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
        val_loss = sum(self.validation_epoch_outputs) / len(self.validation_epoch_outputs)
        self.validation_epoch_outputs.clear()
        self.validation_preds.clear()

        for name, value in self.trainer.callback_metrics.items():
            logger.info(f'{name} at epoch {self.current_epoch}:, {float(value.item())}')