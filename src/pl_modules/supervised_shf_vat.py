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
        self.num_images = kwargs['num_images']
        self.validation_epoch_accs = []
        self.validation_preds = []

        self.loss_cce = torch.nn.CrossEntropyLoss()

        self.wrong = 1
        self.correct = 0
        self.total = 0

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
        # Perform prediction and calculate loss and accuracy
        vf_inp, vg_inp, t_inp, audio_g, audio_h = inp_data

        action_logits, xyzrpy_pred, weights = self.mdl.forward(vg_inp, audio_h, t_inp)

        loss, immi_loss, aux_loss = self.compute_loss(
            demo, action_logits, xyzrpy_gt, xyzrpy_pred
        )

        action_pred = torch.argmax(action_logits, dim=1)
        acc = (action_pred == demo).sum() / action_pred.numel()
        self.log(f"{mode}_immi_loss:", immi_loss)
        self.log(f"{mode}_aux_loss:", aux_loss)
        self.log(f"{mode}_acc:", acc)
        return loss, immi_loss, aux_loss, acc, action_pred

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader

    def training_step(self, batch, batch_idx):
        """ Calculate training loss and accuracy after each batch """
        loss, immi_loss, aux_loss, acc, action_pred = self._calculate_loss(batch, mode="train")

        return loss

    def validation_step(self, batch, batch_idx):
        """ Calculate validation loss and accuracy after each batch
            Also store the intermediate validation accuracy and prediction results of first sample of the batch
        """
        val_loss, val_immi_loss, val_aux_loss, val_step_acc, preds = self._calculate_loss(batch, mode="val")
        self.validation_preds.append([batch[1][0], preds[0]])
        self.validation_epoch_accs.append(val_step_acc)

    def on_validation_epoch_end(self) -> None:
        """ Calculate the validation accuracy after an entire epoch.

        Returns: validation accuracy of an entire epoch

        """
        val_acc = sum(self.validation_epoch_accs) / len(self.validation_epoch_accs)
        self.validation_epoch_accs.clear()
        self.validation_preds.clear()
        self.log("val_acc", val_acc, on_step=False, on_epoch=True, prog_bar=True)

        logger.info(f'val_acc at epoch {self.current_epoch}:, {float(val_acc.item())}')

        return val_acc
