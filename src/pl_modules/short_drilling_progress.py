import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
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
        self.validation_epoch_losses = []
        self.validation_preds = []

    def configure_optimizers(self):
        """ configure the optimizer and scheduler """
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def _calculate_loss(self, batch, mode="train"):
        """ Calculation of loss and prediction accuracy using output and label"""
        # batch:
        # acx_C: torch.Size([32, 15000])
        # acy_C: torch.Size([32, 15000])
        # acz_C: torch.Size([32, 15000])
        #
        # Fz_C: torch.Size([32, 3000])
        # Fy_C: torch.Size([32, 3000])
        # Fx_C: torch.Size([32, 3000])
        #
        # Is_C: torch.Size([32, 3000])
        # Iz_C: torch.Size([32, 3000])
        #
        # s1_C: torch.Size([32, 3000])
        # s1_C_corr: torch.Size([32, 3000])
        # s2_C: torch.Size([32, 3000])
        # s2_C_corr: torch.Size([32, 3000])
        # Y_corr: torch.Size([32, 375])
        # Fetch data and transform categories to one-hot vectors

        short_time_progress = batch['Y_corr'][:, -1:] - batch['Y_corr'][:, 0:1]
        # print(short_time_progress.shape)
        labels = short_time_progress
        acc_cage_x = batch['apx_C']
        acc_cage_y = batch['apy_C']
        acc_cage_z = batch['apz_C']
        acc_ptu_x = batch['acx_C']
        acc_ptu_y = batch['acy_C']
        acc_ptu_z = batch['acz_C']
        f_x = batch['Fx_C']
        f_y = batch['Fy_C']
        f_z = batch['Fz_C']
        i_s = batch['Is_C']
        i_z = batch['Iz_C']
        # Perform regression
        out = self.mdl(acc_cage_x, acc_cage_y, acc_cage_z,
                       acc_ptu_x, acc_ptu_y, acc_ptu_z,
                       f_x, f_y, f_z,
                       i_s, i_z)
        preds = out[0]
        # print(preds.dtype, labels.dtype)
        loss = F.mse_loss(preds.view(-1, preds.size(-1)), labels).float()
        # print(loss.dtype)
        # Logging
        self.log(f"{mode}_loss", loss)
        if mode == 'val' or mode == 'test':
            return loss, preds
        return loss

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader

    def training_step(self, batch, batch_idx):
        """ Calculate training loss and accuracy after each batch """
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        """ Calculate validation loss and accuracy after each batch
            Also store the intermediate validation accuracy and prediction results of first sample of the batch
        """
        val_loss, preds = self._calculate_loss(batch, mode="val")
        self.validation_epoch_losses.append(val_loss)
        self.validation_preds.append(preds)

    def on_validation_epoch_end(self) -> None:
        """ Calculate the validation accuracy after an entire epoch.

        Returns: validation accuracy of an entire epoch

        """
        avg_val_loss = sum(self.validation_epoch_losses) / len(self.validation_epoch_losses)
        self.validation_epoch_losses.clear()
        self.log("avg_val_loss", avg_val_loss, on_step=False, on_epoch=True, prog_bar=True)

        logger.info(f'avg_val_loss at epoch {self.current_epoch}:, {float(avg_val_loss.item())}')

        return avg_val_loss
