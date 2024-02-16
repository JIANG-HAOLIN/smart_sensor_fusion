import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as pl
import logging
from utils.metrics import top_k_accuracy

import logging

logger = logging.getLogger(__name__)


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class AlohaPolicy(pl.LightningModule):

    def __init__(self, mdl: nn.Module, optimizer, scheduler,
                 train_loader, val_loader, test_loader,
                 train_tasks, mask_type,
                 weight,
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
        self.mask_type = mask_type
        self.weight = weight

    def configure_optimizers(self):
        """ configure the optimizer and scheduler """

        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def compute_loss(self, demo, pred_logits, xyz_gt, xyz_pred):
        immi_loss = self.loss_cce(pred_logits, demo)
        aux_loss = F.mse_loss(xyz_gt, xyz_pred)
        return immi_loss + aux_loss * 1.0, immi_loss, aux_loss

    def _calculate_loss(self, batch, mode="train"):
        """ Calculation of loss and prediction accuracy using output and label"""
        total_loss = 0
        metrics = {}

        # Fetch data and transform categories to one-hot vectors
        observation = batch["observation"]
        action = batch["action"]
        pose = batch["pose"]
        optical_flow = batch["optical_flow"]
        action_seq = batch["action_seq"]
        pose_seq = batch["pose_seq"]
        vf_inp, vg_inp, t_inp, audio_g, audio_h = observation
        multimod_inputs = {
            "vision": vg_inp,
            "tactile": t_inp,
            "audio": audio_h
        }
        task = self.train_tasks.split("+")

        # Perform prediction and calculate loss and accuracy
        if pose_seq is not None:  # training time

            qpos = pose_seq[:, 0, :]
            is_pad = torch.zeros([pose_seq.shape[0], pose_seq.shape[1]], device=qpos.device).bool()
            out = self.mdl(qpos,
                           multimod_inputs,
                           actions=pose_seq,
                           is_pad=is_pad,
                           mask=None,
                           mask_type=self.mask_type,
                           task=task,
                           mode=mode, )
            metrics.update(out["obs_encoder_out"]["ssl_losses"])
            a_hat, is_pad_hat, (mu, logvar) = out["vae_output"]
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            all_l1 = F.l1_loss(pose_seq, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            metrics['l1_loss'] = l1
            metrics['kl'] = total_kld[0]
            metrics['vae_loss'] = metrics['l1_loss'] + metrics['kl'] * self.weight["kl_divergence"]
            total_loss += metrics['vae_loss']
            for key, value in out["obs_encoder_out"]["ssl_losses"].items():
                total_loss += value * self.weight[key]

        metrics["total_loss"] = total_loss
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
        train_step_output = self._calculate_loss(batch, mode="train")
        return train_step_output["total_loss"]

    def validation_step(self, batch, batch_idx):
        """ Calculate validation loss and accuracy after each batch
            Also store the intermediate validation accuracy and prediction results of first sample of the batch
        """
        val_step_output = self._calculate_loss(batch, mode="val")
        self.validation_epoch_outputs.append(val_step_output["total_loss"])

    def on_validation_epoch_end(self) -> None:
        """ Calculate the validation accuracy after an entire epoch.

        Returns: validation accuracy of an entire epoch

        """
        val_loss = sum(self.validation_epoch_outputs) / len(self.validation_epoch_outputs)
        self.validation_epoch_outputs.clear()
        self.validation_preds.clear()

        for name, value in self.trainer.callback_metrics.items():
            logger.info(f'{name} at epoch {self.current_epoch}:, {float(value.item())}')
