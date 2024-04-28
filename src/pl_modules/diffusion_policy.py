import hydra.utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as pl
import logging
from utils.metrics import top_k_accuracy
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from typing import Dict, Tuple, Callable

import math
from einops import rearrange, reduce
import copy
import logging
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


class DiffusionPolicyFramework(pl.LightningModule):

    def __init__(self, mdl: nn.Module, optimizer, scheduler,
                 train_loader, val_loader, test_loader,
                 train_tasks, mask_type,
                 weight,
                 action="real_delta_target",
                 ema=False,
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
        self._dummy_variable = nn.Parameter()
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.last_train_batch = None
        self.validation_epoch_outputs = []
        self.validation_preds = []
        self.train_tasks = train_tasks
        self.mask_type = mask_type
        self.weight = weight
        self.mdl = mdl
        self.kwargs = kwargs
        self.ema_mdl = copy.deepcopy(self.mdl) if ema else None
        self.ema = hydra.utils.instantiate(ema, model=self.ema_mdl) if ema else None
        self.automatic_optimization = False
        self.action = action

    def _calculate_loss(self, batch, mode, ema="",  batch_idx=None,):
        self.last_train_batch = batch
        total_loss = 0
        metrics = {}
        # normalize input
        assert 'valid_mask' not in batch

        chunk_len = batch["traj"]["gripper"]["action"].shape[1]
        multimod_inputs = {
            "vision": batch["observation"]["v_fix"],
            "qpos": torch.cat([batch["traj"]["target_glb_pos_ori"]["obs"], batch["traj"]["gripper"]["obs"][..., :1]], dim= -1).float()
        }

        gripper_action = torch.cat([batch["traj"]["gripper"]["obs"][:, 1:, :][..., :1], batch["traj"]["gripper"]["action"][:, :, :1]], dim=1)
        if self.action == "real_delta_target":
            action = torch.cat([batch["traj"]["target_real_delta"]["obs"][:, 1:, :], batch["traj"]["target_real_delta"]["action"]], dim=1)
        elif self.action == "position":
            action = torch.cat([batch["traj"]["target_glb_pos_ori"]["obs"][:, 1:, :], batch["traj"]["target_glb_pos_ori"]["action"]], dim=1)
        elif self.action == "real_delta_source":
            action = torch.cat([batch["traj"]["source_real_delta"]["obs"][:, 1:, :], batch["traj"]["source_real_delta"]["action"]], dim=1)
        elif self.action == "direct_vel":
            action = torch.cat([batch["traj"]["direct_vel"]["obs"][:, 1:, :],batch["traj"]["direct_vel"]["action"]], dim=1)
        action = torch.cat([action, gripper_action], dim=-1).float()

        task = self.train_tasks.split("+")

        mdl = self.mdl
        if ema == "ema" and self.ema is not None and mode != "train":
            mdl = self.ema_mdl

        mdl_out, target, loss_mask = mdl(actions=action,
                                          multimod_inputs=multimod_inputs,
                                          mask=self.mask_type,
                                          task=task,
                                          mode=mode, )
        pred = mdl_out["pred"]
        metrics.update(mdl_out["obs_encoder_out"]["ssl_losses"])

        diffusion_loss = F.mse_loss(pred, target, reduction='none')
        diffusion_loss = diffusion_loss * loss_mask.type(diffusion_loss.dtype).to(diffusion_loss.device)
        diffusion_loss = reduce(diffusion_loss, 'b ... -> b (...)', 'mean')
        diffusion_loss = diffusion_loss.mean()
        metrics["diffusion_loss"] = diffusion_loss
        total_loss += diffusion_loss
        # if "imitation" in task:
        #     loss, immi_loss, aux_loss = self.compute_loss(
        #         demo, output["predict"]["action_logits"], xyzrpy_gt, output["predict"]["xyzrpy"]
        #     )
        #     total_loss += loss
        #     action_logits = output["predict"]["action_logits"]
        #     action_pred = torch.argmax(output["predict"]["action_logits"], dim=1)
        #     acc = (action_pred == demo).sum() / action_pred.numel()
        #     top_1_accu = top_k_accuracy(action_logits, demo, 1)
        #     top_3_accu = top_k_accuracy(action_logits, demo, 3)
        #     top_5_accu = top_k_accuracy(action_logits, demo, 5)
        #     self.log_dict({
        #         f"{mode}_sup_loss": loss,
        #         f"{mode}_immi_loss": immi_loss,
        #         f"{mode}_aux_loss": aux_loss,
        #         f"{mode}_acc": acc,
        #         f"{mode}_top_1_acc": top_1_accu,
        #         f"{mode}_top_3_acc": top_3_accu,
        #         f"{mode}_top_5_acc": top_5_accu,
        # 
        #     })
        #     step_output["imitation_acc"] = acc
        if mode == "val" and batch_idx % chunk_len ==0:
            gt_action = action[:, -chunk_len:, :]
            mdl = self.mdl
            if self.ema is not None:
                mdl = self.ema_mdl

            result = mdl.predict_action(multimod_inputs)
            pred_action = result['action']
            l1_loss = torch.nn.functional.l1_loss(pred_action, gt_action).mean()
            self.log("val_l1_error", l1_loss)


        for key, value in mdl_out["obs_encoder_out"]["ssl_losses"].items():
            total_loss += value * self.weight[key]

        if mode == "train":
            self.log("learning_rate", self.scheduler.get_last_lr()[0], on_step=True, prog_bar=True)
        metrics["total_loss"] = total_loss
        mod_metric = {}
        for key, value in metrics.items():
            mod_metric[f"{mode}_{key}"] = value
        self.log_dict(mod_metric)

        return metrics

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def configure_optimizers(self):
        """ configure the optimizer and scheduler """

        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader

    def training_step(self, batch, batch_idx):
        """ Calculate training loss and accuracy after each batch """
        opt = self.optimizers()
        sch = self.lr_schedulers()

        train_step_output = self._calculate_loss(batch, mode="train")
        opt.zero_grad()
        self.manual_backward(train_step_output["total_loss"])
        opt.step()
        sch.step()

        if self.ema is not None:
            self.ema.step(self.mdl)

    def validation_step(self, batch, batch_idx):
        """ Calculate validation loss and accuracy after each batch
            Also store the intermediate validation accuracy and prediction results of first sample of the batch
        """
        if self.ema is not None:
            val_step_output_ema = self._calculate_loss(batch, mode="val", ema="ema", batch_idx=batch_idx)
        val_step_output = self._calculate_loss(batch, mode="val", batch_idx=batch_idx)



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

