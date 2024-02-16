import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as pl
import logging
from utils.metrics import top_k_accuracy
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from typing import Dict, Tuple

import math
from einops import rearrange, reduce

# from diffusion_policy.model.common.normalizer import LinearNormalizer
from src.models.diffusion_encoder_decoder import TransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator

from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules

import logging

logger = logging.getLogger(__name__)


class DiffusionTransformerHybridImagePolicy(pl.LightningModule):

    def __init__(self, mdl: nn.Module, optimizer, scheduler,
                 train_loader, val_loader, test_loader,
                 train_tasks, mask_type,
                 weight,
                 ######################################
                 shape_meta: dict,
                 noise_scheduler: DDPMScheduler,
                 # task params
                 horizon,
                 n_action_steps,
                 n_obs_steps,
                 num_inference_steps=None,

                 obs_as_cond=True,
                 pred_action_steps_only=False,
                 #########################################
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
        self.num_stack = kwargs['num_stack']
        self.validation_epoch_outputs = []
        self.validation_preds = []
        self.loss_cce = torch.nn.CrossEntropyLoss()
        self.train_tasks = train_tasks
        self.mask_type = mask_type
        self.weight = weight

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        obs_feature_dim = mdl.obs_encoder.output_dim

        self.obs_encoder = mdl.obs_encoder
        self.mdl = mdl
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        # self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= inference  ============
    def conditional_sample(self,
                           condition_data, condition_mask,
                           cond=None, generator=None,
                           # keyword arguments to scheduler.step
                           **kwargs
                           ):
        model = self.mdl
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator,
                **kwargs
            ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict  # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            cond = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da + Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            cond=cond,
            **self.kwargs)

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    # def set_normalizer(self, normalizer: LinearNormalizer):
    #     self.normalizer.load_state_dict(normalizer.state_dict())

    # def get_optimizer(
    #         self,
    #         transformer_weight_decay: float,
    #         obs_encoder_weight_decay: float,
    #         learning_rate: float,
    #         betas: Tuple[float, float]
    # ) -> torch.optim.Optimizer:
    #     optim_groups = self.mdl.get_optim_groups(
    #         weight_decay=transformer_weight_decay)
    #     optim_groups.append({
    #         "params": self.obs_encoder.parameters(),
    #         "weight_decay": obs_encoder_weight_decay
    #     })
    #     optimizer = torch.optim.AdamW(
    #         optim_groups, lr=learning_rate, betas=betas
    #     )
    #     return optimizer

    def _calculate_loss(self, batch, mode):
        total_loss = 0
        metrics = {}
        # normalize input
        assert 'valid_mask' not in batch

        observation = batch["observation"]
        action = batch["action"]
        pose = batch["pose"]
        optical_flow = batch["optical_flow"]
        action_seq = batch["action_seq"]
        pose_seq = batch["pose_seq"]
        nactions = pose_seq
        vf_inp, vg_inp, t_inp, audio_g, audio_h = observation
        multimod_inputs = {
            "vision": vg_inp,
            "tactile": t_inp,
            "audio": audio_h
        }
        task = self.train_tasks.split("+")

        # nobs = self.normalizer.normalize(batch['obs'])
        # nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        To = self.n_obs_steps

        # handle different ways of passing observation
        trajectory = nactions
        # reshape B, T, ... to B*T

        if self.pred_action_steps_only:
            start = To - 1
            end = start + self.n_action_steps
            trajectory = nactions[:, start:end]

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]

        # Predict the noise residual

        mdl_out = self.mdl(sample=noisy_trajectory,
                           timestep=timesteps,
                           multimod_inputs=multimod_inputs,
                           mask=self.mask_type,
                           task=task,
                           mode=mode, )
        pred = mdl_out["pred"]
        metrics.update(mdl_out["obs_encoder_out"]["ssl_losses"])

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        diffusion_loss = F.mse_loss(pred, target, reduction='none')
        diffusion_loss = diffusion_loss * loss_mask.type(diffusion_loss.dtype)
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

        for key, value in mdl_out["obs_encoder_out"]["ssl_losses"].items():
            total_loss += value * self.weight[key]

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
