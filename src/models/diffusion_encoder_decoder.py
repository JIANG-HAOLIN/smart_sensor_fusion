import hydra.utils
from typing import Union, Optional, Tuple, Dict
from omegaconf import DictConfig, OmegaConf
import logging
import torch
import torch.nn as nn
from src.models.utils.positional_encoding import SinusoidalPosEmb
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

import logging

logger = logging.getLogger(__name__)


class TransformerForDiffusion(nn.Module):
    def __init__(self,
                 action_decoder: DictConfig,
                 obs_encoder: Optional[DictConfig] = None,
                 ) -> None:
        super().__init__()
        self._dummy_variable = nn.Parameter()
        input_dim = action_decoder.input_dim
        output_dim = action_decoder.output_dim
        t_p = action_decoder.t_p  # Tp prediction horizon
        n_obs_steps = action_decoder.n_obs_steps
        cond_dim = action_decoder.cond_dim
        n_layer = action_decoder.n_layer
        n_head = action_decoder.n_head
        n_emb = action_decoder.n_emb
        p_drop_emb = action_decoder.p_drop_emb
        p_drop_attn = action_decoder.p_drop_attn
        causal_attn = action_decoder.causal_attn
        time_as_cond = action_decoder.time_as_cond
        n_cond_layers = action_decoder.n_cond_layers

        # compute number of tokens for main trunk and condition encoder
        if action_decoder.n_obs_steps is None:
            n_obs_steps = t_p
        else:
            self.n_obs_steps = n_obs_steps

        T = t_p
        T_cond = 1
        if not time_as_cond:
            T += 1
            T_cond -= 1
        obs_as_cond = cond_dim > 0
        if obs_as_cond:
            assert time_as_cond
            T_cond += n_obs_steps

        # input embedding stem
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # cond encoder
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = None

        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)

        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        encoder_only = False
        if T_cond > 0:
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))
            if n_cond_layers > 0:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4 * n_emb,
                    dropout=p_drop_attn,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer,
                    num_layers=n_cond_layers
                )
            else:
                self.encoder = nn.Sequential(
                    nn.Linear(n_emb, 4 * n_emb),
                    nn.Mish(),
                    nn.Linear(4 * n_emb, n_emb)
                )
            # decoder
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True  # important for stability
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=n_layer
            )
        else:
            # encoder only BERT
            encoder_only = True

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_layer
            )

        # attention mask
        if causal_attn:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)

            if time_as_cond and obs_as_cond:
                S = T_cond
                t, s = torch.meshgrid(
                    torch.arange(T),
                    torch.arange(S),
                    indexing='ij'
                )
                mask = t >= (s - 1)  # add one dimension since time is the first token in cond
                mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
                self.register_buffer('memory_mask', mask)
            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        # decoder head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        # constants
        self.T = T
        self.T_cond = T_cond
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.encoder_only = encoder_only

        # init
        self.apply(self._init_weights)
        logger.info(
            "number of parameters in TransformerForDiffusion Object: %e", sum(p.numel() for p in self.parameters())
        )

        self.obs_encoder = hydra.utils.instantiate(obs_encoder, _recursive_=False)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def _init_weights(self, module):
        ignore_types = (nn.Dropout,
                        SinusoidalPosEmb,
                        nn.TransformerEncoderLayer,
                        nn.TransformerDecoderLayer,
                        nn.TransformerEncoder,
                        nn.TransformerDecoder,
                        nn.ModuleList,
                        nn.Mish,
                        nn.Sequential)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def get_optim_groups(self, weight_decay: float = 1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        no_decay.add("_dummy_variable")
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
                len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
                len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def configure_optimizers(self,
                             learning_rate: float = 1e-4,
                             weight_decay: float = 1e-3,
                             betas: Tuple[float, float] = (0.9, 0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def forward(self,
                sample: torch.Tensor,
                timestep: Union[torch.Tensor, float, int],
                multimod_inputs: Optional[Dict] = None,
                mask: Optional[str] = "latent_mask",
                task: Optional[str] = "imitation",
                mode: str = "train",
                additional_input: Optional[torch.Tensor] = None,
                **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B,T',cond_dim)
        output: (B,T,input_dim)
        """
        batch_size = sample.shape[0]
        obs_encoder_out = self.obs_encoder(multimod_inputs=multimod_inputs,
                                           mask=mask,
                                           task=task,
                                           mode=mode,
                                           additional_input=additional_input, )
        nobs_features = obs_encoder_out["repr"]['fused_encoded_inputs']
        # reshape back to B, To, Do
        cond = nobs_features
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        # (B,1,n_emb)

        # process input
        input_emb = self.input_emb(sample)

        # encoder
        cond_embeddings = time_emb

        cond_obs_emb = self.cond_obs_emb(cond)
        # (B,To,n_emb)
        cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)
        tc = cond_embeddings.shape[1]
        position_embeddings = self.cond_pos_emb[:, :tc, :]  # each position maps to a (learnable) vector
        x = self.drop(cond_embeddings + position_embeddings)
        x = self.encoder(x)
        memory = x
        # (B,T_cond,n_emb)

        # decoder
        token_embeddings = input_emb
        t = token_embeddings.shape[1]
        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        # (B,T,n_emb)
        x = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=self.mask,
            memory_mask=self.memory_mask
        )
        # (B,T,n_emb)

        # head
        x = self.ln_f(x)
        x = self.head(x)
        # (B,T,n_out)

        return {"pred": x, "obs_encoder_out": obs_encoder_out}


class DiffusionTransformerHybridImagePolicy(pl.LightningModule):

    def __init__(self,
                 action_decoder: DictConfig,
                 obs_encoder,
                 noise_scheduler: DictConfig,
                 inference_args: DictConfig,
                 ######################################
                 shape_meta: dict,
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
        self.noise_scheduler = hydra.utils.instantiate(noise_scheduler)
        num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        if inference_args.num_inference_steps is not None:
            self.num_inference_steps = num_inference_steps
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

        self.mdl = TransformerForDiffusion(action_decoder, obs_encoder)
        self.action_dim = action_dim
        self.t_p = action_decoder.t_p  # Tp prediction horizon
        self.n_obs_steps = action_decoder.n_obs_steps

    def forward(self, actions,
                multimod_inputs,
                mask,
                task,
                mode, ):

        # handle different ways of passing observation
        trajectory = actions
        # reshape B, T, ... to B*T

        condition_mask = torch.full(trajectory.shape, False)

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
                           mask=mask,
                           task=task,
                           mode=mode,
                           additional_input=None, )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        return mdl_out, target, loss_mask

    def conditional_sample(self,
                           condition_data,
                           condition_mask,
                           multimod_inputs=None,
                           generator=None,
                           **kwargs
                           ):
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
            model_output = self.mdl(sample=trajectory,
                                    timestep=t,
                                    multimod_inputs=multimod_inputs,
                                    mask=None,
                                    task="repr",
                                    mode="val", )
            pred = model_output["pred"]
            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                pred, t, trajectory,
                generator=generator,
            ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(self,
                       multimod_inputs,
                       **kwargs) -> Dict[str, torch.Tensor]:
        batch_size = multimod_inputs["vision"].shape[0]
        t_p = self.t_p
        Da = self.action_dim

        device = self.device
        dtype = self.dtype

        shape = (batch_size, t_p, Da)

        cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            multimod_inputs=multimod_inputs,
            **kwargs)

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = naction_pred

        action_reduced_horizon = action_pred[:, self.n_obs_steps - 1:]

        result = {
            'action': action_reduced_horizon,
            'action_pred': action_pred
        }
        return result


class LowdimMaskGenerator(nn.Module):
    def __init__(self,
                 action_dim, obs_dim,
                 # obs mask setup
                 max_n_obs_steps=2,
                 fix_obs_steps=True,
                 # action mask
                 action_visible=False
                 ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.max_n_obs_steps = max_n_obs_steps
        self.fix_obs_steps = fix_obs_steps
        self.action_visible = action_visible
        self._dummy_variable = nn.Parameter()

    @torch.no_grad()
    def forward(self, shape, seed=None):
        device = self.device
        B, T, D = shape
        assert D == (self.action_dim + self.obs_dim)

        # create all tensors on this device
        rng = torch.Generator(device=device)
        if seed is not None:
            rng = rng.manual_seed(seed)

        # generate dim mask
        dim_mask = torch.zeros(size=shape,
                               dtype=torch.bool, device=device)
        is_action_dim = dim_mask.clone()
        is_action_dim[..., :self.action_dim] = True
        is_obs_dim = ~is_action_dim

        # generate obs mask
        if self.fix_obs_steps:
            obs_steps = torch.full((B,),
                                   fill_value=self.max_n_obs_steps, device=device)
        else:
            obs_steps = torch.randint(
                low=1, high=self.max_n_obs_steps + 1,
                size=(B,), generator=rng, device=device)

        steps = torch.arange(0, T, device=device).reshape(1, T).expand(B, T)
        obs_mask = (steps.T < obs_steps).T.reshape(B, T, 1).expand(B, T, D)
        obs_mask = obs_mask & is_obs_dim

        # generate action mask
        if self.action_visible:
            action_steps = torch.maximum(
                obs_steps - 1,
                torch.tensor(0,
                             dtype=obs_steps.dtype,
                             device=obs_steps.device))
            action_mask = (steps.T < action_steps).T.reshape(B, T, 1).expand(B, T, D)
            action_mask = action_mask & is_action_dim

        mask = obs_mask
        if self.action_visible:
            mask = mask | action_mask

        return mask

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def dict_apply(
        x: Dict[str, torch.Tensor],
        func: Callable[[torch.Tensor], torch.Tensor]
) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result
