import os
import sys
import torch
import argparse
import numpy as np
import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import copy
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils.visualizations import scatter_tsne, scatter_pca, scatter_pca_3d, scatter_tnse_3d, scatter_tsne_selected
from src.datasets.dummy_robot_arm import get_debug_loaders
from utils.quaternion import q_exp_map, q_log_map
import time


def inference(cfg: DictConfig, args: argparse.Namespace):
    torch.set_float32_matmul_precision('medium')
    cfgs = HydraConfig.get()
    cfg_path = cfgs.runtime['config_sources'][1]['path']
    checkpoints_folder_path = os.path.abspath(os.path.join(cfg_path, 'checkpoints'))
    ckpt_path = args.ckpt_path
    for p in os.listdir(checkpoints_folder_path):
        if 'best' in p and p.split('.')[-1] == 'ckpt':
            ckpt_path = p
    checkpoints_path = os.path.join(checkpoints_folder_path, ckpt_path)
    if os.path.isfile(checkpoints_path):
        print("Found pretrained model, loading...")
        model: torch.nn.Module = hydra.utils.instantiate(cfg.models.model, _recursive_=False).to(args.device)
        checkpoint_state_dict = torch.load(checkpoints_path)['state_dict']
        clone_state_dict = {key[4:]: checkpoint_state_dict[key] for key in checkpoint_state_dict.keys()}
        model.load_state_dict(clone_state_dict)
        model.eval()
        trials_outs = []
        trials_names = []
        train_loaders, val_loaders, _ = get_debug_loaders(**cfg.datasets.dataloader)
        l = len(train_loaders)

        max_timesteps = 1000
        num_queries = 30
        all_time_position = torch.zeros([max_timesteps, max_timesteps + num_queries, 3]).cuda()
        all_time_orientation = torch.zeros([max_timesteps, max_timesteps + num_queries, 4]).cuda()
        with torch.no_grad():
            for idx1, loader in enumerate([val_loaders]):
                name = str(idx1) + ("val" if idx1 >= l else "train")
                trials_names.append(name)
                trial_outs = []
                inference_time = []
                for t, batch in enumerate(loader):
                    total_loss = 0
                    metrics = {}

                    inp_data = batch["observation"]
                    delta = batch["target_delta_seq"][:, 1:]
                    pose = batch["target_pose_seq"]
                    vf_inp, vg_inp, _, _ = inp_data
                    multimod_inputs = {
                        "vision": [vf_inp.to(args.device), vg_inp.to(args.device)],
                    }
                    qpos = pose[:, 0, :].to(args.device)

                    # Perform prediction and calculate loss and accuracy
                    in_t = time.time()
                    output = model(qpos,
                                        multimod_inputs,
                                        actions=None,
                                        is_pad=None,
                                        mask=None,
                                        mask_type=None,
                                        task="repr",
                                        mode="val")
                    a_hat, is_pad_hat, (mu, logvar) = output["vae_output"]
                    out_delta = a_hat
                    base = qpos.squeeze(0).detach().cpu().numpy()
                    base_position = base[:3]
                    base_orientation = base[3:]
                    v = out_delta.squeeze(0).permute(1, 0).detach().cpu().numpy()
                    v_position = v[:3]
                    v_orientation = v[3:]




                    out_delta = torch.tensor(q_exp_map(v_orientation, base_orientation)).permute(1, 0)
                    all_time_orientation[[t], t:t + num_queries] = out_delta.float().to(args.device)
                    orientation_for_curr_step = all_time_orientation[:, t]
                    actions_populated = torch.all(orientation_for_curr_step != 0, axis=1)
                    orientation_for_curr_step = orientation_for_curr_step[actions_populated]

                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(orientation_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()

                    weights = np.expand_dims(exp_weights, axis=0)
                    raw_orientation = orientation_for_curr_step[0].detach().cpu().numpy()
                    orientation = orientation_for_curr_step.permute(1, 0).detach().cpu().numpy()
                    for i in range(5):
                        tangent_space_vector = q_log_map(orientation, raw_orientation)
                        tangent_space_vector = np.sum(tangent_space_vector * weights, axis=1, keepdims=True)
                        if sum(np.abs(tangent_space_vector)) < 1e-6:
                            break
                        raw_orientation = q_exp_map(tangent_space_vector, raw_orientation)[:, 0]

                    out_position = np.expand_dims(base_position, axis=-1) + v_position
                    out_position = torch.from_numpy(out_position).permute(1, 0)
                    all_time_position[[t], t:t + num_queries] = out_position.float().to(args.device)
                    position_for_curr_step = all_time_position[:, t]
                    actions_populated = torch.all(position_for_curr_step != 0, axis=1)
                    position_for_curr_step = position_for_curr_step[actions_populated]
                    weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (position_for_curr_step * weights).sum(dim=0, keepdim=True)
                    raw_position = raw_action.squeeze(0).cpu().numpy()

                    raw_output = np.concatenate([raw_position, raw_orientation], axis=0)

                    compute_time = round(time.time() - in_t, 4)

                    # out = torch.cat([
                    #     output["obs_encoder_out"]["repr"]["encoded_inputs"]["vision"],
                    #     output["obs_encoder_out"]["repr"]['fused_encoded_inputs'],
                    #     output["obs_encoder_out"]["repr"]['cross_time_repr'][:, 1:, :],
                    # ], dim=0).permute(1, 0, 2)
                    # out = out.detach().cpu().numpy()
                    # trial_outs.append(out)
                    inference_time.append(compute_time)
                # trials_outs.append(np.concatenate(trial_outs, axis=0))
                print(f"trial {name}: total infernce time {compute_time},\n"
                      f"average inference time for each step: {sum(inference_time) / len(inference_time)}"
                      f"example inference time:{inference_time[:10]}")

        # output_png_path = os.path.join(checkpoints_folder_path, ckpt_path + '_infer')
        # scatter_tsne([i[:, :3] for i in trials_outs], ["vision", "audio", "tactile", ],
        #              trials_names, output_png_path+"_modalities")
        # scatter_tsne([i[:, 3:] for i in trials_outs], ["fused_inputs", "cross_time_output"],
        #              trials_names, output_png_path+"_fused_and_ct")
        # scatter_tsne([i[:, 3:4] for i in trials_outs], ["fused_inputs", ],
        #              trials_names, output_png_path+"_fused_inputs")
        # scatter_tsne([i[:, -1:] for i in trials_outs], ["cross_time_output", ],
        #              trials_names, output_png_path+"_ct_output")
    else:
        print(f'pretrained Model at {checkpoints_path} not found')


if __name__ == "__main__":
    project_path = os.path.abspath(os.path.join(__file__, '..', '..'))
    sys.path.append(
        project_path)  # without this line:hydra.errors.InstantiationException: Error locating target 'src.datasets.bautiro_drilling_dataset.get_loaders', set env var HYDRA_FULL_ERROR=1 to see chained exception. full_key: datasets.dataloader

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        default='name=alohaname=vaelatent=0.5_03-07-21:01:18')
    parser.add_argument('--ckpt_path', type=str,
                        default='not needed anymore')
    parser.add_argument('--device', type=str,
                        default='cuda')

    args = parser.parse_args()

    if 'checkpoints' in os.listdir(args.config_path):
        hydra.initialize(config_path=args.config_path, version_base=None, )
        cfg = hydra.compose(config_name='config', return_hydra_config=True, )
        HydraConfig().cfg = cfg
        OmegaConf.resolve(cfg)

        inference(cfg=cfg, args=args)
    else:
        # inference on multiple checkpoints
        for sub_dir in os.listdir(args.config_path):
            sub_dir_path = os.path.join(args.config_path, sub_dir)
            if os.path.isdir(sub_dir_path) and sub_dir != '.hydra':
                print(f'current hydra config: {sub_dir}')
                hydra.initialize(config_path=sub_dir_path, version_base=None, )
                cfg = hydra.compose(config_name='config', return_hydra_config=True, )
                HydraConfig().cfg = cfg
                OmegaConf.resolve(cfg)

                inference(cfg=cfg, args=args)
                GlobalHydra.instance().clear()
