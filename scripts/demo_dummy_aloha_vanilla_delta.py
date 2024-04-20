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
import copy
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils.visualizations import scatter_tsne, scatter_pca, scatter_pca_3d, scatter_tnse_3d, scatter_tsne_selected
from src.datasets.dummy_robot_arm import get_loaders, Normalizer
from utils.quaternion import q_exp_map, q_log_map, exp_map_seq, log_map_seq, q_to_rotation_matrix, q_from_rot_mat
import time
from utils.visualizations import plot_tensors


POSE_OFFSET = np.array([0.0, 0.0, -0.05, 1.0, 0.0, 0.0, 0.0])  # wxyz


def apply_offset(pose, pose_offset):
    # Computes T_pose * T_pose_offset => pose_new
    # Assuming quaternions are in wxyz format
    t = pose[:3]
    R = q_to_rotation_matrix(pose[3:])

    to = pose_offset[:3]
    Ro = q_to_rotation_matrix(pose_offset[3:])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    To = np.eye(4)
    To[:3, :3] = Ro
    To[:3, 3] = to

    T_new = np.matmul(T, To)
    pose_new = np.zeros((7,))
    pose_new[:3] = T_new[:3, 3]
    pose_new[3:] = q_from_rot_mat(T_new[:3, :3])
    return pose_new


def set_random_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def inference(cfg: DictConfig, args: argparse.Namespace):
    # set_random_seed(42)
    torch.set_float32_matmul_precision('medium')
    cfgs = HydraConfig.get()
    cfg_path = cfgs.runtime['config_sources'][1]['path']
    checkpoints_folder_path = os.path.abspath(os.path.join(cfg_path, 'checkpoints'))
    ckpt_path = args.ckpt_path
    for p in os.listdir(checkpoints_folder_path):
        if "best" in p and p.split('.')[-1] == 'ckpt':
            ckpt_path = p
    checkpoints_path = os.path.join(checkpoints_folder_path, ckpt_path)
    if os.path.isfile(checkpoints_path):
        print("Found pretrained model, loading...")
        model: torch.nn.Module = hydra.utils.instantiate(cfg.models.model, _recursive_=False).to(args.device)
        checkpoint_state_dict = torch.load(checkpoints_path)['state_dict']
        clone_state_dict = {key[4:]: checkpoint_state_dict[key] for key in checkpoint_state_dict.keys()}
        model.load_state_dict(clone_state_dict)
        model.eval()
        pm = []
        pmr = []
        loss_list = []
        trials_outs = []
        trials_names = []
        pred_out = []
        og_a_hat_list = []
        real_a_list = []

        train_loaders, val_loaders, train_inference_loaders = get_loaders(**cfg.datasets.dataloader, debug=True)
        l = len(train_loaders)

        normalizer = Normalizer.from_json(os.path.join(cfg_path, "normalizer_config.json"))

        max_timesteps = 1000
        query_frequency = 10
        all_time_position = torch.zeros([max_timesteps, max_timesteps + query_frequency, 4]).cuda()
        all_time_orientation = torch.zeros([max_timesteps, max_timesteps + query_frequency, 4]).cuda()

        with torch.no_grad():
            for idx1, loader in enumerate([train_inference_loaders]):
                name = str(idx1) + ("val" if idx1 >= l else "train")
                trials_names.append(name)
                trial_outs = []
                inference_time = []
                for t, batch in enumerate(loader):
                    if t % query_frequency == 0:
                        print(t)

                        start_time = time.time()
                        total_loss = 0
                        metrics = {}

                        real_delta = batch["traj"]["target_real_delta"]["action"][:, :, :].float()
                        real_delta_source = batch["traj"]["source_real_delta"]["action"][:, :, :].float()
                        direct_vel = batch["traj"]["direct_vel"]["action"][:, :, :].float()
                        pose = batch["traj"]["target_glb_pos_ori"]["action"][:, :, :].float()

                        pose_gripper = batch["traj"]["gripper"]["action"][:, :, :1].float()

                        qpos = batch["traj"]["target_glb_pos_ori"]["obs"][:, -1, :].float()

                        qpos_gripper = batch["traj"]["gripper"]["obs"][:, -1, :1].float()

                        qpos = torch.cat([qpos, qpos_gripper], dim=-1).to(args.device)

                        inp_data = batch["observation"]
                        for key, value in inp_data.items():
                            inp_data[key] = value.to(args.device)
                        multimod_inputs = {
                            "vision": inp_data,
                        }

                        inference_type = "real_delta_target"
                        if inference_type == "real_delta_target":
                            actions = real_delta
                            action_type = "target_real_delta"

                        elif inference_type == "position":
                            actions = pose[:, :, :]
                            action_type = "source_glb_pos_ori"

                        elif inference_type == "real_delta_source":
                            actions = real_delta_source[:, :, :]
                            action_type = "source_real_delta"

                        elif inference_type == "direct_vel":
                            actions = direct_vel
                            action_type = "direct_vel"

                        actions = torch.cat([actions, pose_gripper], dim=-1).to(args.device)

                        is_pad = torch.zeros([actions.shape[0], actions.shape[1]], device=qpos.device).bool()

                        all_action, raw_action, all_time_position, all_time_orientation, og_a_hat = model.rollout(
                            qpos,
                            multimod_inputs,
                            env_state=None,
                            actions=None,
                            is_pad=None,
                            all_time_position=all_time_position,
                            all_time_orientation=all_time_orientation,
                            t=t,
                            args=args,
                            v_scale=1,
                            inference_type=inference_type,
                            num_queries=query_frequency,
                            normalizer=normalizer,
                            action_type=action_type,
                            )
                        # all_action = torch.from_numpy(all_action)
                        # all_l1 = F.l1_loss(actions, all_action.to(actions.device), reduction='none')
                        # l1 = (all_l1 * ~is_pad.unsqueeze(-1).to(all_l1.device)).mean()
                        # print(l1)
                        #

                        og_a_hat_list.append(og_a_hat[0, :query_frequency, :])
                        real_a_list.append(actions[0, :query_frequency, :])

                        gripper = all_action[:, :1]
                        all_action = all_action[:, 1:]
                        if inference_type == "position":
                            for l in range(all_action.shape[0]):
                                all_action[l] = apply_offset(all_action[l], POSE_OFFSET)

                        pose = normalizer.denormalize(pose[0, :query_frequency, :], "target_glb_pos_ori")
                        pose_gripper = normalizer.denormalize(pose_gripper[0, :query_frequency, :], "gripper")
                        pm.append(torch.cat([pose, pose_gripper], dim=-1))
                        all_action = log_map_seq(all_action, np.array([0, 0, 0, 0, 1, 0, 0]))


                        pmr.append(torch.from_numpy(np.concatenate([all_action[:query_frequency, :], gripper[:query_frequency, :]], axis=-1)))

                        inference_time.append(time.time() - start_time)
                print(
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

        # print(torch.cat(loss_list).mean())
        pm = torch.cat(pm, dim=0)

        plot_tensors([pm, torch.cat(pmr, dim=0)], ["real", "pred"])

        og_a_hat = torch.cat(og_a_hat_list, dim=0)
        real_a = torch.cat(real_a_list, dim=0)
        plot_tensors([og_a_hat, real_a], ["pred", "real"])



        np.save("example_traj.npy", pm)

        x = pm[:, 0]
        y = pm[:, 1]
        z = pm[:, 2]

        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        markerline, stemlines, baseline = ax.stem(
            x, y, z, linefmt='none', markerfmt='.', orientation='z', )
        # markerline.set_markerfacecolor('none')
        ax.set_aspect('equal')

        plt.show()

    else:
        print(f'pretrained Model at {checkpoints_path} not found')


if __name__ == "__main__":
    project_path = os.path.abspath(os.path.join(__file__, '..', '..'))
    sys.path.append(
        project_path)  # without this line:hydra.errors.InstantiationException: Error locating target 'src.datasets.bautiro_drilling_dataset.get_loaders', set env var HYDRA_FULL_ERROR=1 to see chained exception. full_key: datasets.dataloader

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        default="../checkpoints/cupboard/name=alohaname=vae_vanillaaction=real_delta_targetname=coswarmuplr=5e-05weight_decay=0.0001kl_divergence=10hidden_dim=512output_layer_index=-1source=True_04-16-08:18:34")
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
