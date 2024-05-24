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
        clone_state_dict = {key[8:]: checkpoint_state_dict[key] for key in checkpoint_state_dict.keys() if "ema_mdl.mdl" in key}  # name key changed from mdl to ema_mdl
        clone_state_dict["_dummy_variable"] = torch.Tensor(0)
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

        train_loaders, val_loaders, train_inference_loaders = get_loaders(**cfg.datasets.dataloader, debug=True, load_json=os.path.join(cfg_path, "normalizer_config.json"))
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
                        pose = batch["traj"]["source_glb_pos_ori"]["action"][:, :, :].float()

                        pose_gripper = batch["traj"]["gripper"]["action"][:, :, :1].float()

                        qpos = batch["traj"]["target_glb_pos_ori"]["obs"][:, -1, :].float()
                        qpos_gripper = batch["traj"]["gripper"]["obs"][:, -1, :1].float()
                        qpos = torch.cat([qpos, qpos_gripper], dim=-1)

                        multimod_inputs = {
                            "vision": batch["observation"]["v_fix"].to(args.device),
                            "qpos": torch.cat([batch["traj"]["target_glb_pos_ori"]["obs"].float(),
                                               batch["traj"]["gripper"]["obs"][..., :1].float()], dim=-1).to(args.device)
                        }

                        inference_type = cfg.pl_modules.pl_module.action
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

                        result = model.predict_action(multimod_inputs)
                        pred_action = result['action_pred'].detach().cpu().numpy()
                        actions = actions.detach().cpu().numpy()

                        og_a_hat_list.append(pred_action[0, :query_frequency, :])
                        real_a_list.append(actions[0, :query_frequency, :])

                        gripper = pred_action[0, :, -1:]
                        pred_action = pred_action[0, :, :-1]

                        pose = normalizer.denormalize(pose[0, :query_frequency, :], "target_glb_pos_ori")
                        pose_gripper = normalizer.denormalize(pose_gripper[0, :query_frequency, :], "gripper")


                        inference_time.append(time.time() - start_time)
                print(
                    f"average inference time for each step: {sum(inference_time) / len(inference_time)}"
                    f"example inference time:{inference_time[:10]}")


        og_a_hat = np.concatenate(og_a_hat_list, axis=0)
        real_a = np.concatenate(real_a_list, axis=0)
        plot_tensors([og_a_hat, real_a], ["pred", "real"])


        plt.show()

    else:
        print(f'pretrained Model at {checkpoints_path} not found')


if __name__ == "__main__":
    project_path = os.path.abspath(os.path.join(__file__, '..', '..'))
    sys.path.append(
        project_path)  # without this line:hydra.errors.InstantiationException: Error locating target 'src.datasets.bautiro_drilling_dataset.get_loaders', set env var HYDRA_FULL_ERROR=1 to see chained exception. full_key: datasets.dataloader

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        default="../checkpoints/cuponboard_diff/name=diffusion_policyname=DiffusionPolicyaction=positionname=coswarmuplr=5e-05weight_decay=1e-05frameskip=5_target_=diffusers.schedulers.scheduling_ddpm.DDPMScheduler_04-28-18:07:45")
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
