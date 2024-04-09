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
from utils.quaternion import q_exp_map, q_log_map, exp_map_seq, log_map_seq
import time


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
        pm = []
        pmr = []
        loss_list = []
        trials_outs = []
        trials_names = []
        train_loaders, val_loaders, train_inference_loaders = get_debug_loaders(**cfg.datasets.dataloader)
        l = len(train_loaders)

        max_timesteps = 1000
        num_queries = 10
        query_frequency = 5
        all_time_position = torch.zeros([max_timesteps, max_timesteps + num_queries, 3]).cuda()
        all_time_orientation = torch.zeros([max_timesteps, max_timesteps + num_queries, 4]).cuda()
        with torch.no_grad():
            for idx1, loader in enumerate([val_loaders]):
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

                        real_delta = batch["smooth_future_real_delta"]
                        real_delta_direct = batch["smooth_future_real_delta_direct"]
                        relative_delta = batch["smooth_future_relative_delta"]
                        pose = batch["smooth_future_glb_pos_ori"]
                        qpos = batch["smooth_previous_glb_pos_ori"][:, -1, :].to(args.device)
                        # qpos = pose[:, 0, :].to(args.device)

                        inp_data = batch["observation"]

                        for key, value in inp_data.items():
                            inp_data[key] = value.to(args.device)
                        multimod_inputs = {
                            "vision": inp_data,
                        }

                        inference_type = "relative_delta"
                        if inference_type == "real_delta":
                            actions = real_delta[:, 1:, :]
                        elif inference_type == "position":
                            actions = pose[:, 1:, :]
                        elif inference_type == "relative_delta":
                            actions = relative_delta[:, 1:, :]
                        elif inference_type == "real_delta_direct":
                            actions = real_delta_direct[:, 1:, :]

                        is_pad = torch.zeros([actions.shape[0], actions.shape[1]], device=qpos.device).bool()

                        all_action, raw_action, all_time_position, all_time_orientation = model.rollout(
                            qpos,
                            multimod_inputs,
                            env_state=None,
                            actions=None,
                            is_pad=None,
                            all_time_position=all_time_position,
                            all_time_orientation=all_time_orientation,
                            t=t,
                            args=args,
                            v_scale=0.10 / 10,
                            inference_type=inference_type,
                            num_queries=num_queries,
                            )
                        # all_action = torch.from_numpy(all_action)
                        # all_l1 = F.l1_loss(actions, all_action.to(actions.device), reduction='none')
                        # l1 = (all_l1 * ~is_pad.unsqueeze(-1).to(all_l1.device)).mean()
                        # print(l1)
                        #
                        # o = all_action[..., 3:]
                        # print(torch.sum(o ** 2, dim=-1) ** 0.5)
                        # o_real = actions[..., 3:]
                        # print(torch.sum(o_real ** 2, dim=-1) ** 0.5)

                        # pm.append(delta[0, 1:query_frequency + 1, :])
                        # pmr.append(all_action[:query_frequency, :])
                        pose = pose[0, 1:query_frequency + 1, :].detach().cpu().numpy()
                        # pose = exp_map_seq(pose, np.array([0, 0, 0, 0, 1, 0, 0]))
                        pm.append(pose)
                        all_action = log_map_seq(all_action, np.array([0, 0, 0, 0, 1, 0, 0]))
                        pmr.append(all_action[:query_frequency, :])

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
        pm = np.concatenate(pm, axis=0)
        if isinstance(pmr, torch.Tensor):
            pmr = torch.cat(pmr, dim=0).detach().cpu().numpy()
        else:
            pmr = np.concatenate(pmr, axis=0)

        t = np.arange(len(pm))
        tr = t.copy()
        plt.figure()
        plt.subplot(711)
        plt.plot(t, pm[:, :1], '.-', )
        plt.plot(tr, pmr[:, :1], '-')
        # plt.plot(tfwd, pmfwd[:, :3], 'd-')
        plt.subplot(712)
        plt.plot(t, pm[:, 1:2], '.-')
        plt.plot(tr, pmr[:, 1:2], '-')
        # plt.plot(tfwd, pmfwd[:, 3:], 'd-')
        plt.subplot(713)
        plt.plot(t, pm[:, 2:3], '.-')
        plt.plot(tr, pmr[:, 2:3], '-')
        plt.subplot(714)
        plt.plot(t, pm[:, 3:4], '.-')
        plt.plot(tr, pmr[:, 3:4], '-')
        plt.subplot(715)
        plt.plot(t, pm[:, 4:5], '.-')
        plt.plot(tr, pmr[:, 4:5], '-')
        plt.subplot(716)
        plt.plot(t, pm[:, 5:6], '.-')
        plt.plot(tr, pmr[:, 5:6], '-')
        # plt.subplot(717)
        # plt.plot(t, pm[:, 6:7], '.-')
        # plt.plot(tr, pmr[:, 6:7], '-')
        plt.show()

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
                        default="../checkpoints/name=alohaname=vae_vanillaaction=relative_deltaname=coswarmuplr=1e-05weight_decay=0.0001kl_divergence=10hidden_dim=256output_layer_index=0source=True_03-31-06:05:53")
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
