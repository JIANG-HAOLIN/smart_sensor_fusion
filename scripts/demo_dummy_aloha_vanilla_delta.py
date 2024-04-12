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
from src.datasets.dummy_robot_arm import get_debug_loaders, get_loaders
from utils.quaternion import q_exp_map, q_log_map, exp_map_seq, log_map_seq
import time
from utils.visualizations import plot_tensors


def set_random_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

denormalize = lambda x, arr: (x + 1)/2*(arr["max"] - arr["min"]) + arr["min"]

norm_state = {'resample': {'target_real_delta': {'max': np.array([0.01833077, 0.01471558, 0.01419077, 0.0090178 , 0.00767898,
       0.01499568]), 'min': np.array([-0.01137316, -0.01453483, -0.01489501, -0.00618764, -0.00964906,
       -0.01807459]), 'mean': np.array([ 3.13497064e-04,  4.33342494e-04, -1.08199064e-03,  4.00690490e-05,
       -2.47188444e-04,  2.91489864e-04]), 'std': np.array([0.00257714, 0.00284448, 0.0027104 , 0.0010101 , 0.00154394,
       0.00194711])}, 'target_glb_pos_ori': {'max': np.array([0.60799567, 0.24106479, 0.37450424, 0.10115965, 0.04896518,
       0.17713495]), 'min': np.array([ 0.29311409, -0.06481589,  0.08938819, -0.02968255, -0.10085378,
       -0.15162727]), 'mean': np.array([ 0.45605428,  0.08565441,  0.174311  ,  0.02461957, -0.03768246,
        0.01832523]), 'std': np.array([0.068358  , 0.07183608, 0.0707843 , 0.02024454, 0.02290095,
       0.04824965])}, 'source_real_delta': {'max': np.array([ 0.05323321,  0.04156064, -0.01151191,  0.04448153,  0.05435663,
        0.03082125]), 'min': np.array([-0.05535634, -0.04399007, -0.08517262, -0.03585722, -0.04552806,
       -0.03912532]), 'mean': np.array([-0.00534913,  0.00288355, -0.05514399,  0.0020868 ,  0.01027822,
        0.00162523]), 'std': np.array([0.01629853, 0.0115366 , 0.01200146, 0.01321739, 0.01472549,
       0.00834285])}, 'source_glb_pos_ori': {'max': np.array([0.60589246, 0.24761381, 0.32945419, 0.11862382, 0.06972952,
       0.1831929 ]), 'min': np.array([ 0.2776695 , -0.06622958,  0.03222788, -0.02801248, -0.09814695,
       -0.15366502]), 'mean': np.array([ 0.4504151 ,  0.08810488,  0.12027499,  0.02617468, -0.0271667 ,
        0.02004663]), 'std': np.array([0.06902871, 0.07362574, 0.07155867, 0.02378412, 0.0213967 ,
       0.04946909])}, 'gripper': {'max': np.array([0.03066929]), 'min': np.array([0.00145977, ]), 'mean': np.array([0.01886992, ]), 'std': np.array([0.01372382, ])}, 'direct_vel': {'max': np.array([ 0.04288793,  0.03173504, -0.01813139,  0.04065353,  0.04867868,
        0.0184301 ]), 'min': np.array([-0.04977977, -0.03515093, -0.07595439, -0.03135369, -0.04040732,
       -0.02396019]), 'mean': np.array([-0.00563918,  0.00245047, -0.05403601,  0.00200457,  0.01039245,
        0.00130618]), 'std': np.array([0.01390834, 0.0089605 , 0.00962011, 0.01280517, 0.0141328 ,
       0.00703205])}}, 'smooth': {'target_real_delta': {'max': np.array([0.01769036, 0.01410254, 0.01244026, 0.00721367, 0.00660652,
       0.01507445]), 'min': np.array([-0.01034324, -0.01433304, -0.01295911, -0.00580396, -0.00849982,
       -0.01739865]), 'mean': np.array([ 3.13550801e-04,  4.33161468e-04, -1.08197222e-03,  4.08131623e-05,
       -2.48204061e-04,  2.91412280e-04]), 'std': np.array([0.00253795, 0.00280512, 0.00266423, 0.00096524, 0.00149994,
       0.00189667])}, 'target_glb_pos_ori': {'max': np.array([0.60782946, 0.24107972, 0.37453574, 0.10093857, 0.04868671,
       0.17745757]), 'min': np.array([ 0.29317631, -0.06488244,  0.08923991, -0.02988128, -0.10058427,
       -0.15150303]), 'mean': np.array([ 0.45605428,  0.08565441,  0.174311  ,  0.02461957, -0.03768246,
        0.01832523]), 'std': np.array([0.06835725, 0.07183541, 0.07078369, 0.02024171, 0.02289883,
       0.04824871])}, 'source_real_delta': {'max': np.array([ 0.0533458 ,  0.04051842, -0.01175208,  0.04351691,  0.05407896,
        0.03038214]), 'min': np.array([-0.05561119, -0.04391569, -0.08456834, -0.0348076 , -0.04496127,
       -0.03898464]), 'mean': np.array([-0.0053492 ,  0.00288325, -0.05514397,  0.00208666,  0.01027814,
        0.00162536]), 'std': np.array([0.01628969, 0.01152728, 0.01199221, 0.0132063 , 0.01471578,
       0.00832774])}, 'source_glb_pos_ori': {'max': np.array([0.60580062, 0.24757468, 0.32923787, 0.11831453, 0.06992805,
       0.1833786 ]), 'min': np.array([ 0.27800035, -0.06643748,  0.03202078, -0.02810872, -0.09806573,
       -0.15393454]), 'mean': np.array([ 0.4504151 ,  0.08810488,  0.12027499,  0.02617468, -0.0271667 ,
        0.02004663]), 'std': np.array([0.06902801, 0.0736251 , 0.07155798, 0.0237819 , 0.02139435,
       0.04946812])}, 'gripper': {'max': np.array([0.03066929]), 'min': np.array([0.00145977]), 'mean': np.array([0.01886992]), 'std': np.array([0.01372382])}, 'direct_vel': {'max': np.array([ 0.04246886,  0.03144862, -0.01852505,  0.04025868,  0.04797959,
        0.01792963]), 'min': np.array([-0.04955756, -0.03480149, -0.07592206, -0.03116205, -0.03996206,
       -0.02402754]), 'mean': np.array([-0.00563918,  0.00245047, -0.05403601,  0.00200456,  0.01039244,
        0.00130622]), 'std': np.array([0.01390194, 0.00895157, 0.00961124, 0.01279775, 0.01412746,
       0.00702197])}}}

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
        pred_out = []
        og_a_hat_list = []
        real_a_list = []
        # a, b, c = get_loaders(**cfg.datasets.dataloader)
        # norm_state = a.dataset.datasets[0].norm_state
        train_loaders, val_loaders, train_inference_loaders = get_debug_loaders(**cfg.datasets.dataloader)
        l = len(train_loaders)

        max_timesteps = 1000
        num_queries = 10
        query_frequency = 10
        all_time_position = torch.zeros([max_timesteps, max_timesteps + num_queries, 4]).cuda()
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
                        elif inference_type == "position":
                            actions = pose[:, :, :]
                        elif inference_type == "real_delta_source":
                            actions = real_delta_source[:, :, :]
                        elif inference_type == "direct_vel":
                            actions = direct_vel
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
                            num_queries=num_queries,
                            action_norm_state=norm_state["resample"]["target_real_delta"],
                            qpos_norm_state=norm_state["resample"]["target_glb_pos_ori"],
                            gripper_norm_state=norm_state["resample"]["gripper"]
                            )
                        # all_action = torch.from_numpy(all_action)
                        # all_l1 = F.l1_loss(actions, all_action.to(actions.device), reduction='none')
                        # l1 = (all_l1 * ~is_pad.unsqueeze(-1).to(all_l1.device)).mean()
                        # print(l1)
                        #

                        og_a_hat_list.append(og_a_hat[0, :query_frequency, :])
                        real_a_list.append(actions[0, :query_frequency, :])

                        pos_stat = norm_state["resample"]["target_glb_pos_ori"]



                        gripper = all_action[:, :1]
                        all_action = all_action[:, 1:]
                        pose = denormalize(pose[0, :query_frequency, :], pos_stat)
                        pose_gripper = denormalize(pose_gripper[0, :query_frequency, :], norm_state["resample"]["gripper"])
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
                        default="../checkpoints/cupboard/name=alohaname=vae_vanillaaction=real_delta_targetname=coswarmuplr=0.0001weight_decay=0.0001kl_divergence=10hidden_dim=512output_layer_index=-1source=True_04-12-09:33:48")
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
