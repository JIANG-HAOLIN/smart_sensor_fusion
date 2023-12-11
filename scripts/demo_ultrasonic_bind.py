import os
import sys
import torch
import argparse
import numpy as np
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import copy
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils.visualizations import scatter_tsne


def inference(cfg: DictConfig, args: argparse.Namespace):
    torch.set_float32_matmul_precision('medium')
    cfg_path = HydraConfig.get().runtime['config_sources'][1]['path']
    checkpoints_folder_path = os.path.abspath(os.path.join(cfg_path, '..', 'checkpoints'))
    ckpt_path = args.ckpt_path
    for p in os.listdir(checkpoints_folder_path):
        if 'best' in p:
            ckpt_path = p
    checkpoints_path = os.path.join(checkpoints_folder_path, ckpt_path)
    if os.path.isfile(checkpoints_path):
        print("Found pretrained model, loading...")
        _, val_loader, _ = hydra.utils.instantiate(cfg.datasets.dataloader, val_batch_size=1, val_trajs=(range(1, 33)))
        model: torch.nn.Module = hydra.utils.instantiate(cfg.models.model, _recursive_=False).to('cpu')
        checkpoint_state_dict = torch.load(checkpoints_path)['state_dict']
        clone_state_dict = {key[4:]: checkpoint_state_dict[key] for key in checkpoint_state_dict.keys()}
        model.load_state_dict(clone_state_dict)
        model.eval()
        outs = []
        traj_outs = []
        traj_name = []
        i = -1
        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                traj_idx = batch['traj_idx']

                time_step = (batch['Y_corr'][:, -1:] + batch['Y_corr'][:, 0:1])/2
                # print(short_time_progress.shape)
                label = time_step
                s_1 = batch['s1_C']
                s_2 = batch['s2_C']
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
                out = model(acc_cage_x, acc_cage_y, acc_cage_z,
                            acc_ptu_x, acc_ptu_y, acc_ptu_z,
                            f_x, f_y, f_z,
                            i_s, i_z,
                            s_1, s_2)

                s, (acc, f, c) = out
                out = torch.stack([s, acc, f, c], dim=1)
                out = out.detach().cpu().numpy()

                if idx == 0:
                    i += 1
                    pre_traj_idx = traj_idx
                    traj_outs.append(out)
                else:
                    if traj_idx != pre_traj_idx:
                        i += 1
                        traj_outs = np.concatenate(traj_outs, axis=0)
                        outs.append(traj_outs)
                        traj_name.append(pre_traj_idx)
                        traj_outs = []
                        pre_traj_idx = traj_idx
                    else:
                        traj_outs.append(out)
            traj_outs = np.concatenate(traj_outs, axis=0)
            outs.append(traj_outs)
            traj_name.append(traj_idx)
            traj_outs = []
            pre_traj_idx = traj_idx
        scatter_tsne(outs, ['ultra_sonic', 'acceleration', 'force', 'current'],
                     traj_name, os.path.join(checkpoints_folder_path, ckpt_path + '_infer_test.png'))

    else:
        print(f'pretrained Model at {checkpoints_path} not found')


if __name__ == "__main__":
    project_path = os.path.abspath(os.path.join(__file__, '..', '..'))
    sys.path.append(
        project_path)  # without this line:hydra.errors.InstantiationException: Error locating target 'src.datasets.bautiro_drilling_dataset.get_loaders', set env var HYDRA_FULL_ERROR=1 to see chained exception. full_key: datasets.dataloader

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        default='../results/short_term_drilling_progress_prediction/bautiro_drilling'
                                '/short_drilling_progress_bind_vanilla/exp_debug/12-06-17:43:43/.hydra')
    parser.add_argument('--ckpt_path', type=str,
                        default='not needed anymore')
    args = parser.parse_args()

    hydra.initialize(config_path=args.config_path, version_base=None, )
    cfg = hydra.compose(config_name='config', return_hydra_config=True, )
    HydraConfig().cfg = cfg
    OmegaConf.resolve(cfg)

    inference(cfg=cfg, args=args)
