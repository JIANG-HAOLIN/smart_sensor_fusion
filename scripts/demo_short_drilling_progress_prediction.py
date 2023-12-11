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


def inference(cfg: DictConfig, args: argparse.Namespace):
    num_val_trajs = 32
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
        _, val_loader, _ = hydra.utils.instantiate(cfg.datasets.dataloader,
                                                   val_batch_size=1,
                                                   val_trajs=(range(1, num_val_trajs + 1)))
        fig, ax = plt.subplots(num_val_trajs, 1, figsize=(20*1, 10*num_val_trajs))
        model: torch.nn.Module = hydra.utils.instantiate(cfg.models.model, _recursive_=False).to('cpu')
        checkpoint_state_dict = torch.load(checkpoints_path)['state_dict']
        clone_state_dict = {key[4:]: checkpoint_state_dict[key] for key in checkpoint_state_dict.keys()}
        model.load_state_dict(clone_state_dict)
        model.eval()
        preds = []
        labels = []
        losses = []
        i = -1
        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                traj_idx = batch['traj_idx']
                if idx == 0:
                    pre_traj_idx = traj_idx
                short_time_progress = batch['Y_corr'][:, -1:] - batch['Y_corr'][:, 0:1]
                # print(short_time_progress.shape)
                label = short_time_progress
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
                            i_s, i_z)
                pred = out[0].detach()
                label_copy = copy.deepcopy(label)
                losses.append(F.mse_loss(pred.view(-1, pred.size(-1)), label).float())
                preds.append(pred.flatten().numpy())
                labels.append(label_copy.flatten().numpy())
                if traj_idx != pre_traj_idx:
                    i += 1
                    x = np.arange(len(labels)) * (cfg.datasets.dataloader.step_size / 50000)
                    preds = np.concatenate(preds)
                    labels = np.concatenate(labels)
                    ax[i].plot(x, labels, '.-', label='groud_truth')
                    ax[i].plot(x, preds, '.-', label='predictions')
                    ax[i].set_title(f'trajectory: {str(pre_traj_idx.item())}')
                    ax[i].legend()
                    preds = []
                    labels = []
                    pre_traj_idx = traj_idx
            val_loss = sum(losses)/len(losses)
            print(f'validation loss: {val_loss}')
            i += 1
            x = np.arange(len(labels)) * (cfg.datasets.dataloader.step_size / 50000)
            preds = np.concatenate(preds)
            labels = np.concatenate(labels)
            ax[i].plot(x, labels, '.-', label='groud_truth')
            ax[i].plot(x, preds, '.-', label='predictions')
            ax[i].set_title(f'trajectory: {str(pre_traj_idx.item())}')
            # output_folder = os.path.abspath(os.path.join(__file__, '..', __file__.split('/')[-1]))
            # if not os.path.exists(output_folder):
            #     os.makedirs(output_folder)
            # we output the fig to the original results folder
            ax[i].legend()
        fig.savefig(os.path.join(checkpoints_folder_path, ckpt_path + '_infer.png'),
                    bbox_inches='tight',)
        plt.show()
    else:
        print(f'pretrained Model at {checkpoints_path} not found')


if __name__ == "__main__":
    project_path = os.path.abspath(os.path.join(__file__, '..', '..'))
    sys.path.append(project_path)  # without this line:hydra.errors.InstantiationException: Error locating target 'src.datasets.bautiro_drilling_dataset.get_loaders', set env var HYDRA_FULL_ERROR=1 to see chained exception. full_key: datasets.dataloader

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        default='../results/short_term_drilling_progress_prediction/bautiro_drilling'
                                '/earlycat_short_drilling_progress_prediction_vanilla/exp_NoYcorrNorm/12-10-15:52:25'
                                '/.hydra')
    parser.add_argument('--ckpt_path', type=str,
                        default='11-30-16:21:51-jobid=0-epoch=8-step=216.ckpt')
    args = parser.parse_args()

    hydra.initialize(config_path=args.config_path, version_base=None, )
    cfg = hydra.compose(config_name='config', return_hydra_config=True, )
    HydraConfig().cfg = cfg
    OmegaConf.resolve(cfg)

    inference(cfg=cfg, args=args)
