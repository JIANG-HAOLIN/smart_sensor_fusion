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
import time
from utils.quaternion import q_exp_map


def inference(cfg: DictConfig, args: argparse.Namespace):
    # import random
    # import numpy as np
    # seed = 42
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

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
        pm = []
        pmr = []
        train_loaders, val_loaders, _ = get_debug_loaders(**cfg.datasets.dataloader)
        l = len(train_loaders)
        with torch.no_grad():
            for idx1, loader in enumerate([train_loaders]):
                name = str(idx1) + ("val" if idx1 >= l else "train")
                trials_names.append(name)
                trial_outs = []
                inference_time = []
                for idx2, batch in enumerate(loader):
                    total_loss = 0
                    metrics = {}

                    inp_data = batch["observation"]
                    pose = batch["target_pose_seq"]
                    qpos = pose[:, 0, :]
                    delta = batch["target_delta_seq"][:, 1].float()
                    vf_inp, vg_inp, _, _ = inp_data
                    multimod_inputs = {
                        "vision": [vf_inp.to(args.device), vg_inp.to(args.device)],
                    }

                    # Perform prediction and calculate loss and accuracy
                    t = time.time()
                    output = model.forward(multimod_inputs,
                                           mask=cfg.pl_modules.pl_module.masked_train,
                                           task="repr",
                                           mode="inference",
                                           )
                    out_delta = output["predict"]["xyzrpy"]
                    base = qpos.squeeze(0).detach().cpu().numpy()
                    base_position = base[:3]
                    base_orientation = base[3:]
                    v = out_delta.permute(1, 0).detach().cpu().numpy()
                    v_position = v[:3]
                    v_orientation = v[3:]

                    pm.append(delta)
                    pmr.append(out_delta)

                    raw_orientation = q_exp_map(v_orientation, base_orientation)
                    raw_position = np.expand_dims(base_position, axis=-1) + v_position
                    raw_output = np.squeeze(np.concatenate([raw_position, raw_orientation], axis=0), axis=-1)
                    compute_time = round(time.time() - t, 2)

                    out = torch.cat([
                        output["repr"]["encoded_inputs"]["vision"],
                        output["repr"]['fused_encoded_inputs'],
                        output["repr"]['cross_time_repr'][:, 1:, :],
                    ], dim=0).permute(1, 0, 2)
                    out = out.detach().cpu().numpy()
                    trial_outs.append(out)
                    inference_time.append(output["time"])
                trials_outs.append(np.concatenate(trial_outs, axis=0))
                print(f"trial {name}: total infernce time {compute_time},\n"
                      f"average inference time for each step: {sum(inference_time) / len(inference_time)}"
                      f"example inference time:{inference_time[:10]}")

        pm = torch.cat(pm, dim=0).detach().cpu().numpy()
        pmr = torch.cat(pmr, dim=0).detach().cpu().numpy()
        t = np.arange(len(pm))
        tr = t.copy()
        plt.figure()
        plt.subplot(611)
        plt.plot(t, pm[:, :1], '.-', )
        plt.plot(tr, pmr[:, :1], '.-')
        # plt.plot(tfwd, pmfwd[:, :3], 'd-')
        plt.subplot(612)
        plt.plot(t, pm[:, 1:2], '.-')
        plt.plot(tr, pmr[:, 1:2], '.-')
        # plt.plot(tfwd, pmfwd[:, 3:], 'd-')
        plt.subplot(613)
        plt.plot(t, pm[:, 2:3], '.-')
        plt.plot(tr, pmr[:, 2:3], '.-')
        plt.subplot(614)
        plt.plot(t, pm[:, 3:4], '.-')
        plt.plot(tr, pmr[:, 3:4], '.-')
        plt.subplot(615)
        plt.plot(t, pm[:, 4:5], '.-')
        plt.plot(tr, pmr[:, 4:5], '.-')
        plt.subplot(616)
        plt.plot(t, pm[:, 5:6], '.-')
        plt.plot(tr, pmr[:, 5:6], '.-')
        plt.show()

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
                        default='../results/name=nomask_iminame=ssnce_earlysum_vatt_additionallatent=0.5imitation=1.0_03-10-19:58:20')
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
