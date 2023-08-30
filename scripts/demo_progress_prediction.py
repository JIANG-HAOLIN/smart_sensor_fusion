import os
import sys
import torch
import argparse
import numpy as np
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from src.datasets.progress_prediction import ImitationEpisode
from torch.utils.data import DataLoader

project_path = os.path.abspath(os.path.join(__file__, '..', '..'))
sys.path.append(project_path)  # so we can import the modules inside the project when running in terminal
print(project_path)

def get_val_loader(val_csv: str, args, data_folder: str, **kwargs):
    val_csv = os.path.join(project_path, val_csv)
    data_folder = os.path.join(project_path, data_folder)
    val_set = ImitationEpisode(val_csv, args, 0, data_folder, False)
    return DataLoader(val_set, 1, num_workers=8, shuffle=False)


def inference(cfg: DictConfig, args: argparse.Namespace) -> None:
    """

    Args:
        cfg: hydra config file
        args: the input arguments

    Output the prediction sequence and visualization of the attention map

    """
    from utils.plot_attn import plot_attention_maps
    torch.set_float32_matmul_precision('medium')


    val_loader = get_val_loader(**cfg.datasets.dataloader, project_path=project_path)


    if os.path.isfile(args.ckpt_path):
        print("Found pretrained model, loading...")
        model: torch.nn.Module = hydra.utils.instantiate(cfg.models.model).to('cuda')
        checkpoint_state_dict = torch.load(args.ckpt_path)['state_dict']
        clone_state_dict = {key[4:]: checkpoint_state_dict[key] for key in checkpoint_state_dict.keys()}
        model.load_state_dict(clone_state_dict)
        model.eval()
        out = []
        for idx, batch_input in enumerate(val_loader):
            out.append(model(batch_input[0].to('cuda')))

        attn_maps = out[1]
        print('input sequence', test_seq, '\n',
              'test label:', torch.flip(test_seq, dims=(1,)), '\n',
              'output prediction:', out[0].argmax(-1), '\n', )
        attn_maps = torch.stack(attn_maps, dim=0)
        plot_attention_maps(test_seq, attn_maps, idx=0)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str,
                        required=True,
                        help="the path of pretrained .ckpt model, should have shape "
                             "like:smart_sensor_fusion/results/model name/dataset name/task "
                             "name+time stamp+test/checkpoints/...ckpt")
    parser.add_argument('--test_seq', type=int, nargs='+', help='input test sequence, should be a list, e.g. '
                                                                '--test_seq 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7')
    args = parser.parse_args()

    initialize(version_base=None, config_path="../configs", job_name="test_app")
    cfg = compose(config_name="config")
    print(OmegaConf.to_yaml(cfg))

    inference(cfg=cfg, args=args)
