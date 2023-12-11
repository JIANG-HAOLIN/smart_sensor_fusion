import os
import sys
import torch
import argparse
from hydra.core.hydra_config import HydraConfig
import numpy as np
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf, open_dict


@hydra.main(version_base=None, config_path="../configs/models", config_name='basic_transformer')
def inference(cfg: DictConfig) -> None:
    """

    Args:
        cfg: hydra config file

    Output the prediction sequence and visualization of the attention map

    """
    from utils.visualizations import plot_attention_maps
    test_seq = torch.randint(10, size=(1, 17), device='cpu') if cfg.inference.test_seq is None \
        else torch.from_numpy(np.array(cfg.inference.test_seq)).unsqueeze(0).to('cpu')
    torch.set_float32_matmul_precision('medium')
    batch_input = torch.nn.functional.one_hot(test_seq, num_classes=10).float()
    if os.path.isfile(cfg.inference.ckpt_path):
        print("Found pretrained model, loading...")
        pretrained_mdl: torch.nn.Module = hydra.utils.instantiate(cfg.model).to('cpu')
        checkpoint_state_dict = torch.load(cfg.inference.ckpt_path)['state_dict']
        clone_state_dict = {key[4:]: checkpoint_state_dict[key] for key in checkpoint_state_dict.keys()}
        pretrained_mdl.load_state_dict(clone_state_dict)
        pretrained_mdl.eval()
        out = pretrained_mdl(batch_input)
        print('Done loading')
        attn_maps = out[1]
        print('input sequence', test_seq, '\n',
              'test label:', torch.flip(test_seq, dims=(1,)), '\n',
              'output prediction:', out[0].argmax(-1), '\n', )
        attn_maps = torch.stack(attn_maps, dim=0)
        plot_attention_maps(test_seq, attn_maps, idx=0)


if __name__ == "__main__":
    project_path = os.path.abspath(os.path.join(__file__, '..', '..'))
    sys.path.append(project_path)  # so we can import the modules inside the project when running in terminal

    inference()
