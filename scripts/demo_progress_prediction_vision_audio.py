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


project_path = os.path.abspath(os.path.join(__file__, '..', '..'))


@hydra.main(version_base=None, config_path=" ", config_name='config')
def inference(cfg: DictConfig) -> None:
    """

    Args:
        cfg: hydra config file
        args: the input arguments

    Output the validation accuracy and visualization of the confusion matrix

    """
    from utils.visualizations import plot_confusion_matrix
    torch.set_float32_matmul_precision('medium')
    _, val_loader, _ = hydra.utils.instantiate(cfg.datasets.dataloader, project_path=project_path)
    cfg_path = HydraConfig.get().runtime['config_sources'][1]['path']
    checkpoints_path = os.path.abspath(os.path.join(cfg_path, '..', 'checkpoints', cfg.models.inference.ckpt_path))
    if os.path.isfile(checkpoints_path):
        print("Found pretrained model, loading...")
        model: torch.nn.Module = hydra.utils.instantiate(cfg.models.model, _recursive_=False).to('cpu')
        checkpoint_state_dict = torch.load(checkpoints_path)['state_dict']
        clone_state_dict = {key[4:]: checkpoint_state_dict[key] for key in checkpoint_state_dict.keys()}
        model.load_state_dict(clone_state_dict)
        model.eval()
        outs = []
        labels = []
        val_accu = 0
        with torch.no_grad():
            for idx, batch_input in enumerate(val_loader):
                inp_data, _, _, _, _, label = batch_input
                label_copy = copy.deepcopy(label)
                # Perform prediction and calculate loss and accuracy
                output = model.forward(inp_data[1][:, -1, :, :], inp_data[4])
                out, attn_map = output[0].detach(), output[1]
                out = torch.argmax(out, dim=1, keepdim=False)
                acc = (out == label).float().mean()
                val_accu = (val_accu * idx + acc) / (idx + 1)
                outs.append(out.numpy())
                labels.append(label_copy.numpy())
            print(val_accu)
            outs = np.asarray(outs)
            labels = np.asarray(labels)
            plot_confusion_matrix(outs, labels, save_pth=os.path.join(project_path + '/scripts'))


if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
    inference()
