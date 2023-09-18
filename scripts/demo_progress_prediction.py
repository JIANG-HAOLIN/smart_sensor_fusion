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


project_path = os.path.abspath(os.path.join(__file__, '..', '..'))


def get_val_loader(val_csv: str, args, data_folder: str, **kwargs):
    from src.datasets.progress_prediction import ImitationEpisode
    val_csv = os.path.join(project_path, val_csv)
    data_folder = os.path.join(project_path, data_folder)
    val_set = ImitationEpisode(val_csv, args, 0, data_folder, False)
    return DataLoader(val_set, 1, num_workers=8, shuffle=False)


@hydra.main(version_base=None, config_path=" ", config_name='config')
def inference(cfg: DictConfig) -> None:
    """

    Args:
        cfg: hydra config file
        args: the input arguments

    Output the validation accuracy and visualization of the confusion matrix

    """
    from utils.plot_confusion_matrix import plot_confusion_matrix
    torch.set_float32_matmul_precision('medium')
    val_loader = get_val_loader(**cfg.datasets.dataloader, project_path=project_path)
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
                label = batch_input[1].detach()
                input = batch_input[0].detach()
                output = model(input)
                out, attn_map = output[0].detach(), output[1]
                out = torch.argmax(out, dim=1, keepdim=False)
                acc = (label == out).float().mean()
                val_accu = (val_accu * idx + acc) / (idx + 1)
                outs.append(out.numpy())
                labels.append(label.numpy())
            print(val_accu)
            outs = np.asarray(outs)
            labels = np.asarray(labels)
            plot_confusion_matrix(outs, labels, save_pth=os.path.join(project_path + '/scripts'))


if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
    inference()
