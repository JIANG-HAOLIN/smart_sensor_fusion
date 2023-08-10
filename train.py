# train.py
import logging
from omegaconf import DictConfig, OmegaConf
import hydra
import argparse


try:
    import pytorch_lightning as pl
except ModuleNotFoundError:
    print("module pytorch_lighting not found")
import os
import sys
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from src.Models.trafo_predictor import TransformerPredictor as test_structure
from src.Datasets.number_sequence import get_loaders


print(
    '---------------------------------------\n'
    '----------------- Start ---------------\n'
    '---------------------------------------'
)

log = logging.getLogger(__name__)
project_path = os.path.abspath(os.path.join(__file__, '..'))
print('project_path:\n', project_path, '\n')
sys.path.append(project_path)
# print('sys.path:', '\n', *sys.path)
print('sys.path:',)
for p in sys.path:
    print(p)
conf_dir_path = os.path.join(project_path, 'conf')

print(
    '---------------------------------------\n'
    '-------------train func start ---------\n'
    '---------------------------------------'
)

@hydra.main(config_path = 'configs', config_name = 'config', version_base= None)
def train(cfg: DictConfig) -> None:
    os.environ['HYDRA_FULL_ERROR'] = '1'

    log.info(f"if Cuda available:{torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"Cuda info:\n{torch.cuda.get_device_properties('cuda')}")
    else:
        log.info(f'no Cuda detected, using CPU instead !!')
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    log.info(f"Current working directory: {os.getcwd()}")
    log.info(f"Original working directory: {hydra.utils.get_original_cwd()}")
    log.info(f"Current Project path: {project_path}")

    model: nn.Module = hydra.utils.instantiate(cfg.models.model)
    optimizer = optim.Adam(params=model.parameters(), **cfg.optimizers.optimizer)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, **cfg.optimizers.scheduler)
    train_loader, val_loader, _ = get_loaders(**cfg.datasets.dataloader)
    pl_module = hydra.utils.instantiate(cfg.pl_modules.pl_module, model, lr_scheduler, optimizer, train_loader, val_loader,)


if __name__ == "__main__":
    train()



