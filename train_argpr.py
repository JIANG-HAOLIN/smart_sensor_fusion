# train_argpr.py
import logging
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import os
import sys
import torch.nn as nn
import torch
from src.trainer_util import launch_trainer
import argparse
from typing import Optional


# @hydra.main(config_path='configs', config_name='config', version_base=None)
def train(cfg: DictConfig, args: argparse.Namespace) -> None:
    os.environ['HYDRA_FULL_ERROR'] = '1'
    os.environ['NUMEXPR_MAX_THREADS'] = '16'
    os.environ['NUMEXPR_NUM_THREADS'] = '8'
    torch.set_float32_matmul_precision('medium')
    project_path = os.path.abspath(os.path.join(__file__, '..'))
    out_dir_path = HydraConfig.get().run.dir
    # out_dir_path = cfg.hydra.run.dir
    if not os.path.exists(os.path.join(out_dir_path, 'configs')):
        os.makedirs(os.path.join(out_dir_path, 'configs'))
    with open(os.path.join(out_dir_path, 'configs', "config.yaml",), "w") as f:
        OmegaConf.save(config=cfg, f=f)

    log = logging.getLogger(__name__)
    log.info('*-------- train func starts --------*')
    log.info('output folder:' + out_dir_path + '\n')
    log.info('project_path:' + project_path + '\n')
    sys.path.append(project_path)
    log.info('sys.path:', )
    for p in sys.path:
        log.info(p)

    log.info(f"if Cuda available:{torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"Cuda info:\n{torch.cuda.get_device_properties('cuda')}")
    else:
        log.info(f'no Cuda detected, using CPU instead !!')
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    log.info(f"Current working directory: {os.getcwd()}")
    log.info(f"Original working directory: {hydra.utils.get_original_cwd()}")
    # log.info(f"Original working directory: {cfg.hydra.runtime.cwd}")
    log.info(f"Current Project path: {project_path}")
    log.info(f"current experiment output path: {out_dir_path}")

    model: nn.Module = hydra.utils.instantiate(cfg.models.model, _recursive_=False).to('cuda')
    optimizer = hydra.utils.instantiate(cfg.optimizers.optimizer, params=model.parameters())
    lr_scheduler = hydra.utils.instantiate(cfg.optimizers.scheduler, optimizer=optimizer)
    train_loader, val_loader, test_loader = hydra.utils.instantiate(cfg.datasets.dataloader, project_path=project_path)
    pl_module = hydra.utils.instantiate(cfg.pl_modules.pl_module, model,
                                        optimizer, lr_scheduler,
                                        train_loader, val_loader, test_loader)

    launch_trainer(pl_module, out_dir_path=out_dir_path,
                   model_name=cfg.models.name, dataset_name=cfg.datasets.name, task_name=cfg.task_name,
                   **cfg.trainers.launch_trainer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str,
                        default='config_vit',
                        help="the path of config file for specific task from configs folder")
    args = parser.parse_args()

    hydra.initialize(config_path="configs", version_base=None,)
    cfg = hydra.compose(config_name=args.config_name, return_hydra_config=True,)
    HydraConfig().cfg = cfg
    OmegaConf.resolve(cfg)

    train(cfg=cfg, args=args)
