import logging
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import os
import sys
import torch.nn as nn
import torch
from src.trainer_util import launch_trainer
from datetime import datetime

log = logging.getLogger(__name__)


def set_random_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@hydra.main(config_path='configs', config_name='config_progress_prediction', version_base='1.2')
def train(cfg: DictConfig) -> None:
    set_random_seed(42)
    os.environ['HYDRA_FULL_ERROR'] = '1'
    os.environ['NUMEXPR_MAX_THREADS'] = '16'
    os.environ['NUMEXPR_NUM_THREADS'] = '8'
    torch.set_float32_matmul_precision('medium')
    project_path = os.path.abspath(os.path.join(__file__, '..'))
    hydra_cfg_og = HydraConfig.get()
    multirun_dir_path = hydra_cfg_og.sweep.dir

    log.info('*-------- train func starts --------*')
    log.info('output folder:' + multirun_dir_path + '\n')
    log.info('project_path:' + project_path + '\n')
    sys.path.append(project_path)
    log.info('sys.path:', )
    for p in sys.path:
        log.info(p)

    log.info(f"if Cuda available:{torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"Cuda info:\n{torch.cuda.get_device_properties('cuda')}")
        log.info(f"Cuda version:{torch.version.cuda}")
    else:
        log.info(f'no Cuda detected, using CPU instead !!')
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    log.info(f"Current working directory: {os.getcwd()}")
    log.info(f"Original working directory: {hydra.utils.get_original_cwd()}")
    log.info(f"Current Project path: {project_path}")
    log.info(f"current multi-run outp1ut path: {multirun_dir_path}")

    from utils.hydra_utils import extract_sweeper_output_label
    label = extract_sweeper_output_label(cfg, hydra_cfg_og.runtime.choices)
    log.info(f"current running output label: {label}")
    out_dir_path = os.path.join(multirun_dir_path, label + '_' + datetime.now().strftime("%m-%d-%H:%M:%S"))
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)
    log.info(f"current experiment output path: {out_dir_path}")
    with open(os.path.join(out_dir_path, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)

    model: nn.Module = hydra.utils.instantiate(cfg.models.model, _recursive_=False).to('cuda')
    log.info(f"model trainable params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    log.info(f"model non-trainable params:{sum(p.numel() for p in model.parameters() if not p.requires_grad)}")
    optimizer = hydra.utils.instantiate(cfg.optimizers.optimizer, params=model.parameters())
    lr_scheduler = hydra.utils.instantiate(cfg.optimizers.scheduler, optimizer=optimizer)
    train_loader, val_loader, test_loader = hydra.utils.instantiate(cfg.datasets.dataloader, project_path=project_path)
    pl_module = hydra.utils.instantiate(cfg.pl_modules.pl_module, model,
                                        optimizer, lr_scheduler,
                                        train_loader, val_loader, test_loader, _recursive_=False)

    launch_trainer(pl_module, out_dir_path=out_dir_path, label=label, hydra_conf=cfg,
                   model_name=cfg.models.name, dataset_name=cfg.datasets.name, task_name=cfg.task_name,
                   **cfg.trainers.launch_trainer)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    train()
