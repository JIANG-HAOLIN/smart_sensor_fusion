import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


def extract_sweeper_output_label(hydra_conf: DictConfig, choices: DictConfig):
    labels = ''
    for full_var_name in hydra_conf.variable_name.split('+'):
        conf = hydra_conf
        for attr in full_var_name.split('.'):
            # hydra_conf = hydra_conf.__getattr__(attr)  # also works
            conf = conf[attr]
        if isinstance(conf, DictConfig):  # only works for trainers, pl_modules, optimizers, models, datasets
            conf = choices[full_var_name].split('/')[-1]
        variable_name = full_var_name.strip().split('.')[-1]
        label = f'{variable_name}={conf}'
        labels = labels + label
    return labels
