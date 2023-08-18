import os
import sys
import torch
import argparse
import numpy as np
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf


def inference(cfg: DictConfig, args: argparse.Namespace) -> None:
    """

    Args:
        cfg: hydra config file
        args: the input arguments

    Output the prediction sequence and visualization of the attention map

    """
    from utils.plot_attn import plot_attention_maps
    test_seq = torch.randint(10, size=(1, 17), device='cuda') if args.test_seq is None \
        else torch.from_numpy(np.array(args.test_seq)).unsqueeze(0).to('cuda')
    torch.set_float32_matmul_precision('medium')
    batch_input = torch.nn.functional.one_hot(test_seq, num_classes=10).float()
    if os.path.isfile(args.ckpt_path):
        print("Found pretrained model, loading...")
        # pretrained_mdl = TransformerPredictorPl.load_from_checkpoint(args.ckpt_path) # if we load the whole lightningModule
        pretrained_mdl = torch.load(args.ckpt_path)['hyper_parameters']['mdl']
        pretrained_mdl.eval()
        # pretrained_mdl.freeze()  # only for lightningModule
        # out = pretrained_mdl.mdl(batch_input)  # only for lightningModule
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
    print(project_path)
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str,
                        default='/home/jin4rng/Documents/code/smart_sensor_fusion/results'
                                '/simple_transformer_multiblocks/logits/transpose08-18-11:58:20test/checkpoints/08-18'
                                '-11:58:21-jobid=0-epoch=1-step=780.ckpt',
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
