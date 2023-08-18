import os
import torch
from src.pl_modules.tutorial_6_transpose import TransformerPredictorPl

from utils.plot_attn import plot_attention_maps


def inference(test_seq: torch.Tensor,
              ckpt_path: str) -> None:
    """

    Args:
        test_seq: input test sequence with shape [batch size, 17]
        ckpt_path: absolution path of checkpoint

    Output the prediction sequence and visualization of the attention map

    """
    torch.set_float32_matmul_precision('medium')
    batch_input = torch.nn.functional.one_hot(test_seq, num_classes=10).float()
    if os.path.isfile(ckpt_path):
        print("Found pretrained model, loading...")
        pretrained_mdl = TransformerPredictorPl.load_from_checkpoint(ckpt_path)
        pretrained_mdl.eval()
        pretrained_mdl.freeze()
        out = pretrained_mdl.mdl(batch_input)
        attn_maps = out[1]
        print('test label:', torch.flip(test_seq, dims=(1,)), '\n',
              'output prediction:', out[0].argmax(-1), '\n',)
        attn_maps = torch.stack(attn_maps, dim=0)
        plot_attention_maps(test_seq, attn_maps, idx=0)
        print('Done loading')


if __name__ == "__main__":
    test_seq = torch.randint(10, size=(1, 17), device='cuda')
    inference(test_seq,
              '/home/jin4rng/Documents/code/smart_sensor_fusion/results/simple_transformer_multiblocks/logits/transpose08-17-15:28:27test/checkpoints/08-17-15:28:28-jobid=0-epoch=2-step=1170.ckpt')


