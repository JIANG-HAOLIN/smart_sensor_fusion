import torch.nn as nn
import torch
import numpy as np
from typing import List


class SelectToken(nn.Module):
    """from ImageBind helpers"""

    def __init__(self, index) -> None:
        super().__init__()
        self.index = index

    def forward(self, x):
        assert x.ndim == 3
        return x[:, self.index, ...]


class Normalize1Dim(nn.Module):
    """from ImageBind helpers"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return nn.functional.normalize(x, dim=self.dim, p=2)


class LearnableLogitScaling(nn.Module):
    """from ImageBInd helpers"""

    def __init__(
            self,
            logit_scale_init: float = 1 / 0.07,
            learnable: bool = True,
            max_logit_scale: float = 100,
    ) -> None:
        super().__init__()
        self.max_logit_scale = max_logit_scale
        self.logit_scale_init = logit_scale_init
        self.learnable = learnable
        log_logit_scale = torch.ones([]) * np.log(self.logit_scale_init)
        if learnable:
            self.log_logit_scale = nn.Parameter(log_logit_scale)
        else:
            self.register_buffer("log_logit_scale", log_logit_scale)

    def forward(self, x):
        return torch.clip(self.log_logit_scale.exp(), max=self.max_logit_scale) * x

    def extra_repr(self):
        st = f"logit_scale_init={self.logit_scale_init},learnable={self.learnable}," \
             f" max_logit_scale={self.max_logit_scale}"
        return st


class ImageBindNceHeader(nn.Module):
    def __init__(self, model_dim: int = 256,
                 dropout: float = 0.):
        super().__init__()
        self.proj_net = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
            Normalize1Dim(dim=-1),
            LearnableLogitScaling(logit_scale_init=5.0, learnable=True),
        )

    def forward(self, x):
        return self.proj_net(x)


def shuffle_sequence(sequence: List, reorder_prob: float = 0.15):
    """
    function to reorder the elements of a sequence with
    Args:
        sequence: input list to be shuffled(with arbitrary type of elements)
        reorder_prob: the chance of each element to be picked to be shuffled

    Returns: the shuffled sequence and target(shuffled position annotated with where the original elements is after the
             shuffle, while the unshuffled annotated as -1

    """
    import random
    import copy

    sequence = copy.deepcopy(sequence)
    target = [-1] * len(sequence)
    selected_pos = []
    to_be_shuffled_index = []
    for i, pos_id in enumerate(sequence):
        prob = random.random()
        if prob < reorder_prob:
            selected_pos.append(i)
            to_be_shuffled_index.append(pos_id)

    c = copy.deepcopy(list(zip(selected_pos, to_be_shuffled_index)))
    random.shuffle(c)
    shuffled_selected_pos, shuffled_index = zip(*c)
    # print(selected_pos, to_be_shuffled_index)
    # print(shuffled_selected_pos, shuffled_index)
    for i, (og_pos, cur_pos, element) in enumerate(zip(selected_pos, shuffled_selected_pos, shuffled_index)):
        sequence[og_pos] = element
        target[cur_pos] = og_pos
    # print(sequence)
    # print(target)
    return sequence, target


def get_mask_sequence1d(seq_len: int, mask_prob: float = 0.15, mask_length: int = 10,
                        unmask_mark: float = 1., mask_mark: float = 0.):
    """randomly select starting indices with a prob of mask_poss and mask the subsequent mask_length time-steps.
    only for 1D sequence !!!
    Args:

        seq_len: len of mask
        mask_prob: prob of selecting each element as start index for masking
        mask_length: length of masking region
        mask_mark: label for mask elements
        unmask_mark: label for unmasked elements

    Returns: a mask where unmask region marked as 1 while masked region marked as 0

    """
    import random
    mask = [unmask_mark] * seq_len
    init_indices = []
    for idx, ele in enumerate(mask):
        prob = random.random()
        if prob < mask_prob:
            init_indices.append(idx)
    for init_index in init_indices:
        end_index = min(init_index + mask_length, seq_len)
        for idx in range(init_index, end_index):
            mask[idx] = mask_mark
    return mask
