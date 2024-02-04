import unittest
import torch
from src.models.utils.helpers import get_scatter_idx_target, get_mask_sequence1d


class Test(unittest.TestCase):

    def test_get_scatter_idx_target(self):
        sequence = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        seq, target = get_scatter_idx_target(sequence, 0.3, fix=True)
        print(seq, target)

    def test_get_mask_sequence1d(self):
        seq_len = 10
        mask = get_mask_sequence1d(seq_len,
                                   mask_prob=0.5,
                                   mask_length=1, )
        # print(mask)
        print(mask.count(0) / len(mask))
        print(torch.tensor(mask))
        print(1 - torch.tensor(mask))


if __name__ == '__main__':
    unittest.main()
