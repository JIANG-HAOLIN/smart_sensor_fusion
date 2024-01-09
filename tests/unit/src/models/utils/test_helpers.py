import unittest
import torch
from src.models.utils.helpers import shuffle_sequence, get_mask_sequence1d


class Test(unittest.TestCase):

    def test_shuffle_sequence(self):
        sequence = ['0bear', '1tiger', '2riven', 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        seq, target = shuffle_sequence(sequence, 0.3)
        print(seq, target)

    def test_get_mask_sequence1d(self):
        seq_len = 100
        mask = get_mask_sequence1d(seq_len)
        print(mask)


if __name__ == '__main__':
    unittest.main()
