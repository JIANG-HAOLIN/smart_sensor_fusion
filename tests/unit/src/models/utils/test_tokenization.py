import torch
from src.models.utils.tokenization import Vanilla2dTokenization, Vanilla1dTokenization
import unittest


class TestVanilla1dTokenization(unittest.TestCase):
    def test_1(self):
        test_seq = torch.arange(1, 97).reshape(1, 2, 48).float()
        tokenizor = Vanilla1dTokenization(channel_size=2, out_dim=17, patch_size=5, input_size=48)
        out = tokenizor(test_seq)
        self.assertEqual(out.shape, torch.Size([1, 9, 17]))


class TestVanilla2dTokenization(unittest.TestCase):
    def test_2(self):
        test_seq = torch.arange(108).reshape(1, 2, 6, 9).float()
        tokenizor = Vanilla2dTokenization(channel_size=2, model_dim=17, patch_size=(2, 2), input_size=(6, 9))
        out = tokenizor(test_seq)
        self.assertEqual(out.shape, torch.Size([1, 12, 17]))


if __name__ == '__main__':
    unittest.main()
