import unittest
import torch
from src.models.transformer_implementations import TransformerEncoder, TransformerEncoderVanilla


class TestTransformerEncoder(unittest.TestCase):

    def test_transformer_encoder(self):
        tf = TransformerEncoder(token_dim=10,
                                num_blocks=3,
                                num_heads=2,
                                dropout=0.,
                                norm_first=True)
        input = torch.randn([2, 17, 10])
        out = tf(input)
        self.assertEqual(out[0].shape, torch.Size([2, 17, 10]))
        self.assertEqual(len(out[1]), 3)
        self.assertEqual(out[1][0].shape, torch.Size([2, 2, 17, 17]))

    def test_transformer_encoder_vanilla(self):
        tf = TransformerEncoderVanilla(token_dim=10,
                                       num_blocks=3,
                                       num_heads=2,
                                       dropout=0.,
                                       norm_first=True)
        input = torch.randn([2, 17, 10])
        out = tf(input)
        self.assertEqual(out[0].shape, torch.Size([2, 17, 10]))
        self.assertEqual(len(out[1]), 3)
        self.assertEqual(out[1][0].shape, torch.Size([2, 2, 17, 17]))


if __name__ == '__main__':
    unittest.main()
