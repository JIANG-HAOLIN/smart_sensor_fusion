import unittest
import torch
from src.models.utils.positional_encoding import StandardPositionalEncoding, TemporalPositionalEncoding


class TestAddNumbers(unittest.TestCase):

    def test_standard_postional_encoding(self):
        pe = StandardPositionalEncoding()
        input = torch.randn([2, 1000, 256])
        self.assertEqual(pe(input).shape, torch.Size([2, 1000, 256]))


class TestTemporalPositionalEncoding(unittest.TestCase):
    def test_temporal_positional_encoding(self):
        pe = TemporalPositionalEncoding()
        input = torch.randn([2, 128, 64])
        self.assertEqual(pe(input).shape, torch.Size([2, 128, 64]))

if __name__ == '__main__':
    unittest.main()
