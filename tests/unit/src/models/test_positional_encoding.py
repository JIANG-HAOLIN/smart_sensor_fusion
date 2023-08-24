import unittest
import torch
from src.models.utils.positional_encoding import StandardPositionalEncoding


class TestAddNumbers(unittest.TestCase):

    def test_standard_postional_encoding(self):
        pe = StandardPositionalEncoding()
        input = torch.randn([2, 1000, 256])
        self.assertEqual(pe(input).shape, torch.Size([2, 1000, 256]))


if __name__ == '__main__':
    unittest.main()
