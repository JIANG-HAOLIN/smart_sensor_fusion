import unittest
import torch
from src.models.utils.embeddings import ModalTypeEmbedding


class TestModelTypeEmbedding(unittest.TestCase):

    def test_model_type_embedding(self):
        pe = ModalTypeEmbedding(2, 256)
        input = torch.randn([2, 1000, 256])
        self.assertEqual(pe(input, 0).shape, torch.Size([2, 1000, 256]))


if __name__ == '__main__':
    unittest.main()
