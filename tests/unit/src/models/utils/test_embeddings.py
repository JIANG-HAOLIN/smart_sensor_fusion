import unittest
import torch
from src.models.utils.embeddings import ModalTypeEmbedding, MdlLbCat, MdlEmbCat
from src.models.utils.embeddings import LearnablePosEmb


class TestModelTypeEmbedding(unittest.TestCase):

    def test_model_type_embedding(self):
        pe = ModalTypeEmbedding(2, 256)
        input = torch.randn([2, 1000, 256])
        self.assertEqual(pe(input, 0).shape, torch.Size([2, 1000, 256]))


class TestMdlLbCat(unittest.TestCase):

    def test_mdl_lb_cat(self):
        pe = MdlLbCat(5)
        input = torch.randn([2, 1000, 256])
        self.assertEqual(pe(input, 3).shape, torch.Size([2, 1000, 256 + 5]))


class TestMdlEmbCat(unittest.TestCase):

    def test_mdl_emb_cat(self):
        pe = MdlEmbCat(2, 256)
        input = torch.randn([2, 1000, 256])
        self.assertEqual(pe(input, 0).shape, torch.Size([2, 1000, 512]))


class TestLearnablePosEmb(unittest.TestCase):

    def test_learnable_pos_emb(self):
        pe = LearnablePosEmb(2, 4)
        input = torch.zeros([4, 2, 3, 4])
        self.assertEqual(pe(input, 1).shape, torch.Size([4, 2, 3, 4]))
        print(pe(input, 1))


if __name__ == '__main__':
    unittest.main()
