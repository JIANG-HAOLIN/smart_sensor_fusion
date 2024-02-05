import unittest
import torch
from src.models.utils.header import ClassificationHead, MLPHead


class Test(unittest.TestCase):

    def test_ClassficationHead(self):
        batch_size = 4
        model_dim = 256
        num_classes = 10
        input = torch.randn([batch_size, model_dim])
        mdl = ClassificationHead(model_dim=model_dim, num_classes=num_classes)
        out = mdl(input)
        print(out.shape)
        self.assertEqual(out.shape, torch.Size([batch_size, num_classes]))

    def test_MLPHead(self):
        batch_size = 4
        model_dim = 256
        out_dim = 128
        input = torch.randn([batch_size, model_dim])
        mdl = MLPHead(in_dim=model_dim, hidden_dim=128, out_dim=out_dim, norm="layer")
        out = mdl(input)
        print(out.shape)
        self.assertEqual(out.shape, torch.Size([batch_size, out_dim]))


if __name__ == '__main__':
    unittest.main()
