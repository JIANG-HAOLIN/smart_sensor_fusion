import unittest
import torch
from src.models.trafo_predictor import TransformerPredictor, TransformerPredictor_Pytorch


class TestTransformerpredictor(unittest.TestCase):

    def test_transformer_predictor(self):
        tf = TransformerPredictor(input_dim=10,
                                  model_dim=32,
                                  num_heads=2,
                                  num_classes=10,
                                  num_layers=3,
                                  dropout=0.0, )
        input = torch.randn([2, 17, 10])
        out = tf(input)
        self.assertEqual(out[0].shape, torch.Size([2, 17, 10]))
        self.assertEqual(out[1][0].shape, torch.Size([2, 2, 17, 17]))


class TestTransformerpredictor_Pytorch(unittest.TestCase):

    def test_transformer_prediction_pytorch(self):
        tf = TransformerPredictor_Pytorch(input_dim=10,
                                          model_dim=32,
                                          num_heads=2,
                                          num_classes=10,
                                          num_layers=3,
                                          dropout=0.0, )
        input = torch.randn([2, 17, 10])
        out = tf(input)
        self.assertEqual(out[0].shape, torch.Size([2, 17, 10]))
        self.assertEqual(out[1][0].shape, torch.Size([2, 2, 17, 17]))


if __name__ == '__main__':
    unittest.main()
