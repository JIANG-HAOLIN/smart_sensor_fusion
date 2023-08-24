import unittest
import torch
from src.models.trafo_predictor_pytorch import TransformerPredictor


class TestAddNumbers(unittest.TestCase):

    def test_trafo_prediction_multi(self):
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

if __name__ == '__main__':
    unittest.main()
