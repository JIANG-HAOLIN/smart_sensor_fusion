import unittest
import torch
from src.models.trafo_predictor import TransformerPredictor as NewTransformerPredictor
from src.models.trafo_classifier_vit import TransformerClassifierVit


class TestTransformerClassifierVit(unittest.TestCase):

    def test_transformer_classifier_bit(self):
        tf = TransformerClassifierVit(channel_size=3,
                                      model_dim=32,
                                      num_heads=4,
                                      num_classes=2,
                                      num_layers=3,
                                      dropout=0.0, )
        input = torch.randn([2, 3, 128, 256])
        out = tf(input)
        self.assertEqual(out[0].shape, torch.Size([2, 2]))
        self.assertEqual(out[1][0].shape, torch.Size([2, 4, 2049, 2049]))


if __name__ == '__main__':
    unittest.main()
