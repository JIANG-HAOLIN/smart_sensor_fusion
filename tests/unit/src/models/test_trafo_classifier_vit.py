import unittest
import torch
from src.models.trafo_predictor import TransformerPredictor as NewTransformerPredictor
from src.models.trafo_classifier_vit import TransformerClassifierVit, VitImageBind


class TestTransformerClassifierVit(unittest.TestCase):

    def test_transformer_classifier_vit(self):
        tf = TransformerClassifierVit(channel_size=2,
                                      model_dim=32,
                                      num_heads=4,
                                      num_classes=2,
                                      num_layers=3,
                                      dropout=0.0,
                                      input_size=(64, 251))
        input = torch.randn([2, 2, 64, 251])
        out = tf(input)
        expected_outsize = (int(64 / 4) * int(251 / 4) + 1)
        self.assertEqual(out[0].shape, torch.Size([2, 2]))
        self.assertEqual(out[1][0].shape, torch.Size([2, 4, expected_outsize, expected_outsize]))

    def test_vit_image_bind(self):
        tf = VitImageBind(model_dim=512,
                          num_heads=8,
                          dropout=0.0,
                          input_dropout=0.0,
                          num_layers=2,
                          num_pos_emb=31, )
        input = torch.randn([2, 30, 512])
        out = tf(input)
        expected_outsize = 31
        self.assertEqual(out[0].shape, torch.Size([2, 512]))
        self.assertEqual(out[1][0].shape, torch.Size([2, 8, expected_outsize, expected_outsize]))


if __name__ == '__main__':
    unittest.main()
