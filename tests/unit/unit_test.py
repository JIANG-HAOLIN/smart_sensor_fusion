import unittest
import torch
from src.datasets.number_sequence import ReverseDataset, data
from src.models.trafo_predictor_multiblocks import TransformerPredictor
from src.models.trafo_predictor import TransformerPredictor as NewTransformerPredictor
from src.models.positional_encoding import StandardPositionalEncoding
from src.models.transformer_implementations import TransformerEncoder


class TestAddNumbers(unittest.TestCase):

    def test_num_sequence(self):
        dataset = ReverseDataset()
        train_loader = data.DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True, pin_memory=True)
        inp_data, labels = train_loader.dataset[0]
        self.assertEqual(inp_data.shape, (dataset.seq_len,))
        self.assertEqual(labels.shape, (dataset.seq_len,))

    def test_transformer_encoder(self):
        tf = TransformerEncoder(token_dim=10,
                                num_blocks=3,
                                num_heads=2,
                                dropout=0.)
        input = torch.randn([2, 17, 10])
        out = tf(input)
        self.assertEqual(out[0].shape, torch.Size([2, 17, 10]))
        self.assertEqual(len(out[1]), 3)
        self.assertEqual(out[1][0].shape, torch.Size([2, 2, 17, 17]))

    def test_new_transformer_predictor(self):
        tf = NewTransformerPredictor(input_dim=10,
                                     model_dim=32,
                                     num_heads=2,
                                     num_classes=10,
                                     num_layers=3,
                                     dropout=0.0, )
        input = torch.randn([2, 17, 10])
        out = tf(input)
        self.assertEqual(out[0].shape, torch.Size([2, 17, 10]))
        self.assertEqual(out[1][0].shape, torch.Size([2, 2, 17, 17]))

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

    def test_standard_postional_encoding(self):
        pe = StandardPositionalEncoding()
        input = torch.randn([2, 1000, 256])
        self.assertEqual(pe(input).shape, torch.Size([2, 1000, 256]))


if __name__ == '__main__':
    unittest.main()
