import unittest
import torch
from src.models.ultrasonic_bind import SeparateEncoder


class TestTransformerEncoder(unittest.TestCase):

    def test_transformer_encoder(self):
        tf = SeparateEncoder()
        input = [torch.randn([3, 37500]), torch.randn([3, 37500]), torch.randn([3, 37500]),
                 torch.randn([3, 37500]), torch.randn([3, 37500]), torch.randn([3, 37500]),
                 torch.randn([3, 1000]), torch.randn([3, 1000]), torch.randn([3, 1000]),
                 torch.randn([3, 1000]), torch.randn([3, 1000]),
                 torch.randn([3, 1000]), torch.randn([3, 1000]),
                 ]
        out = tf(*input)
        print(out)

if __name__ == '__main__':
    unittest.main()