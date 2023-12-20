from src.models.encoders.res_net_18 import make_audio_encoder
from src.models.encoders.res_net_18 import make_vision_encoder
import torch
import unittest


class TestResNet18(unittest.TestCase):
    def test_make_audio_encoder(self):
        mdl = make_audio_encoder(out_dim=128, out_layer="layer4.1.relu_1")
        input = torch.randn([2, 1, 64, 251])
        out = mdl(input)
        self.assertEqual(out.shape, torch.Size([2, 1, 128]))

    def test_make_vision_encoder(self):
        mdl = make_vision_encoder(out_dim=128, out_layer="layer4.1.relu_1")
        input = torch.randn([2, 3, 67, 90])
        out = mdl(input)
        self.assertEqual(out.shape, torch.Size([2, 1, 128]))


if __name__ == '__main__':
    unittest.main()
