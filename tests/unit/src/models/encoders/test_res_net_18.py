from src.models.encoders.res_net_18 import make_audio_encoder
import torch
import unittest


class TestMelSpec(unittest.TestCase):
    def test_mel_spec(self):
        mdl = make_audio_encoder(128)
        input = torch.randn([2, 1, 64, 251])
        out = mdl(input)
        self.assertEqual(out.shape, torch.Size([2, 128, 8, 32]))


if __name__ == '__main__':
    unittest.main()