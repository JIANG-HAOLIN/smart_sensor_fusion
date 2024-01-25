from src.models.utils.mel_spec import MelSpec
import torch
import unittest


class TestMelSpec(unittest.TestCase):
    def test_mel_spec(self):
        windows_size = 0.025
        length = 80000
        sr = 16000
        hop = 0.01
        n_mel = 64
        mel = MelSpec(windows_size=windows_size,
                      length=length,
                      hop=hop,
                      n_mels=n_mel,
                      sr=sr,)
        input = torch.randn([2, 1, length])
        out = mel(input)
        self.assertEqual(out.shape, torch.Size([2, 1, n_mel, int(length/(sr*hop))+1]))
        self.assertEqual(out.shape[2:], mel.out_size)
        print(out.shape)


if __name__ == '__main__':
    unittest.main()
