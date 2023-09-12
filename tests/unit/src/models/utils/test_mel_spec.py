from src.models.utils.mel_spec import MelSpec
import torch
import unittest


class TestMelSpec(unittest.TestCase):
    def test_mel_spec(self):
        length = 40000
        sr = 16000
        hop_ratio = 0.05
        n_mel = 64
        mel = MelSpec(length=length, hop_ratio=hop_ratio, n_mels=n_mel, sr=sr, )
        input = torch.randn([2, 1, length])
        out = mel(input)
        self.assertEqual(out.shape, torch.Size([2, 1, n_mel, int(length/(sr*hop_ratio))+1]))
        self.assertEqual(out.shape[2:], mel.out_size)
        print(out.shape)


if __name__ == '__main__':
    unittest.main()
