from src.models.utils.mel_spec import MelSpec
import torch
import unittest


class TestTransformerClassifierVit(unittest.TestCase):
    def test_mel_spec(self):
        length = 600000
        mel = MelSpec(length=length)
        input = torch.randn([2, 1, length])
        out = mel(input)
        self.assertEqual(out.shape, torch.Size([2, 1, 64, int(length/160)+1]))
        self.assertEqual(out.shape[2:], mel.out_size)


if __name__ == '__main__':
    unittest.main()
