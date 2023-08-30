import torch
from src.models.utils.to_patches import Img2Patches
import unittest


class TestAddNumbers(unittest.TestCase):
    def test_to_patches(self):
        in_h = 64
        in_w = 251
        patch_size = (8, 8)
        i2p = Img2Patches(input_size=(in_h, in_w), patch_size=patch_size)
        input = torch.randn([2, 1, in_h, in_w])
        out = i2p(input)
        self.assertEqual(out.shape, torch.Size([2,
                                                int(in_h/patch_size[0])*int(in_w/patch_size[1]),
                                                patch_size[0]*patch_size[1]]))


if __name__ == '__main__':
    unittest.main()
