import torch
from src.models.utils.to_patches import Img2Patches, img_2_patches, seq_2_patches
import unittest


class Test1(unittest.TestCase):
    def test_to_patches(self):
        in_h = 7
        in_w = 7
        patch_size = (2, 2)
        i2p = Img2Patches(input_size=(in_h, in_w), patch_size=patch_size)
        input = torch.arange(1, 50).reshape(1, 1, 7, 7)
        out = i2p(input)
        print(out)
        self.assertEqual(out.shape, torch.Size([1,
                                                int(in_h/patch_size[0])*int(in_w/patch_size[1]),
                                                patch_size[0]*patch_size[1]]))


class Test2(unittest.TestCase):
    def test_img_2_patches(self):
        test_tensor = torch.arange(1, 49).reshape(1, 1, 6, 8).float()
        out = img_2_patches(test_tensor, patch_size=(2, 3))
        self.assertEqual(out.shape, torch.Size([1, 6, 6]))


class Test3(unittest.TestCase):
    def test_seq_2_patches(self):
        test_seq = torch.arange(1, 49).reshape(1, 1, 48).float()
        out = seq_2_patches(test_seq, patch_size=2)
        self.assertEqual(out.shape, torch.Size([1, 24, 2]))


if __name__ == '__main__':
    unittest.main()
