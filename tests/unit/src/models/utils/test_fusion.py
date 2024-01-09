import unittest
import torch
from src.models.utils.fusion import EarlySum, EarlySumLinear


class Test(unittest.TestCase):

    def test_EarlySum(self):
        mod_names = ["vision", "audio", "tactile"]
        mdl = EarlySum(mod_names)

        inputs = {
            "vision": torch.randn([4, 5, 256]),
            "audio": torch.randn([4, 5, 256]),
            "tactile": torch.randn([4, 5, 256]),
        }

        sum = mdl(inputs)
        print(sum.shape)
        print(mdl)
        self.assertEqual(sum.shape, torch.Size([4, 5, 256]))

    def test_EarlySumLinear(self):
        mod_names = ["vision", "audio", "tactile"]
        mdl = EarlySumLinear(mod_names)

        inputs = {
            "vision": torch.randn([4, 5, 256]),
            "audio": torch.randn([4, 5, 256]),
            "tactile": torch.randn([4, 5, 256]),
        }

        sum = mdl(inputs)
        print(sum.shape)
        print(mdl)
        self.assertEqual(sum.shape, torch.Size([4, 5, 256]))


if __name__ == '__main__':
    unittest.main()
