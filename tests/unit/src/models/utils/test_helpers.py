import unittest
import torch



class Test(unittest.TestCase):

    def test_get_scatter_idx_target(self):
        from src.models.utils.helpers import get_scatter_idx_target
        sequence = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        seq, target = get_scatter_idx_target(sequence, 0.3, fix=True)
        print(seq, target)

    def test_get_mask_sequence1d(self):
        from src.models.utils.helpers import get_mask_sequence1d
        seq_len = 10
        mask = get_mask_sequence1d(seq_len,
                                   mask_prob=0.5,
                                   mask_length=1, )
        # print(mask)
        print(mask.count(0) / len(mask))
        print(torch.tensor(mask))
        print(1 - torch.tensor(mask))

    def test_cosine_loss_fn(self):
        from src.models.utils.helpers import get_mask_sequence1d
        from src.models.utils.helpers import cosine_loss_fn
        def loss_fn(x, y):
            x = torch.nn.functional.normalize(x, dim=-1, p=2)
            y = torch.nn.functional.normalize(y, dim=-1, p=2)
            return (2 - 2 * (x * y).sum(dim=-1)).mean()
        mask = get_mask_sequence1d(8,
                                   mask_prob=0.5,
                                   mask_length=1, )
        mask = torch.tensor(mask).reshape(2, 4, 1)
        # input = torch.arange(1, 17).reshape(2, 4, 2).float()
        # target = torch.arange(2, 18).reshape(2, 4, 2).float()
        input = torch.randn([2, 4, 4])
        target = torch.rand([2, 4, 4])
        print(cosine_loss_fn(input, target, mask))
        print(loss_fn(input, target))
        print(cosine_loss_fn(input, target, None))

    def test_mse_fn(self):
        from src.models.utils.helpers import get_mask_sequence1d
        from src.models.utils.helpers import mse_fn
        mask = get_mask_sequence1d(8,
                                   mask_prob=0.5,
                                   mask_length=1, )
        mask = torch.tensor(mask).reshape(2, 4, 1)
        # input = torch.arange(1, 33).reshape(2, 4, 4).float()
        # target = torch.arange(2, 34).reshape(2, 4, 4).float()
        input = torch.randn([2, 4, 4])
        target = torch.rand([2, 4, 4])
        print(mse_fn(input, target, mask))
        print(torch.nn.functional.mse_loss(input, target))
        print(mse_fn(input, target, None))


if __name__ == '__main__':
    unittest.main()
