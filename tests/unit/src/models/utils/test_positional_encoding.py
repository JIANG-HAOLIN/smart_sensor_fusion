import unittest
import torch
from src.models.utils.positional_encoding import StandardPositionalEncoding, TemporalPositionalEncoding
import matplotlib.pyplot as plt

class TestStandardPositionalEncoding(unittest.TestCase):

    def test_standard_positional_encoding(self):
        posemb = StandardPositionalEncoding()
        input = torch.randn([2, 1000, 256])

        pe = posemb.pe.squeeze()[:100, :128].T.cpu().numpy()

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3))
        pos = ax.imshow(pe, cmap="RdGy", extent=(1, pe.shape[1] + 1, pe.shape[0] + 1, 1))
        fig.colorbar(pos, ax=ax)
        ax.set_xlabel("Position in sequence")
        ax.set_ylabel("Hidden dimension")
        ax.set_title("Positional encoding over hidden dimensions")
        ax.set_xticks([1] + [i * 10 for i in range(1, 1 + pe.shape[1] // 10)])
        ax.set_yticks([1] + [i * 10 for i in range(1, 1 + pe.shape[0] // 10)])
        plt.show()

        self.assertEqual(posemb(input).shape, torch.Size([2, 1000, 256]))


class TestTemporalPositionalEncoding(unittest.TestCase):
    def test_temporal_positional_encoding(self):
        pe = TemporalPositionalEncoding()
        input = torch.randn([2, 128, 64])
        self.assertEqual(pe(input).shape, torch.Size([2, 128, 64]))

if __name__ == '__main__':
    unittest.main()
