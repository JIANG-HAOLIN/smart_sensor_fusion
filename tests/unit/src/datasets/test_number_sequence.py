import unittest
from src.datasets.number_sequence import ReverseDataset, data



class TestAddNumbers(unittest.TestCase):

    def test_num_sequence(self):
        dataset = ReverseDataset()
        train_loader = data.DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True, pin_memory=True)
        inp_data, labels = train_loader.dataset[0]
        self.assertEqual(inp_data.shape, (dataset.seq_len,))
        self.assertEqual(labels.shape, (dataset.seq_len,))

if __name__ == '__main__':
    unittest.main()

