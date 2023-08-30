import argparse
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from src.datasets.progress_prediction import ImitationEpisode
import unittest


class TestAddNumbers(unittest.TestCase):
    def test_progress_prediction(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--num_stack", default=5, type=int)
        parser.add_argument("--frameskip", default=5, type=int)
        p = parser.parse_args()

        data_folder = "/home/jin4rng/Documents/code/see_hear_feel/data/test_recordings"
        train_csv = "/home/jin4rng/Documents/code/see_hear_feel/data/train.csv"
        val_csv = "/home/jin4rng/Documents/code/see_hear_feel/data/val.csv"
        train_num_episode = len(pd.read_csv(train_csv))
        val_num_episode = len(pd.read_csv(val_csv))
        batch_size = 32

        train_set = torch.utils.data.ConcatDataset(
            [
                ImitationEpisode(train_csv, p, i, data_folder)
                for i in range(train_num_episode)
            ]
        )
        val_set = torch.utils.data.ConcatDataset(
            [
                ImitationEpisode(val_csv, p, i, data_folder, False)
                for i in range(val_num_episode)
            ]
        )

        train_loader = DataLoader(
            train_set, batch_size, num_workers=8,
        )
        val_loader = DataLoader(val_set, 1, num_workers=8, shuffle=False)
        for idx, data in enumerate(train_loader):
            self.assertEqual(data[0].shape[1:], torch.Size([1, 40000]))
        for idx, data in enumerate(val_loader):
            self.assertEqual(data[0].shape[1:], torch.Size([1, 40000]))


if __name__ == '__main__':
    unittest.main()
