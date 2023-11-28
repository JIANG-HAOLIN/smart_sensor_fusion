import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.interpolate_array import linearly_interpolated_array


class BautiroDrillingDataset(Dataset):
    """Input pipeline for bautiro drilling dataset using moving window and returns only the signal with _C or _corr"""

    def __init__(self, data_folder: str, window_size: int = 1000, step_size: int = 500, train: bool = True):
        """

        Args:
            data_folder_path: path to the data folder
            window_size: the size of moving window
            step_size: the step size of moving window
            train: whether it is for training process
        """
        self.data_folder_path = os.path.join(data_folder, 'processed_drilling_data/separate_data/np_arr')
        signal_list = os.listdir(self.data_folder_path)
        if 'Fz_T_corr' in signal_list:
            signal_list.remove('Fz_T_corr')
            signal_list.remove('Fz_T_res')
        signal_list_c = signal_list.copy()
        signal_list_c = [sig for sig in signal_list_c if '_C' in sig or '_corr' in sig]
        self.signal_list = signal_list_c
        self.signal_folder_path = [os.path.join(self.data_folder_path, i) for i in self.signal_list]
        self.window_size = window_size
        self.step_size = step_size
        self.train = train
        self.traj_list = list(range(1, 14)) + list(range(17, 30)) if train else [14, 15, 16, 30, 31, 32]
        self.mov_len_dict = self.get_movement_len()
        self.cumulative_lengths, self.cumulative_step = self.calculate_cumulative_lengths_and_steps()


    def get_movement_len(self):
        """get valid length according to acx"""
        len_dict = {}
        for signal in self.signal_folder_path:
            if 'acx' in signal:
                for traj in self.traj_list:
                    traj_path = os.path.join(signal, str(traj) + '.npy')
                    arr = np.load(traj_path, mmap_mode='r')
                    arr_len = np.count_nonzero(~np.isnan(arr)) - 1
                    print(f'at {traj_path}: valid len {arr_len} time {arr_len/50000} \n')
                    len_dict[traj] = arr_len
        return len_dict

    def calculate_cumulative_lengths_and_steps(self):
        """output the cumulate number of observation and steps of moving windows for each signal"""
        cumulative_step = {}
        cumulative_lengths = {}
        for signal in self.signal_folder_path:
            signal_name = signal.split('/')[-1]
            for traj in self.traj_list:
                arr_len = self.mov_len_dict[traj]
                num = (((arr_len - self.window_size) // self.step_size) + 1 + 1)
                if signal_name not in cumulative_lengths:
                    cumulative_lengths[signal_name] = [arr_len]
                    cumulative_step[signal_name] = [num]
                else:
                    cumulative_lengths[signal_name].append(cumulative_lengths[signal_name][-1] + arr_len)
                    cumulative_step[signal_name].append(cumulative_step[signal_name][-1] + num)
        return cumulative_lengths, cumulative_step

    def get_array_and_offset(self, step: int):
        """for each step getting the corresponding start and end of sequence"""
        # because for each user all signals have the same duration so iteration over signals is not needed here as i
        # use break to end the iteration after the first signal
        interval_dict = {}
        for signal, cumulative_step_list in self.cumulative_step.items():
            for idx, p2 in enumerate(cumulative_step_list):
                if step + 1 < p2:
                    pre_step = 0 if idx == 0 else cumulative_step_list[idx - 1]
                    start = (step - pre_step) * self.step_size
                    end = start + self.window_size - 1
                    return self.traj_list[idx], start, end
                elif step + 1 == p2:
                    pre_len = 0 if idx == 0 else self.cumulative_lengths[signal][idx - 1]
                    end = self.cumulative_lengths[signal][idx] - pre_len - 1
                    start = end - self.window_size + 1
                    return self.traj_list[idx], start, end

    def cat_pos_remove_nan(self, subset: np.array) -> np.array:
        """concatenate the positional index and remove the NaN vaule to produce input for early concatenation models"""
        not_nan = ~np.isnan(subset)
        pos = np.arange(len(subset))[not_nan]
        pos = pos.reshape(1, -1)
        subset = subset[not_nan].reshape(1, -1)
        out = np.concatenate([pos, subset], axis=0)
        return out

    def interpolate_downsample(self, subset: np.array) -> np.array:
        subset = linearly_interpolated_array(subset)
        return subset

    def get_frequency(self, signal_name: str) -> int:
        if 'ac' in signal_name or 'ap' in signal_name:
            return 4
        elif '_C' in signal_name:
            return 50
        else:
            return 400

    def __getitem__(self, step):
        data = {}
        traj_idx, start, end = self.get_array_and_offset(step)
        for idx, signal_path in enumerate(self.signal_folder_path):
            signal_name = self.signal_list[idx]
            sig_freq = self.get_frequency(signal_name)
            traj_path = os.path.join(signal_path, str(traj_idx) + '.npy')
            subset = np.load(traj_path, mmap_mode='r')
            subset = subset[start: end + 1]
            if np.all(np.isnan(subset)):
                print(f'all NaN array at {signal_path, traj_idx, start, end}')
            subset = self.interpolate_downsample(subset)
            subset = subset[::sig_freq]
            subset = torch.tensor(subset, dtype=torch.float32)
            data[signal_name] = subset
        return data

    def __len__(self):
        num = 0
        for signal in self.signal_folder_path:
            for traj in self.traj_list:
                arr_len = self.mov_len_dict[traj]
                num += (((arr_len - self.window_size) // self.step_size) + 1 + 1)
            break
        return num


def get_loaders(data_folder: str, window_size: int = 1000, step_size: int = 500, train: bool = True,
                batch_size: int = 32, shuffle: bool = True, drop_last: bool = True, **kwargs):
    train_dataset = BautiroDrillingDataset(data_folder, window_size, step_size, train)
    val_dataset = BautiroDrillingDataset(data_folder, window_size, step_size, train=False)

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=16), \
        DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=16), \
        DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=16),
