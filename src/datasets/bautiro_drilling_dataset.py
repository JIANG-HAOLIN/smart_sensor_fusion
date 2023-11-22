import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class BautiroDrillingDataset(Dataset):
    def __init__(self, data_folder_path: str, window_size: int = 1000, step_size: int = 500, train: bool = True):
        self.data_folder_path = data_folder_path
        signal_list = os.listdir(self.data_folder_path)
        if 'Fz_T_corr' in signal_list:
            signal_list.remove('Fz_T_corr')
            signal_list.remove('Fz_T_res')
        self.signal_list = signal_list
        self.signal_folder_path = [os.path.join(data_folder_path, i) for i in self.signal_list]
        self.window_size = window_size
        self.step_size = step_size
        self.train = train
        self.traj_list = list(range(1, 14)) + list(range(17, 30)) if train else [14, 15, 16, 30, 31, 32]
        self.cumulative_lengths, self.cumulative_step = self.calculate_cumulative_lengths_and_steps()

    def calculate_cumulative_lengths_and_steps(self):
        cumulative_step = {}
        cumulative_lengths = {}
        for signal in self.signal_folder_path:
            signal_name = signal.split('/')[-1]
            for traj in self.traj_list:
                traj_path = os.path.join(signal, str(traj) + '.npy')
                arr_len = len(np.load(traj_path, mmap_mode='r'))
                num = (((arr_len - self.window_size) // self.step_size) + 1 + 1)
                if signal_name not in cumulative_lengths:
                    cumulative_lengths[signal_name] = [arr_len]
                    cumulative_step[signal_name] = [num]
                else:
                    cumulative_lengths[signal_name].append(cumulative_lengths[signal_name][-1] + arr_len)
                    cumulative_step[signal_name].append(cumulative_step[signal_name][-1] + num)
        return cumulative_lengths, cumulative_step

    def get_array_and_offset(self, step):
        # because for each user all signals have the same duration so iteration over signals is not needed here as i
        # use break to end the iteration after the first signal
        interval_dict = {}
        for signal, cumulative_step_list in self.cumulative_step.items():
            for idx, p2 in enumerate(cumulative_step_list):
                if step + 1 < p2:
                    pre_step = 0 if idx == 0 else cumulative_step_list[idx - 1]
                    start = (step - pre_step)*self.step_size
                    end = start + self.window_size - 1
                    return self.traj_list[idx], start, end
                elif step + 1 == p2:
                    pre_len = 0 if idx == 0 else self.cumulative_lengths[signal][idx - 1]
                    end = self.cumulative_lengths[signal][idx] - pre_len - 1
                    start = end - self.window_size + 1
                    return self.traj_list[idx], start, end

    def __getitem__(self, step):
        data = {}
        traj_idx, start, end = self.get_array_and_offset(step)
        for idx, signal_path in enumerate(self.signal_folder_path):
            traj_path = os.path.join(signal_path, str(traj_idx) + '.npy')
            # array = np.load(traj_path, mmap_mode='r')
            subset = np.load(traj_path, mmap_mode='r')
            subset = subset[start: end + 1]
            subset = torch.tensor(subset)
            data[self.signal_list[idx]] = subset
        return data

    def __len__(self):
        # because for each user all signals have the same duration so iteration over signals is not needed here as i
        # use break to end the iteration after the first signal
        num = 0
        for signal in self.signal_folder_path:
            for traj in self.traj_list:
                traj_path = os.path.join(signal, str(traj) + '.npy')
                arr_len = len(np.load(traj_path, mmap_mode='r'))
                num += (((arr_len - self.window_size) // self.step_size) + 1 + 1)
            break
        return num


def get_loaders(data_folder_path: str, window_size: int = 1000, step_size: int = 500, train: bool = True,
                batch_size: int = 1000, shuffle: bool = True, drop_last: bool = True):
    dataset = BautiroDrillingDataset(data_folder_path, window_size, step_size, train)
    return  DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


if __name__ == '__main__':
    from tqdm import tqdm
    project_path = os.path.abspath(os.path.join(__file__, '..', '..', '..'))
    data_folder_path = os.path.join(project_path, 'data/processed_drilling_data/separate_data/np_arr')

    # if not os.path.exists(data_folder_path):
    #     os.makedirs(data_folder_path)
    # for i in list(range(1, 14)) + list(range(17, 30)):
    #     for j in os.listdir(data_folder_path):
    #         arr = np.arange(i * 10, i * 10 + 10)
    #         np.save(os.path.join(data_folder_path, j, str(i) + '.npy'), arr)
    #
    # project_path = os.path.abspath(os.path.join(__file__, '..', '..', '..'))
    # data_folder_path = os.path.join(project_path, 'processed_drilling_data/separate_data/np_arr')
    # for i in list(range(1, 14)) + list(range(17, 30)):
    #     for j in os.listdir(data_folder_path):
    #         print(np.load(os.path.join(data_folder_path, j, str(i) + '.npy')))

    train_loader = get_loaders(data_folder_path, shuffle=True, window_size=10000, step_size=5000)
    for idx, data in tqdm(enumerate(train_loader)):
        for key, tensor in data.items():
            print(f'{key}: {tensor.shape}')
