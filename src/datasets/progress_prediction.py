"""https://github.com/JunzheJosephZhu/see_hear_feel/tree/master/src/datasets"""
import json
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import torchaudio
import soundfile as sf
import os
import torch


class EpisodeDataset(Dataset):
    def __init__(self, log_file, data_folder=None):
        """
        log_file - the path of log file(.csv)
        data_folder - the path of data folder
        """
        super().__init__()
        self.logs = pd.read_csv(log_file)
        self.data_folder = data_folder
        self.sr = 44100
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=int(self.sr * 0.025),
            hop_length=int(self.sr * 0.01),
            n_mels=64,
            center=False,
        )
        pass

    def get_episode(self, idx):
        """
        Return:
            folder for trial
            logs
            audio tracks
            total number of frames in episode
        """

        def load(file):
            fullpath = os.path.join(trial, file)
            if os.path.exists(fullpath):
                return sf.read(fullpath)[0]
            else:
                return None

        format_time = self.logs.iloc[idx].Time.replace(":", "_")
        trial = os.path.join(self.data_folder, format_time)
        with open(os.path.join(trial, "timestamps.json")) as ts:
            timestamps = json.load(ts)

        audio_holebase_left = load("audio_holebase_left.wav")
        audio_holebase_right = load("audio_holebase_right.wav")
        audio_holebase = [x for x in [audio_holebase_left, audio_holebase_right] if x is not None]
        audio_holebase = torch.as_tensor(np.stack(audio_holebase, 0))

        return (
            trial,
            timestamps,
            audio_holebase,
            len(timestamps["action_history"]),
        )

    def __getitem__(self, idx):
        raise NotImplementedError

    @staticmethod
    def clip_resample(audio, audio_start, audio_end):
        left_pad, right_pad = torch.Tensor([]), torch.Tensor([])
        if audio_start < 0:
            left_pad = torch.zeros((audio.shape[0], -audio_start))
            audio_start = 0
        if audio_end >= audio.size(-1):
            right_pad = torch.zeros((audio.shape[0], audio_end - audio.size(-1)))
            audio_end = audio.size(-1)
        audio_clip = torch.cat(
            [left_pad, audio[:, audio_start:audio_end], right_pad], dim=1
        )
        audio_clip = torchaudio.functional.resample(audio_clip, 44100, 16000)
        return audio_clip

    def __len__(self):
        return len(self.logs)


class ImitationEpisode(EpisodeDataset):
    def __init__(self, log_file, args, dataset_idx, data_folder, train=True):
        super().__init__(log_file, data_folder)
        self.train = train
        self.num_stack = args.num_stack
        self.frameskip = args.frameskip
        self.max_len = (self.num_stack - 1) * self.frameskip
        self.fps = 10
        self.sr = 44100
        self.resolution = (
                self.sr // self.fps
        )
        self.audio_len = self.num_stack * self.frameskip * self.resolution

        # augmentation parameters
        self.EPS = 1e-8
        (self.trial,
         self.timestamps,
         self.audio_holebase,
         self.num_frames,
         ) = self.get_episode(dataset_idx)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        start = idx - self.max_len
        # compute which frames to use
        frame_idx = np.arange(start, idx + 1, self.frameskip)
        frame_idx[frame_idx < 0] = -1
        # load audio
        audio_end = idx * self.resolution
        audio_start = audio_end - self.audio_len  # why self.sr // 2, and start + sr
        audio_clip_h = self.clip_resample(self.audio_holebase, audio_start, audio_end).float()
        # load labels
        label = int(10 * (idx / self.__len__()))
        return audio_clip_h, label


def get_loaders(batch_size: int, train_csv: str, val_csv: str, args, data_folder: str, project_path: str, **kwargs):
    train_csv = os.path.join(project_path, train_csv)
    val_csv = os.path.join(project_path, val_csv)
    data_folder = os.path.join(project_path, data_folder)
    train_num_episode = len(pd.read_csv(train_csv))
    val_num_episode = len(pd.read_csv(val_csv))
    train_set = torch.utils.data.ConcatDataset(
        [
            ImitationEpisode(train_csv, args, i, data_folder)
            for i in range(train_num_episode)
        ]
    )
    val_set = torch.utils.data.ConcatDataset(
        [
            ImitationEpisode(val_csv, args, i, data_folder, False)
            for i in range(val_num_episode)
        ]
    )

    train_loader = DataLoader(
        train_set, batch_size, num_workers=8,
    )
    val_loader = DataLoader(val_set, 1, num_workers=8, shuffle=False)
    return train_loader, val_loader, None
