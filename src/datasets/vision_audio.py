"""https://github.com/JunzheJosephZhu/see_hear_feel/tree/master/src/datasets"""
import json
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import torchaudio
import soundfile as sf

import os
import torch
import torchvision.transforms as T

from PIL import Image


class VisionAudioTactile(Dataset):
    def __init__(self, log_file, args, dataset_idx, data_folder, train=True):
        super().__init__()
        """
        neg_ratio: ratio of silence audio clips to sample
        """
        self.logs = pd.read_csv(log_file)
        self.data_folder = os.path.join(data_folder, 'test_recordings',)
        self.sr = 44100
        self.streams = [
            "cam_gripper_color",
            "cam_fixed_color",
            "left_gelsight_flow",
            "left_gelsight_frame",
        ]
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
        self.resized_height_v = args.resized_height_v
        self.resized_width_v = args.resized_width_v
        self.resized_height_t = args.resized_height_t
        self.resized_width_t = args.resized_width_t
        self._crop_height_v = int(self.resized_height_v * (1.0 - args.crop_percent))
        self._crop_width_v = int(self.resized_width_v * (1.0 - args.crop_percent))
        self._crop_height_t = int(self.resized_height_t * (1.0 - args.crop_percent))
        self._crop_width_t = int(self.resized_width_t * (1.0 - args.crop_percent))

        (self.trial,
         self.timestamps,
         self.audio_gripper,
         self.audio_holebase,
         self.num_frames,) = self.get_episode(dataset_idx, ablation=args.ablation)

        self.action_dim = args.action_dim
        self.task = args.task
        self.use_flow = args.use_flow
        self.modalities = args.ablation.split("_")
        self.no_crop = args.no_crop

        if self.train:
            self.transform_cam = [
                T.Resize((self.resized_height_v, self.resized_width_v), antialias=None),
                T.ColorJitter(brightness=0.2, contrast=0.02, saturation=0.02),
            ]
            self.transform_gel = [
                T.Resize((self.resized_height_t, self.resized_width_t), antialias=None),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]
            self.transform_cam = T.Compose(self.transform_cam)
            self.transform_gel = T.Compose(self.transform_gel)

        else:
            self.transform_cam = T.Compose(
                [
                    T.Resize((self.resized_height_v, self.resized_width_v), antialias=None),
                    T.CenterCrop((self._crop_height_v, self._crop_width_v)),
                ]
            )
            self.transform_gel = T.Compose(
                [
                    T.Resize((self.resized_height_t, self.resized_width_t), antialias=None),
                    T.CenterCrop((self._crop_height_t, self._crop_width_t)),
                ]
            )

        pass

    def get_episode(self, idx, ablation=""):
        """
        Return:
            folder for trial
            logs
            audio tracks
            number of frames in episode
        """
        modes = ablation.split("_")

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
        if "ag" in modes:
            audio_gripper_left = load("audio_gripper_left.wav")
            audio_gripper_right = load("audio_gripper_right.wav")
            audio_gripper = [
                x for x in [audio_gripper_left, audio_gripper_right] if x is not None
            ]
            audio_gripper = torch.as_tensor(np.stack(audio_gripper, 0))
        else:
            audio_gripper = None
        if "ah" in modes:
            audio_holebase_left = load("audio_holebase_left.wav")
            audio_holebase_right = load("audio_holebase_right.wav")
            audio_holebase = [
                x for x in [audio_holebase_left, audio_holebase_right] if x is not None
            ]
            audio_holebase = torch.as_tensor(np.stack(audio_holebase, 0))
        else:
            audio_holebase = None

        return (
            trial,
            timestamps,
            audio_gripper,
            audio_holebase,
            len(timestamps["action_history"]),
        )

    @staticmethod
    def load_image(trial, stream, timestep):
        """
        Args:
            trial: the folder of the current episode
            stream: ["cam_gripper_color", "cam_fixed_color", "left_gelsight_frame"]
                for "left_gelsight_flow", please add another method to this class using torch.load("xxx.pt")
            timestep: the timestep of frame you want to extract
        """
        if timestep == -1:
            timestep = 0
        img_path = os.path.join(trial, stream, str(timestep) + ".png")
        image = (
                torch.as_tensor(np.array(Image.open(img_path))).float().permute(2, 0, 1)
                / 255
        )
        return image

    @staticmethod
    def load_flow(trial, stream, timestep):
        """
        Args:
            trial: the folder of the current episode
            stream: ["cam_gripper_color", "cam_fixed_color", "left_gelsight_frame"]
                for "left_gelsight_flow", please add another method to this class using torch.load("xxx.pt")
            timestep: the timestep of frame you want to extract
        """
        img_path = os.path.join(trial, stream, str(timestep) + ".pt")
        image = torch.as_tensor(torch.load(img_path))
        return image

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

    @staticmethod
    def resize_image(image, size):
        assert len(image.size()) == 3  # [3, H, W]
        return torch.nn.functional.interpolate(
            image.unsqueeze(0), size=size, mode="bilinear"
        ).squeeze(0)

    def __len__(self):
        return self.num_frames

    def get_demo(self, idx):
        keyboard = self.timestamps["action_history"][idx]
        if self.task == "pouring":
            x_space = {-0.0005: 0, 0: 1, 0.0005: 2}
            dy_space = {-0.0012: 0, 0: 1, 0.004: 2}
            keyboard = x_space[keyboard[0]] * 3 + dy_space[keyboard[4]]
        else:
            x_space = {-0.0005: 0, 0: 1, 0.0005: 2}
            y_space = {-0.0005: 0, 0: 1, 0.0005: 2}
            z_space = {-0.0005: 0, 0: 1, 0.0005: 2}

            keyboard = (  # ternary
                    x_space[keyboard[0]] * 3**2
                    + y_space[keyboard[1]] * 3**1
                    + z_space[keyboard[2]] * 3**0
            )
        return keyboard

    def __getitem__(self, idx):
        start = idx - self.max_len
        # compute which frames to use
        frame_idx = np.arange(start, idx + 1, self.frameskip)
        frame_idx[frame_idx < 0] = -1
        # 2i_images
        # to speed up data loading, do not load img if not using
        cam_gripper_framestack = 0
        cam_fixed_framestack = 0
        tactile_framestack = 0
        label = int(10 * (idx / self.__len__()))

        # load first frame for better alignment
        if self.use_flow:
            offset = torch.from_numpy(
                torch.load(
                    os.path.join(self.trial, "left_gelsight_flow", str(0) + ".pt")
                )
            ).type(torch.FloatTensor)
        else:
            offset = self.load_image(self.trial, "left_gelsight_frame", 0)

        # process different streams of data
        if "vg" in self.modalities:
            cam_gripper_framestack = torch.stack(
                [
                    self.transform_cam(
                        self.load_image(self.trial, "cam_gripper_color", timestep)
                    )
                    for timestep in frame_idx
                ],
                dim=0,
            )
        if "vf" in self.modalities:
            cam_fixed_framestack = torch.stack(
                [
                    self.transform_cam(
                        self.load_image(self.trial, "cam_fixed_color", timestep)
                    )
                    for timestep in frame_idx
                ],
                dim=0,
            )
        if "t" in self.modalities:
            if self.use_flow:
                tactile_framestack = torch.stack(
                    [
                        torch.from_numpy(
                            torch.load(
                                os.path.join(
                                    self.trial,
                                    "left_gelsight_flow",
                                    str(timestep) + ".pt",
                                )
                            )
                        ).type(torch.FloatTensor)
                        - offset
                        for timestep in frame_idx
                    ],
                    dim=0,
                )
            else:
                tactile_framestack = torch.stack(
                    [
                        (
                            self.transform_gel(
                                self.load_image(
                                    self.trial, "left_gelsight_frame", timestep
                                )
                                - offset
                                + 0.5
                            ).clamp(0, 1)
                        )
                        for timestep in frame_idx
                    ],
                    dim=0,
                )
                for i, timestep in enumerate(frame_idx):
                    if timestep < 0:
                        tactile_framestack[i] = torch.zeros_like(tactile_framestack[i])

        # random cropping
        if self.train:
            img = self.transform_cam(
                self.load_image(self.trial, "cam_fixed_color", idx)
            )
            if not self.no_crop:
                i_v, j_v, h_v, w_v = T.RandomCrop.get_params(
                    img, output_size=(self._crop_height_v, self._crop_width_v)
                )
            else:
                i_v, h_v = (
                                   self.resized_height_v - self._crop_height_v
                           ) // 2, self._crop_height_v
                j_v, w_v = (
                                   self.resized_width_v - self._crop_width_v
                           ) // 2, self._crop_width_v

            if "vg" in self.modalities:
                cam_gripper_framestack = cam_gripper_framestack[
                                         ..., i_v: i_v + h_v, j_v: j_v + w_v
                                         ]
            if "vf" in self.modalities:
                cam_fixed_framestack = cam_fixed_framestack[
                                       ..., i_v: i_v + h_v, j_v: j_v + w_v
                                       ]
            if "t" in self.modalities:
                if not self.use_flow:
                    img_t = self.transform_gel(
                        self.load_image(self.trial, "left_gelsight_frame", idx)
                    )
                    if not self.no_crop:
                        i_t, j_t, h_t, w_t = T.RandomCrop.get_params(
                            img_t, output_size=(self._crop_height_t, self._crop_width_t)
                        )
                    else:
                        i_t, h_t = (
                                           self.resized_height_t - self._crop_height_t
                                   ) // 2, self._crop_height_t
                        j_t, w_t = (
                                           self.resized_width_t - self._crop_width_t
                                   ) // 2, self._crop_width_t
                    tactile_framestack = tactile_framestack[
                                         ..., i_t: i_t + h_t, j_t: j_t + w_t
                                         ]

        # load audio
        audio_end = idx * self.resolution
        audio_start = audio_end - self.audio_len  # why self.sr // 2, and start + sr
        if self.audio_gripper is not None:
            audio_clip_g = self.clip_resample(
                self.audio_gripper, audio_start, audio_end
            ).float()
        else:
            audio_clip_g = 0
        if self.audio_holebase is not None:
            audio_clip_h = self.clip_resample(
                self.audio_holebase, audio_start, audio_end
            ).float()
        else:
            audio_clip_h = 0

        # load labels
        keyboard = self.get_demo(idx)
        xyzrpy = torch.Tensor(self.timestamps["pose_history"][idx][:6])
        optical_flow = 0

        return (
            (
                cam_fixed_framestack,
                cam_gripper_framestack,
                tactile_framestack,
                audio_clip_g,
                audio_clip_h,
            ),
            keyboard,
            xyzrpy,
            optical_flow,
            start,
            label,
        )


def get_loaders(batch_size: int, args, data_folder: str, **kwargs):
    """

    Args:
        batch_size: batch size
        args: arguments for dataloader
        data_folder: absolute path of directory "data"
        **kwargs: other arguments

    Returns: training loader and validation loader

    """
    train_csv = os.path.join(data_folder, args.train_csv)
    val_csv = os.path.join(data_folder, args.val_csv)
    train_num_episode = len(pd.read_csv(train_csv))
    val_num_episode = len(pd.read_csv(val_csv))
    train_set = torch.utils.data.ConcatDataset(
        [
            VisionAudioTactile(train_csv, args, i, data_folder)
            for i in range(train_num_episode)
        ]
    )
    val_set = torch.utils.data.ConcatDataset(
        [
            VisionAudioTactile(val_csv, args, i, data_folder, False)
            for i in range(val_num_episode)
        ]
    )

    train_loader = DataLoader(
        train_set, batch_size, num_workers=8, shuffle=True,
    )
    val_loader = DataLoader(val_set, 1, num_workers=8, shuffle=False)
    return train_loader, val_loader, None
