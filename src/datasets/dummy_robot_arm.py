"""https://github.com/JunzheJosephZhu/see_hear_feel/tree/master/src/datasets"""
import copy
import json
import os
import random

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
from copy import deepcopy
from PIL import Image
from types import SimpleNamespace
from utils.pose_trajectory_processor import PoseTrajectoryProcessor, ProcessedPoseTrajectory
from utils.quaternion import q_log_map, q_exp_map
from omegaconf import OmegaConf, open_dict
from scipy.interpolate import splrep, BSpline
from utils.quaternion import compute_integral, compute_delta, recover_pose_from_relative_vel


def get_pose_sequence(resampled_trajectory, lb_idx):
    return resampled_trajectory["position"][lb_idx], resampled_trajectory["orientation"][lb_idx]


class DummyDataset(Dataset):
    def __init__(self, traj_path, args, train=True):
        super().__init__()
        """
        neg_ratio: ratio of silence audio clips to sample
        """
        self.traj_path = traj_path
        self.fix_cam_path = os.path.join(traj_path, "camera", "220322060186", "rgb")
        self.gripper_cam_path = os.path.join(traj_path, "camera", "838212074210", "rgb")
        self.pose_traj_processor = PoseTrajectoryProcessor()
        self.sr = 44100
        self.streams = [
            "cam_gripper_color",
            "cam_fixed_color",

        ]
        self.train = train
        self.num_stack = args.num_stack
        self.frameskip = args.frameskip
        self.len_obs = (self.num_stack - 1) * self.frameskip
        self.fps = 10
        self.resolution = (
                self.sr // self.fps
        )

        self.audio_len = self.num_stack * self.frameskip * self.resolution

        # augmentation parameters
        self.EPS = 1e-8

        self.resized_height_v = args.resized_height_v
        self.resized_width_v = args.resized_width_v

        self._crop_height_v = int(self.resized_height_v * (1.0 - args.crop_percent))
        self._crop_width_v = int(self.resized_width_v * (1.0 - args.crop_percent))

        (self.raw_source_trajectory,
         self.resample_source_trajectory,
         ) = self.get_episode(traj_path, ablation=args.ablation,
                              sampling_time=args.sampling_time,
                              json_name="source_robot_trajectory.json")

        (self.raw_target_trajectory,
         self.resample_target_trajectory,
         ) = self.get_episode(traj_path, ablation=args.ablation,
                              sampling_time=args.sampling_time,
                              json_name="target_robot_trajectory.json")

        assert len(self.resample_source_trajectory["relative_real_time_stamps"]) == len(
            self.resample_target_trajectory["relative_real_time_stamps"])
        self.num_frames = len(self.resample_source_trajectory["relative_real_time_stamps"])
        print(self.num_frames)

        self.modalities = args.ablation.split("_")
        self.no_crop = args.no_crop

        if self.train:
            self.transform_cam = [
                T.Resize((self.resized_height_v, self.resized_width_v), antialias=None),
                T.ColorJitter(brightness=0.2, contrast=0.02, saturation=0.02),
            ]
            self.transform_cam = T.Compose(self.transform_cam)

        else:
            self.transform_cam = T.Compose(
                [
                    T.Resize((self.resized_height_v, self.resized_width_v), antialias=None),
                    T.CenterCrop((self._crop_height_v, self._crop_width_v)),
                ]
            )
        self.sampling_time = args.sampling_time / 1000
        self.len_lb = args.len_lb
        # self.p_mean = args.p_mean
        # self.p_std = args.p_std
        # self.o_mean = args.o_mean
        # self.o_std = args.o_std

        pass

    def get_episode(self, traj_path, ablation="", sampling_time=150, json_name=None):
        """
        Return:
            folder for traj_path
            logs
            audio tracks
            number of frames in episode
        """
        modes = ablation.split("_")

        def load(file):
            fullpath = os.path.join(traj_path, file)
            if os.path.exists(fullpath):
                return sf.read(fullpath)[0]
            else:
                return None

        def omit_nan(traj):
            last_t = 0.0
            for idx, (i, v) in enumerate(traj.items()):
                current_t = v["time"]
                assert current_t > last_t, "need to reorder trajectory dict"
                if np.isnan(v["pose"][3:]).any():
                    traj[i]["pose"][3:] = last_o
                #     v["pose"][3:][np.isnan(v["pose"][3:])] = robot_trajectory.items()[idx-1] + robot_trajectory.items()[idx+1]
                if np.isnan(v["pose"][3:]).any():
                    print("nan detected")
                last_t = current_t
                last_o = v["pose"][3:]
            return traj

        if os.path.exists(os.path.join(traj_path, f"resampled_{sampling_time}_{json_name}")):
            with open(os.path.join(traj_path, f"{json_name}")) as ts:
                robot_trajectory = json.load(ts)
                robot_trajectory = omit_nan(robot_trajectory)
                pose_trajectory = self.pose_traj_processor.preprocess_trajectory(robot_trajectory)
            json_path = os.path.join(traj_path, f"resampled_{sampling_time}_{json_name}")
            resampled_trajectory = ProcessedPoseTrajectory.from_path(json_path)
        else:
            json_path = os.path.join(traj_path, json_name)
            with open(json_path) as ts:
                robot_trajectory = json.load(ts)
                robot_trajectory = omit_nan(robot_trajectory)
                pose_trajectory = self.pose_traj_processor.preprocess_trajectory(robot_trajectory)
                resampled_trajectory = self.pose_traj_processor.process_pose_trajectory(pose_trajectory,
                                                                                        sampling_time=sampling_time / 1000)
                resampled_trajectory.save_to_file(os.path.join(traj_path, f"{sampling_time}_{json_name}"))

        # get pseudo time stamp from resampled real time stamp and original real time stamps
        resample_time_step = np.expand_dims(resampled_trajectory.pose_trajectory.time_stamps, axis=1)
        og_time_step = np.expand_dims(np.array(pose_trajectory.time_stamps), axis=0)
        og_time_step = np.tile(og_time_step, (resample_time_step.shape[0], 1))
        diff = np.abs(og_time_step - resample_time_step)
        resample_pseudo_time_stamps = np.argmin(diff, axis=1, keepdims=False)

        # get all raw info
        raw_position_histroy = []
        raw_orientation_history = []
        raw_absolute_real_time_stamps = []

        last_t = 0.0
        for idx, (i, v) in enumerate(robot_trajectory.items()):
            current_t = v["time"]
            assert current_t > last_t, "need to reorder trajectory dict"

            if np.isnan(v["pose"][3:]).any():
                print("nan detected")
            raw_position_histroy.append(v["pose"][:3])
            raw_orientation_history.append(v["pose"][3:])
            raw_absolute_real_time_stamps.append(v["time"])
            last_t = current_t

        # get all resampled info
        resample_position_histroy = []
        resample_orientation_history = []
        for i in resampled_trajectory.pose_trajectory.poses:
            resample_orientation_history.append(
                [i.orientation.w, i.orientation.x, i.orientation.y, i.orientation.z])
            resample_position_histroy.append([i.position.x, i.position.y, i.position.z])

        # if np.isnan(np.array(raw_position_histroy)).any():
        #     print(f"nan detected in raw pos histroy {json_name}")
        # if np.isnan(np.array(raw_orientation_history)).any():
        #     print(f"nan detected in raw orienhistroy {json_name}")
        # if np.isnan(np.array(resample_position_histroy)).any():
        #     print(f"nan detected in resampled pos histroy {json_name}")
        # if np.isnan(np.array(resample_orientation_history)).any():
        #     print(f"nan detected in resampled ori histroy {json_name}")
        raw_traj = {
            "position": np.array(raw_position_histroy),
            "orientation": np.array(raw_orientation_history),
            "absolute_real_time_stamps": np.array(raw_absolute_real_time_stamps),
            "relative_real_time_stamps": np.array(pose_trajectory.time_stamps),
        }

        resample_traj = {
            "position": np.array(resample_position_histroy),
            "orientation": np.array(resample_orientation_history),
            "pseudo_time_stamps": resample_pseudo_time_stamps,
            "relative_real_time_stamps": resampled_trajectory.pose_trajectory.time_stamps,
        }
        return (
            raw_traj,
            resample_traj,
        )

    @staticmethod
    def load_image(rgb_path, idx):
        """
        Args:
            trial: the folder of the current episode
            stream: ["cam_gripper_color", "cam_fixed_color", "left_gelsight_frame"]
                for "left_gelsight_flow", please add another method to this class using torch.load("xxx.pt")
            timestep: the timestep of frame you want to extract
        """
        rgb_path = os.path.join(rgb_path, idx)
        image = Image.open(rgb_path).convert('RGB')
        image = np.array(image)
        # cv2.imshow("show", image)
        # while True:
        #     key = cv2.waitKey(1)
        #     if key == ord("q"):
        #         break
        # image.show()
        image = torch.as_tensor(image).float().permute(2, 0, 1) / 255
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
        return self.num_frames - 10

    @staticmethod
    def get_relative_delta_sequence(pos_seq: np.ndarray, quaternion_seq: np.ndarray) -> (np.ndarray, np.ndarray):
        pos_base = pos_seq[0:1, :]
        pos_seq = pos_seq - pos_base
        quaternion_seq = np.transpose(quaternion_seq, (1, 0))
        base = quaternion_seq[:, 0]
        delta_seq = np.transpose(q_log_map(quaternion_seq, base), (1, 0))
        return pos_seq, delta_seq

    @staticmethod
    def get_real_delta_sequence(pos_seq: np.ndarray, quaternion_seq: np.ndarray) -> (np.ndarray, np.ndarray):
        # if np.isnan(quaternion_seq).any():
        #     print("nan detected before")
        pos_base = np.concatenate((pos_seq[0:1], pos_seq[:-1]), axis=0)
        pos_delta_seq = pos_seq - pos_base
        o_delta_deq = np.zeros([quaternion_seq.shape[0], 3])
        quaternion_seq = np.transpose(quaternion_seq, (1, 0))
        o_base = np.concatenate((quaternion_seq[:, 0:1].copy(), quaternion_seq[:, :-1].copy()), axis=1)
        for i in range(o_delta_deq.shape[0]):
            o_delta_deq[i, :] = q_log_map(quaternion_seq[:, i], o_base[:, i])
        # if np.isnan(o_delta_deq).any():
        #     print("nan detected after")
        return pos_delta_seq, o_delta_deq

    def __getitem__(self, idx):
        start = idx - self.len_obs
        # compute which frames to use
        frame_idx = np.arange(start, idx + 1, self.frameskip)
        frame_idx[frame_idx < 0] = 0

        lb_end = idx + self.len_lb
        lb_idx = np.arange(start, lb_end)
        lb_idx[lb_idx >= self.num_frames] = -1

        lb_idx[lb_idx < 0] = 0
        pose_idx = copy.deepcopy(lb_idx)

        whole_source_position_seq, whole_source_orientation_seq = get_pose_sequence(self.resample_source_trajectory,
                                                                                    pose_idx)
        whole_target_position_seq, whole_target_orientation_seq = get_pose_sequence(self.resample_target_trajectory,
                                                                                    pose_idx)
        previous_position_seq, previous_orientation_seq = \
            whole_target_position_seq[:-self.len_lb, :], whole_target_orientation_seq[:-self.len_lb, :]
        future_position_seq, future_orientation_seq = \
            whole_source_position_seq[-self.len_lb - 1:, :], whole_source_orientation_seq[-self.len_lb - 1:, :]
        future_position_delta_seq, future_orientation_delta_seq = \
            self.get_real_delta_sequence(future_position_seq, future_orientation_seq)
        # target_position_delta_seq = ((target_position_delta_seq - self.p_mean) / self.p_std) * 100
        # target_orientation_delta_seq = ((target_orientation_delta_seq - self.o_mean) / self.o_std) * 100

        future_position_delta_seq = future_position_delta_seq * 10 / self.sampling_time  # cm/s
        future_orientation_delta_seq = future_orientation_delta_seq * 10 / self.sampling_time  # cm/s
        # 2i_images
        # to speed up data loading, do not load img if not using
        cam_gripper_framestack = 0
        cam_fixed_framestack = 0

        # process different streams of data
        if "vg" in self.modalities:
            cam_gripper_framestack = torch.stack(
                [
                    self.transform_cam(
                        self.load_image(self.gripper_cam_path, f"{timestep :06d}" + ".jpg")
                    )
                    for timestep in self.resample_target_trajectory["pseudo_time_stamps"][frame_idx]
                ],
                dim=0,
            )
        if "vf" in self.modalities:
            cam_fixed_framestack = torch.stack(
                [
                    self.transform_cam(
                        self.load_image(self.fix_cam_path, f"{timestep :06d}" + ".jpg")
                    )
                    for timestep in self.resample_target_trajectory["pseudo_time_stamps"][frame_idx]
                ],
                dim=0,
            )

        # random cropping
        if self.train:  # get random crop params (previously resize and color jitter)
            img = self.transform_cam(
                self.load_image(self.gripper_cam_path, f"{idx:06d}" + ".jpg")
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

        return {
            "observation": (
                cam_fixed_framestack,
                cam_gripper_framestack,
            ),
            "start": start,
            "current": idx,
            "end": lb_end,
            "previous_pose_seq": torch.from_numpy(
                np.concatenate([previous_position_seq, previous_orientation_seq], axis=-1)).float(),
            "future_pose_seq": torch.from_numpy(
                np.concatenate([future_position_seq, future_orientation_seq], axis=-1)).float(),
            "future_delta_seq": torch.from_numpy(
                np.concatenate([future_position_delta_seq, future_orientation_delta_seq], axis=-1)).float()
            ,
        }


# class Normalizer(torch.nn.Module):
#     def __init__(self, traj_paths, args, train=True):
#         super().__init__()
#         """
#         neg_ratio: ratio of silence audio clips to sample
#         """
#         # self.fix_cam_path = os.path.join(traj_paths, "camera", "220322060186", "rgb")
#         # self.gripper_cam_path = os.path.join(traj_paths, "camera", "838212074210", "rgb")
#         self.pose_traj_processor = PoseTrajectoryProcessor()
#         self.sr = 44100
#         self.streams = [
#             "cam_gripper_color",
#             "cam_fixed_color",
#
#         ]
#         self.train = train
#         self.num_stack = args.num_stack
#         self.frameskip = args.frameskip
#         self.len_obs = (self.num_stack - 1) * self.frameskip
#         self.fps = 10
#         self.resolution = (
#                 self.sr // self.fps
#         )
#
#         self.audio_len = self.num_stack * self.frameskip * self.resolution
#
#         # augmentation parameters
#         self.EPS = 1e-8
#
#         self.resized_height_v = args.resized_height_v
#         self.resized_width_v = args.resized_width_v
#
#         self._crop_height_v = int(self.resized_height_v * (1.0 - args.crop_percent))
#         self._crop_width_v = int(self.resized_width_v * (1.0 - args.crop_percent))
#
#         all_poses = []
#         for traj in traj_paths:
#             (raw_trajectory,
#              resample_trajectory,
#              ) = self.get_episode(traj, ablation=args.ablation, sampling_time=args.sampling_time)
#             all_poses.append(resample_trajectory)
#         self.all_poses = all_poses
#
#         self.modalities = args.ablation.split("_")
#         self.len_lb = args.len_lb
#
#     def get_episode(self, traj_path, ablation="", sampling_time=150, ):
#         """
#         Return:
#             folder for traj_path
#             logs
#             audio tracks
#             number of frames in episode
#         """
#
#         if os.path.exists(os.path.join(traj_path, f"resampled_{sampling_time}_robot_trajectory.json")):
#             print(f"resampled trajectory exists")
#             with open(os.path.join(traj_path, "robot_trajectory.json")) as ts:
#                 robot_trajectory = json.load(ts)
#                 pose_trajectory = self.pose_traj_processor.preprocess_trajectory(robot_trajectory)
#             json_path = os.path.join(traj_path, f"resampled_{sampling_time}_robot_trajectory.json")
#             resampled_trajectory = ProcessedPoseTrajectory.from_path(json_path)
#         else:
#             print(f"resampled trajectory not exists")
#             json_path = os.path.join(traj_path, "robot_trajectory.json")
#             with open(json_path) as ts:
#                 robot_trajectory = json.load(ts)
#                 pose_trajectory = self.pose_traj_processor.preprocess_trajectory(robot_trajectory)
#                 resampled_trajectory = self.pose_traj_processor.process_pose_trajectory(pose_trajectory,
#                                                                                         sampling_time=sampling_time / 1000)
#                 resampled_trajectory.save_to_file(os.path.join(traj_path, f"{sampling_time}_robot_trajectory.json"))
#
#         # get pseudo time stamp from resampled real time stamp and original real time stamps
#         resample_time_step = np.expand_dims(resampled_trajectory.pose_trajectory.time_stamps, axis=1)
#         og_time_step = np.expand_dims(np.array(pose_trajectory.time_stamps), axis=0)
#         og_time_step = np.tile(og_time_step, (resample_time_step.shape[0], 1))
#         diff = np.abs(og_time_step - resample_time_step)
#         resample_pseudo_time_stamps = np.argmin(diff, axis=1, keepdims=False)
#
#         # get all raw info
#         raw_position_histroy = []
#         raw_orientation_history = []
#         raw_absolute_real_time_stamps = []
#         for i, v in robot_trajectory.items():
#             raw_position_histroy.append(v["pose"][:3])
#             raw_orientation_history.append(v["pose"][3:])
#             raw_absolute_real_time_stamps.append(v["time"])
#
#         # get all resampled info
#         resample_position_histroy = []
#         resample_orientation_history = []
#         for i in resampled_trajectory.pose_trajectory.poses:
#             resample_orientation_history.append(
#                 [i.orientation.w, i.orientation.x, i.orientation.y, i.orientation.z])
#             resample_position_histroy.append([i.position.x, i.position.y, i.position.z])
#
#         raw_traj = {
#             "position": np.array(raw_position_histroy),
#             "orientation": np.array(raw_orientation_history),
#             "absolute_real_time_stamps": np.array(raw_absolute_real_time_stamps),
#             "relative_real_time_stamps": np.array(pose_trajectory.time_stamps),
#         }
#         resample_traj = {
#             "position": np.array(resample_position_histroy),
#             "orientation": np.array(resample_orientation_history),
#             "pseudo_time_stamps": resample_pseudo_time_stamps,
#             "relative_real_time_stamps": resampled_trajectory.pose_trajectory.time_stamps,
#         }
#         return (
#             raw_traj,
#             resample_traj,
#         )
#
#     def __len__(self):
#         return self.num_frames
#
#     def get_pose_sequence(self, p, o, lb_idx):
#         return p[lb_idx], o[lb_idx]
#
#     @staticmethod
#     def get_delta_sequence(pos_seq: np.ndarray, quaternion_seq: np.ndarray) -> (np.ndarray, np.ndarray):
#         pos_base = pos_seq[0:1, :]
#         pos_seq = pos_seq - pos_base
#         quaternion_seq = np.transpose(quaternion_seq, (1, 0))
#         base = quaternion_seq[:, 0]
#         delta_seq = np.transpose(q_log_map(quaternion_seq, base), (1, 0))
#         return pos_seq, delta_seq
#
#     def get_mean_std(self):
#
#         all_delta_p = []
#         all_delta_o = []
#         for i in self.all_poses:
#             p = i["position"]
#             o = i["orientation"]
#             for idx in range(p.shape[0]):
#                 start = idx - self.len_obs
#                 lb_end = idx + self.len_lb
#                 lb_idx = np.arange(start, lb_end)
#                 lb_idx[lb_idx >= p.shape[0]] = -1
#
#                 lb_idx[lb_idx < 0] = 0
#                 pose_idx = copy.deepcopy(lb_idx)
#                 whole_position_seq, whole_orientation_seq = self.get_pose_sequence(p, o, pose_idx)
#                 target_position_seq, target_orientation_seq = \
#                     whole_position_seq[-self.len_lb - 1:, :], whole_orientation_seq[-self.len_lb - 1:, :]
#                 target_position_delta_seq, target_orientation_delta_seq = \
#                     self.get_delta_sequence(target_position_seq, target_orientation_seq)
#                 all_delta_p.append(target_position_delta_seq[1:])
#                 all_delta_o.append(target_orientation_delta_seq[1:])
#         all_delta_p = np.concatenate(all_delta_p, axis=0)
#         p_mean, p_std = all_delta_p.mean(0, keepdims=True), all_delta_p.std(0, keepdims=True)
#         all_delta_o = np.concatenate(all_delta_o, axis=0)
#         o_mean, o_std = all_delta_o.mean(0, keepdims=True), all_delta_o.std(0, keepdims=True)
#         print(f"p_mean:{p_mean}")
#         print(f"p_std:{p_std}")
#         print(f"o_mean{o_mean}")
#         print(f"o_std:{o_std}")
#         return p_mean, p_std, o_mean, o_std


def get_loaders(batch_size: int, args, data_folder: str, drop_last: bool, **kwargs):
    """

    Args:
        batch_size: batch size
        args: arguments for dataloader
        data_folder: absolute path of directory "data"
        drop_last: whether drop_last for train dataloader
        **kwargs: other arguments

    Returns: training loader and validation loader

    """
    trajs = [os.path.join(data_folder, traj) for traj in sorted(os.listdir(data_folder))]
    num_train = int(len(trajs) * 0.9)

    train_trajs_paths = trajs[:num_train]
    print(f"number of training trajectories: {len(train_trajs_paths)}")
    val_trajs_paths = trajs[num_train:]
    print(f"number of validation trajectories: {len(val_trajs_paths)}")

    # normalizer = Normalizer(train_trajs_paths, args, True)
    # args.p_mean, args.p_std, args.o_mean, args.o_std = normalizer.get_mean_std()

    train_set = torch.utils.data.ConcatDataset(
        [
            DummyDataset(traj, args, )
            for i, traj in enumerate(train_trajs_paths)
        ]
    )
    val_set = torch.utils.data.ConcatDataset(
        [
            DummyDataset(traj, args, False)
            for i, traj in enumerate(val_trajs_paths)
        ]
    )

    train_loader = DataLoader(train_set, batch_size, num_workers=8, shuffle=True, drop_last=drop_last, )
    val_loader = DataLoader(val_set, 1, num_workers=8, shuffle=False, drop_last=False, )
    return train_loader, val_loader, None


def get_debug_loaders(batch_size: int, args, data_folder: str, **kwargs):
    """

    Args:
        batch_size: batch size
        args: arguments for dataloader
        data_folder: absolute path of directory "data"11
        drop_last: whether drop_last for train dataloader
        **kwargs: other arguments

    Returns: training loader and validation loader

    """
    trajs = [os.path.join(data_folder, traj) for traj in sorted(os.listdir(data_folder))]
    num_train = int(len(trajs) * 0.9)

    train_trajs_paths = trajs[:num_train]
    print(f"number of training trajectories: {len(train_trajs_paths)}")
    val_trajs_paths = trajs[num_train:]
    print(f"number of validation trajectories: {len(val_trajs_paths)}")

    train_trajs_paths = train_trajs_paths[4:5]
    val_trajs_paths = val_trajs_paths[0:1]

    args = SimpleNamespace(**args) if not isinstance(args, SimpleNamespace) else args
    args.p_mean = np.array([[0.00010239, 0.0003605, -0.00236236]])
    args.p_std = np.array([[0.05333986, 0.07098175, 0.11525309]])
    args.o_mean = np.array([[0.00295716, -0.0009488, 0.000222]])
    args.o_std = np.array([[0.05337297, 0.04242631, 0.21439145]])

    # mean[0.12096352  0.12888882 - 0.30349103  0.00160154 - 0.06071072 - 0.04832873]
    # std[0.4358776  0.5022668  0.50152147 0.6001411  0.7150956  1.1324104]
    # max[2.3246083 2.104693  2.135981  2.5701568 4.644128  5.43282]
    # min[-3.3144479 - 1.9043726 - 1.7756243 - 3.3247855 - 7.9488 - 4.817799]

    train_set = torch.utils.data.ConcatDataset(
        [
            DummyDataset(traj, args, )
            for i, traj in enumerate(train_trajs_paths)
        ]
    )

    train_inference_set = torch.utils.data.ConcatDataset(
        [
            DummyDataset(traj, args, False)
            for i, traj in enumerate(train_trajs_paths)
        ]
    )

    val_set = torch.utils.data.ConcatDataset(
        [
            DummyDataset(traj, args, False)
            for i, traj in enumerate(val_trajs_paths)
        ]
    )

    train_loader = DataLoader(train_set, 1, num_workers=8, shuffle=False, drop_last=True, )
    train_inference_loader = DataLoader(train_inference_set, 1, num_workers=8, shuffle=False, drop_last=False, )
    val_loader = DataLoader(val_set, 1, num_workers=8, shuffle=False, drop_last=False, )
    return train_loader, val_loader, train_inference_loader


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    import cv2

    data_folder_path = '/fs/scratch/rb_bd_dlp_rng-dl01_cr_ROB_employees/students/jin4rng/data/robodemo_3_20'
    args = SimpleNamespace()

    args.ablation = 'vg'
    args.num_stack = 5
    args.frameskip = 5
    args.no_crop = True
    args.crop_percent = 0.0
    args.resized_height_v = 120
    args.resized_width_v = 160
    args.len_lb = 10
    args.sampling_time = 150
    all_step_delta = []
    all_step_pose = []
    all_recover_pose = []
    train_loader, val_loader, _ = get_debug_loaders(batch_size=1, args=args, data_folder=data_folder_path,
                                                    drop_last=True)
    print(len(train_loader))


    # for idx, batch in enumerate(train_loader):
    #     # if idx >= 100:
    #     #     break
    #     print(f"{idx} \n")
    #     obs = batch["observation"]
    #
    #     image_g = obs[1][0][-1].permute(1, 2, 0).numpy()
    #     cv2.imshow("asdf", image_g)
    #     time.sleep(0.2)
    #     key = cv2.waitKey(1)
    #     if key == ord("q"):
    #         break
    #
    #     all_step_delta.append(batch["future_delta_seq"][:, 1])
    #     all_step_pose.append(batch["future_pose_seq"][:, 1])
    #
    # all_step = torch.concatenate(all_step_delta, dim=0)
    # pm = all_step.detach().cpu().numpy()
    # print(np.mean(pm, axis=0))
    # print(np.std(pm, axis=0))
    # print(np.max(pm, axis=0))
    # print(np.min(pm, axis=0))
    # t = np.arange(all_step.shape[0])
    # plt.figure()
    # plt.subplot(711)
    # tck = splrep(t, pm[:, :1], s=0.1)
    # o = np.stack([pm[:, 0], BSpline(*tck)(t)], axis=1)
    # plt.plot(t, o, '-', )
    # plt.subplot(712)
    # plt.plot(t, pm[:, 1:2], '-')
    # # plt.plot(tfwd, pmfwd[:, 3:], 'd-')
    # plt.subplot(713)
    # plt.plot(t, pm[:, 2:3], '-')
    # plt.subplot(714)
    # plt.plot(t, pm[:, 3:4], '-')
    # plt.subplot(715)
    # plt.plot(t, pm[:, 4:5], '-')
    # plt.subplot(716)
    # plt.plot(t, pm[:, 5:6], '-')
    # plt.subplot(717)
    # plt.plot(t, pm[:, 6:7], '-')
    # plt.show()




    for idx, batch in enumerate(train_loader):
        if idx % args.len_lb != 0:
            continue
        # if idx >= 100:
        #     break
        print(f"{idx} \n")
        obs = batch["observation"]

        # for image_g in obs[1][0]:
        #     cv2.imshow("asdf", image_g.permute(1, 2, 0).numpy())
        # key = cv2.waitKey(1)
        # if key == ord("q"):
        #     break

        all_step_delta.append(batch["future_delta_seq"][0, 1:, :])
        all_step_pose.append(batch["future_pose_seq"][0, 1:, :])

        # recover_pose = recover_pose_from_relative_vel(batch["future_delta_seq"][0, 1:, :].detach().cpu().numpy(),
        #                                               batch["previous_pose_seq"][0, -1, :].detach().cpu().numpy(),
        #                                               vel_scale=0.015)
        # all_recover_pose.append(torch.from_numpy(recover_pose))

    all_step = torch.concatenate(all_step_pose, dim=0)
    pm = all_step.detach().cpu().numpy()
    # all_recover_pose = torch.concatenate(all_recover_pose, dim=0)
    # pmr = all_recover_pose.detach().cpu().numpy()
    pmr = torch.cat(all_step_delta, dim = 0).detach().cpu().numpy()
    print(np.mean(pm, axis=0))
    print(np.std(pm, axis=0))
    print(np.max(pm, axis=0))
    print(np.min(pm, axis=0))
    t = np.arange(all_step.shape[0])
    plt.figure()
    plt.subplot(711)
    o = np.stack([pm[:, 0], pmr[:, 0]], axis=1)
    plt.plot(t, o, '-', )
    plt.subplot(712)
    o = np.stack([pm[:, 1], pmr[:, 1]], axis=1)
    plt.plot(t, o, '-')
    plt.subplot(713)
    o = np.stack([pm[:, 2], pmr[:, 2]], axis=1)
    plt.plot(t, o, '-')
    plt.subplot(714)
    o = np.stack([pm[:, 3], pmr[:, 3]], axis=1)
    plt.plot(t, o, '-')
    plt.subplot(715)
    o = np.stack([pm[:, 4], pmr[:, 4]], axis=1)
    plt.plot(t, o, '-')
    plt.subplot(716)
    o = np.stack([pm[:, 5], pmr[:, 5]], axis=1)
    plt.plot(t, o, '-')
    # plt.subplot(717)
    # o = np.stack([pm[:, 6], pmr[:, 6]], axis=1)
    # plt.plot(t, o, '-')
    plt.show()
