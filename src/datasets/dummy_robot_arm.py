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

from utils.quaternion import smooth_traj


def get_pose_sequence(resampled_trajectory, lb_idx):
    return resampled_trajectory["pos_quat"][lb_idx]


class DummyDataset(Dataset):
    def __init__(self, traj_path, args, train=True, norm=False):
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
         self.smooth_resample_source_trajectory
         ) = self.get_episode(traj_path, ablation=args.ablation,
                              sampling_time=args.sampling_time,
                              json_name="source_robot_trajectory.json")

        (self.raw_target_trajectory,
         self.resample_target_trajectory,
         self.smooth_resample_target_trajectory
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

        if norm:
            self.p_mean = args.p_mean
            self.p_std = args.p_std
            self.o_mean = args.o_mean
            self.o_std = args.o_std
            self.smooth_p_mean = args.smooth_p_mean
            self.smooth_p_std = args.smooth_p_std
            self.smooth_o_mean = args.smooth_o_mean
            self.smooth_o_std = args.smooth_o_std
        else:
            self.p_mean = 0
            self.p_std = 1
            self.o_mean = 0
            self.o_std = 1
            self.smooth_p_mean = 0
            self.smooth_p_std = 1
            self.smooth_o_mean = 0
            self.smooth_o_std = 1

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
        smoothed_traj, global_delta, smoothed_global_delta = smooth_traj(
            np.concatenate([np.array(resample_position_histroy), np.array(resample_orientation_history)], axis=1),
            (5e-4, 5e-4, 5e-4, 2e-4, 2e-4, 2e-4))

        raw_traj = {
            "pos_quat": np.concatenate([np.array(raw_position_histroy), np.array(raw_orientation_history)], axis=1),
            "absolute_real_time_stamps": np.array(raw_absolute_real_time_stamps),
            "relative_real_time_stamps": np.array(pose_trajectory.time_stamps),
        }

        resample_traj = {
            "pos_quat": np.concatenate([np.array(resample_position_histroy), np.array(resample_orientation_history)],
                                       axis=1),
            "glb_pos_ori": global_delta,
            "pseudo_time_stamps": resample_pseudo_time_stamps,
            "relative_real_time_stamps": resampled_trajectory.pose_trajectory.time_stamps,
        }

        smooth_resample_traj = {
            "pos_quat": np.concatenate([smoothed_traj[:, :3], smoothed_traj[:, 3:]], axis=1),
            "glb_pos_ori": smoothed_global_delta,
            "pseudo_time_stamps": resample_pseudo_time_stamps,
            "relative_real_time_stamps": resampled_trajectory.pose_trajectory.time_stamps,
        }
        return (
            raw_traj,
            resample_traj,
            smooth_resample_traj
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
        return (self.num_frames - self.len_lb)

    @staticmethod
    def get_relative_delta_sequence(pos_quat: np.ndarray) -> (np.ndarray, np.ndarray):
        pos_delta_seq = pos_quat[:, :3]
        quat_seq = pos_quat[:, 3:]
        pos_base = pos_delta_seq[0:1, :]
        pos_delta_seq = pos_delta_seq - pos_base
        quat_seq = np.transpose(quat_seq, (1, 0))
        base = quat_seq[:, 0]
        ori_delta_seq = np.transpose(q_log_map(quat_seq, base), (1, 0))
        return np.concatenate((pos_delta_seq, ori_delta_seq), axis=1)

    @staticmethod
    def get_real_delta_sequence(pos_quat: np.ndarray) -> (np.ndarray, np.ndarray):
        # if np.isnan(quaternion_seq).any():
        #     print("nan detected before")
        pos_seq = pos_quat[:, :3]
        quaternion_seq = pos_quat[:, 3:]
        pos_base = np.concatenate((pos_seq[0:1], pos_seq[:-1]), axis=0)
        pos_delta_seq = pos_seq - pos_base
        o_delta_seq = np.zeros([quaternion_seq.shape[0], 3])
        quaternion_seq = np.transpose(quaternion_seq, (1, 0))
        o_base = np.concatenate((quaternion_seq[:, 0:1].copy(), quaternion_seq[:, :-1].copy()), axis=1)
        for i in range(o_delta_seq.shape[0]):
            o_delta_seq[i, :] = q_log_map(quaternion_seq[:, i], o_base[:, i])
        return np.concatenate((pos_delta_seq, o_delta_seq), axis=1)

    @staticmethod
    def get_direct_real_delta_sequence(pos_ori: np.ndarray) -> (np.ndarray, np.ndarray):
        # if np.isnan(quaternion_seq).any():
        #     print("nan detected before")
        pos_base = np.concatenate((pos_ori[0:1], pos_ori[:-1]), axis=0)
        pos_delta_seq = pos_ori.copy() - pos_base
        return pos_delta_seq

    def get_output(self,
                   pre,
                   source_trajectory,
                   target_trajectory,
                   seq_idx,
                   mean,
                   std):
        output = {}
        whole_source_pos_quat = source_trajectory["pos_quat"][seq_idx]
        whole_source_glb_pos_ori = source_trajectory["glb_pos_ori"][seq_idx]
        whole_target_pos_quat = target_trajectory["pos_quat"][seq_idx]
        whole_target_glb_pos_ori = target_trajectory["glb_pos_ori"][seq_idx]
        output[f"{pre}previous_pos_quat"] = torch.from_numpy(whole_target_pos_quat[:-self.len_lb, :]).float()
        output[f"{pre}previous_glb_pos_ori"] = torch.from_numpy(whole_target_glb_pos_ori[:-self.len_lb, :]).float()
        future_pos_quat = whole_source_pos_quat[-self.len_lb - 1:, :]
        output[f"{pre}future_pos_quat"] = torch.from_numpy(future_pos_quat).float()
        future_pos_ori = whole_source_glb_pos_ori[-self.len_lb - 1:, :]
        output[f"{pre}future_glb_pos_ori"] = torch.from_numpy(future_pos_ori).float()
        output[f"{pre}future_real_delta_direct"] = torch.from_numpy(
            self.get_direct_real_delta_sequence(future_pos_ori) * 10 / self.sampling_time).float()
        future_real_delta = self.get_real_delta_sequence(future_pos_quat)
        output[f"{pre}future_real_delta"] = torch.from_numpy(
            (future_real_delta * 10 / self.sampling_time - mean) / std).float()  # dm/s
        future_relative_delta = self.get_relative_delta_sequence(future_pos_quat)
        output[f"{pre}future_relative_delta"] = torch.from_numpy(future_relative_delta).float()

        return output

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

        output = self.get_output("",
                                 self.resample_source_trajectory,
                                 self.resample_target_trajectory,
                                 pose_idx,
                                 0,
                                 1)

        smooth_output = self.get_output("smooth_",
                                        self.smooth_resample_source_trajectory,
                                        self.smooth_resample_target_trajectory,
                                        pose_idx,
                                        0,
                                        1)
        output.update(smooth_output)

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
        output.update({
            "observation": {"fix": cam_fixed_framestack,
                            "gripper": cam_gripper_framestack, },
            "start": start,
            "current": idx,
            "end": lb_end,
        })
        return output


class Normalizer(torch.nn.Module):
    def __init__(self, traj_paths, args, train=True):
        super().__init__()
        """
        neg_ratio: ratio of silence audio clips to sample
        """
        # self.fix_cam_path = os.path.join(traj_paths, "camera", "220322060186", "rgb")
        # self.gripper_cam_path = os.path.join(traj_paths, "camera", "838212074210", "rgb")
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

        all_poses = []
        smooth_all_poses = []
        for traj in traj_paths:
            (raw_source_trajectory,
             resample_source_trajectory,
             smooth_resample_source_trajectory,
             ) = self.get_episode(traj, ablation=args.ablation,
                                  sampling_time=args.sampling_time,
                                  json_name="source_robot_trajectory.json")
            all_poses.append(resample_source_trajectory)
            smooth_all_poses.append(smooth_resample_source_trajectory)
        self.all_poses = all_poses
        self.smooth_all_poses = smooth_all_poses

        self.modalities = args.ablation.split("_")
        self.sampling_time = args.sampling_time / 1000
        self.len_lb = args.len_lb

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
        smoothed_traj, global_delta, smoothed_global_delta = smooth_traj(
            np.concatenate([np.array(resample_position_histroy), np.array(resample_orientation_history)], axis=1),
            (5e-4, 5e-4, 5e-4, 3e-4, 3e-4, 3e-4))

        raw_traj = {
            "pos_quat": np.concatenate([np.array(raw_position_histroy), np.array(raw_orientation_history)], axis=1),
            "absolute_real_time_stamps": np.array(raw_absolute_real_time_stamps),
            "relative_real_time_stamps": np.array(pose_trajectory.time_stamps),
        }

        resample_traj = {
            "pos_quat": np.concatenate([np.array(resample_position_histroy), np.array(resample_orientation_history)],
                                       axis=1),
            "global_pos_ori": global_delta[:, 3:],
            "pseudo_time_stamps": resample_pseudo_time_stamps,
            "relative_real_time_stamps": resampled_trajectory.pose_trajectory.time_stamps,
        }

        smooth_resample_traj = {
            "pos_quat": np.concatenate([smoothed_traj[:, :3], smoothed_traj[:, 3:]], axis=1),
            "global_pos_ori": smoothed_global_delta[:, 3:],
            "pseudo_time_stamps": resample_pseudo_time_stamps,
            "relative_real_time_stamps": resampled_trajectory.pose_trajectory.time_stamps,
        }
        return (
            raw_traj,
            resample_traj,
            smooth_resample_traj
        )

    def __len__(self):
        return self.num_frames

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

    def get_mean_std(self):

        all_delta_p = []
        all_delta_o = []
        smooth_all_delta_p = []
        smooth_all_delta_o = []
        for i in range(len(self.all_poses)):
            resample_traj = self.all_poses[i]
            smooth_resample_traj = self.smooth_all_poses[i]
            for idx in range(resample_traj["pos_quat"].shape[0] - self.len_lb):
                start = idx - self.len_obs
                lb_end = idx + self.len_lb
                lb_idx = np.arange(start, lb_end)
                lb_idx[lb_idx >= resample_traj["pos_quat"].shape[0]] = -1

                lb_idx[lb_idx < 0] = 0
                pose_idx = copy.deepcopy(lb_idx)

                whole_source_position_seq, whole_source_orientation_seq = get_pose_sequence(
                    resample_traj,
                    pose_idx)

                future_position_seq, future_orientation_seq = \
                    whole_source_position_seq[-self.len_lb - 1:, :], whole_source_orientation_seq[-self.len_lb - 1:, :]
                future_position_delta_seq, future_orientation_delta_seq = \
                    self.get_real_delta_sequence(future_position_seq, future_orientation_seq)

                future_position_delta_seq = future_position_delta_seq / self.sampling_time  # m/s
                future_orientation_delta_seq = future_orientation_delta_seq / self.sampling_time  # m/s

                smooth_whole_source_position_seq, smooth_whole_source_orientation_seq = get_pose_sequence(
                    smooth_resample_traj,
                    pose_idx)

                smooth_future_position_seq, smooth_future_orientation_seq = \
                    smooth_whole_source_position_seq[-self.len_lb - 1:, :], smooth_whole_source_orientation_seq[
                                                                            -self.len_lb - 1:, :]
                smooth_future_position_delta_seq, smooth_future_orientation_delta_seq = \
                    self.get_real_delta_sequence(smooth_future_position_seq, smooth_future_orientation_seq)

                smooth_future_position_delta_seq = smooth_future_position_delta_seq / self.sampling_time  # m/s
                smooth_future_orientation_delta_seq = smooth_future_orientation_delta_seq / self.sampling_time  # m/s

                all_delta_p.append(future_position_delta_seq[1:])
                all_delta_o.append(future_orientation_delta_seq[1:])
                smooth_all_delta_p.append(smooth_future_position_delta_seq[1:])
                smooth_all_delta_o.append(smooth_future_orientation_delta_seq[1:])
        all_delta_p = np.concatenate(all_delta_p, axis=0)
        p_mean, p_std = all_delta_p.mean(0, keepdims=True), all_delta_p.std(0, keepdims=True)
        all_delta_o = np.concatenate(all_delta_o, axis=0)
        o_mean, o_std = all_delta_o.mean(0, keepdims=True), all_delta_o.std(0, keepdims=True)
        # print(((all_delta_p - p_mean)/p_std).mean(0, keepdims=True), ((all_delta_p - p_mean)/p_std).std(0, keepdims=True))

        smooth_all_delta_p = np.concatenate(smooth_all_delta_p, axis=0)
        smooth_p_mean, smooth_p_std = smooth_all_delta_p.mean(0, keepdims=True), smooth_all_delta_p.std(0,
                                                                                                        keepdims=True)
        smooth_all_delta_o = np.concatenate(smooth_all_delta_o, axis=0)
        smooth_o_mean, smooth_o_std = smooth_all_delta_o.mean(0, keepdims=True), smooth_all_delta_o.std(0,
                                                                                                        keepdims=True)

        print(f"p_mean:{p_mean}")
        print(f"p_std:{p_std}")
        print(f"o_mean{o_mean}")
        print(f"o_std:{o_std}")
        print(f"smooth_p_mean:{smooth_p_mean}")
        print(f"smooth_p_std:{smooth_p_std}")
        print(f"smooth_o_mean:{smooth_o_mean}")
        print(f"smooth_o_std:{smooth_o_std}")
        return p_mean, p_std, o_mean, o_std, smooth_p_mean, smooth_p_std, smooth_o_mean, smooth_o_std


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
    args = SimpleNamespace(**args) if not isinstance(args, SimpleNamespace) else args
    trajs = [os.path.join(data_folder, traj) for traj in sorted(os.listdir(data_folder))]
    num_train = int(len(trajs) * 0.9)

    train_trajs_paths = trajs[:num_train]
    print(f"number of training trajectories: {len(train_trajs_paths)}")
    val_trajs_paths = trajs[num_train:]
    print(f"number of validation trajectories: {len(val_trajs_paths)}")

    normalizer = Normalizer(train_trajs_paths, args, True)
    # args.p_mean, args.p_std, args.o_mean, args.o_std, args.smooth_p_mean, args.smooth_p_std, args.smooth_o_mean, args.smooth_o_std = normalizer.get_mean_std()

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

    train_trajs_paths = train_trajs_paths[6:7]
    val_trajs_paths = val_trajs_paths[0:1]

    args = SimpleNamespace(**args) if not isinstance(args, SimpleNamespace) else args

    # args.p_mean = np.array([[0.00092885, 0.0003142, - 0.00077551]])
    # args.p_std = np.array([[0.16040024, 0.05464929, 0.1340441]])
    # args.o_mean = np.array([[-0.00018653, 0.00027793, - 0.0004129]])
    # args.o_std = np.array([[0.00741913, 0.03305291, 0.07000328]])
    # args.smooth_p_mean = np.array([[0.000947, 0.000333, - 0.00076833]])
    # args.smooth_p_std = np.array([[0.16021389, 0.05383626, 0.13407138]])
    # args.smooth_o_mean = np.array([[-0.00019119, 0.00028462, - 0.00039579]])
    # args.smooth_o_std = np.array([[0.00488407, 0.03090227, 0.06972235]])

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

    data_folder_path = '/fs/scratch/rb_bd_dlp_rng-dl01_cr_ROB_employees/students/jin4rng/data/robodemo_3_27'
    args = SimpleNamespace()

    args.ablation = 'vf_vg'
    args.num_stack = 5
    args.frameskip = 5
    args.no_crop = True
    args.crop_percent = 0.0
    args.resized_height_v = 480
    args.resized_width_v = 640
    args.len_lb = 10
    args.sampling_time = 100
    all_step_delta = []
    all_step_pose = []
    all_recover_pose = []
    train_loader, val_loader, _ = get_debug_loaders(batch_size=1, args=args, data_folder=data_folder_path,
                                              drop_last=False)
    print(len(train_loader))

    for idx, batch in enumerate(train_loader):
        # if idx >= 100:
        #     break
        print(f"{idx} \n")
        obs = batch["observation"]

        image_f = obs["fix"][0][-1].permute(1, 2, 0).numpy()
        image_g = obs["gripper"][0][-1].permute(1, 2, 0).numpy()
        image = np.concatenate([image_f, image_g], axis=0)
        cv2.imshow("asdf", image)
        time.sleep(0.2)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

        all_step_delta.append(batch["future_real_delta"][:, 1])
        all_step_pose.append(batch["future_pos_quat"][:, 1])





    # for idx, batch in enumerate(train_loader):
    #     if idx % args.len_lb != 0:
    #         continue
    #     # if idx >= 100:
    #     #     break
    #     print(f"{idx} \n")
    #     obs = batch["observation"]
    #
    #     # for image_g in obs[1][0]:
    #     #     cv2.imshow("asdf", image_g.permute(1, 2, 0).numpy())
    #     # key = cv2.waitKey(1)
    #     # if key == ord("q"):
    #     #     break
    #
    #     all_step_delta.append(batch["future_real_delta"][0, 1:, :])
    #     all_step_pose.append(batch["future_pos_quat"][0, 1:, :])
    #
    #     recover_pose = recover_pose_from_relative_vel(batch["future_real_delta"][0, 1:, :].detach().cpu().numpy(),
    #                                                   batch["previous_pos_quat"][0, -1, :].detach().cpu().numpy(),
    #                                                   vel_scale=0.015)
    #     all_recover_pose.append(torch.from_numpy(recover_pose))
    #
    # all_step = torch.concatenate(all_step_pose, dim=0)
    # pm = all_step.detach().cpu().numpy()
    # all_recover_pose = torch.concatenate(all_recover_pose, dim=0)
    # pmr = all_recover_pose.detach().cpu().numpy()
    # # pmr = torch.cat(all_step_delta, dim = 0).detach().cpu().numpy()
    # print(np.mean(pm, axis=0))
    # print(np.std(pm, axis=0))
    # print(np.max(pm, axis=0))
    # print(np.min(pm, axis=0))
    # t = np.arange(all_step.shape[0])
    # plt.figure()
    # plt.subplot(711)
    # o = np.stack([pm[:, 0], pmr[:, 0]], axis=1)
    # plt.plot(t, o, '-', )
    # plt.subplot(712)
    # o = np.stack([pm[:, 1], pmr[:, 1]], axis=1)
    # plt.plot(t, o, '-')
    # plt.subplot(713)
    # o = np.stack([pm[:, 2], pmr[:, 2]], axis=1)
    # plt.plot(t, o, '-')
    # plt.subplot(714)
    # o = np.stack([pm[:, 3], pmr[:, 3]], axis=1)
    # plt.plot(t, o, '-')
    # plt.subplot(715)
    # o = np.stack([pm[:, 4], pmr[:, 4]], axis=1)
    # plt.plot(t, o, '-')
    # plt.subplot(716)
    # o = np.stack([pm[:, 5], pmr[:, 5]], axis=1)
    # plt.plot(t, o, '-')
    # plt.subplot(717)
    # o = np.stack([pm[:, 6], pmr[:, 6]], axis=1)
    # plt.plot(t, o, '-')
    # plt.show()







    # all_step_smooth_delta = []
    # all_step_smooth_pose = []
    # all_step_glb_pos_ori = []
    # all_step_smooth_glb_pos_ori = []
    # all_step_real_delta_direct = []
    # all_step_smooth_real_delta_direct = []
    # for idx, batch in enumerate(val_loader):
    #     if idx % args.len_lb != 0:
    #         continue
    #
    #     print(f"{idx} \n")
    #     obs = batch["observation"]
    #
    #     # for image_g in obs[1][0]:
    #     #     cv2.imshow("asdf", image_g.permute(1, 2, 0).numpy())
    #     # key = cv2.waitKey(1)
    #     # if key == ord("q"):
    #     #     break
    #
    #     all_step_delta.append(batch["future_real_delta"][0, 1:, :])
    #     all_step_pose.append(batch["future_pos_quat"][0, 1:, :])
    #     all_step_glb_pos_ori.append(batch["future_glb_pos_ori"][0, 1:, :])
    #     all_step_smooth_delta.append(batch["smooth_future_real_delta"][0, 1:, :])
    #     all_step_smooth_pose.append(batch["smooth_future_pos_quat"][0, 1:, :])
    #     all_step_smooth_glb_pos_ori.append(batch["smooth_future_glb_pos_ori"][0, 1:, :])
    #     all_step_real_delta_direct.append(batch["future_real_delta_direct"][0, 1:, :])
    #     all_step_smooth_real_delta_direct.append(batch["smooth_future_real_delta_direct"][0, 1:, :])
    #
    # all_step = torch.concatenate(all_step_delta, dim=0)
    # pm = all_step.detach().cpu().numpy()
    # pmr = torch.concatenate(all_step_smooth_delta, dim=0).detach().cpu().numpy()
    # all_step_smooth_pose = torch.concatenate(all_step_smooth_pose, dim=0).detach().cpu().numpy()
    # all_step_pose = torch.concatenate(all_step_pose, dim=0).detach().cpu().numpy()
    # all_step_glb_pos_ori = torch.concatenate(all_step_glb_pos_ori, dim=0).detach().cpu().numpy()
    # all_step_smooth_glb_pos_ori = torch.concatenate(all_step_smooth_glb_pos_ori, dim=0).detach().cpu().numpy()
    # all_step_real_delta_direct = torch.concatenate(all_step_real_delta_direct, dim=0).detach().cpu().numpy()
    # all_step_smooth_real_delta_direct = torch.concatenate(all_step_smooth_real_delta_direct,
    #                                                       dim=0).detach().cpu().numpy()
    # print(np.mean(pm, axis=0))
    # print(np.std(pm, axis=0))
    # print(np.max(pm, axis=0))
    # print(np.min(pm, axis=0))
    #
    # print(np.mean(pmr, axis=0))
    # print(np.std(pmr, axis=0))
    # print(np.max(pmr, axis=0))
    # print(np.min(pmr, axis=0))
    #
    # t = np.arange(pm.shape[0])
    #
    # plt.figure()
    # plt.subplot(711)
    # o = np.stack([all_step_pose[:, 0], all_step_smooth_pose[:, 0]], axis=1)
    # plt.plot(t, o, '-', )
    # plt.subplot(712)
    # o = np.stack([all_step_pose[:, 1], all_step_smooth_pose[:, 1]], axis=1)
    # plt.plot(t, o, '-')
    # plt.subplot(713)
    # o = np.stack([all_step_pose[:, 2], all_step_smooth_pose[:, 2]], axis=1)
    # plt.plot(t, o, '-')
    # plt.subplot(714)
    # o = np.stack([all_step_pose[:, 3], all_step_smooth_pose[:, 3]], axis=1)
    # plt.plot(t, o, '-')
    # plt.subplot(715)
    # o = np.stack([all_step_pose[:, 4], all_step_smooth_pose[:, 4]], axis=1)
    # plt.plot(t, o, '-')
    # plt.subplot(716)
    # o = np.stack([all_step_pose[:, 5], all_step_smooth_pose[:, 5]], axis=1)
    # plt.plot(t, o, '-')
    # plt.subplot(717)
    # o = np.stack([all_step_pose[:, 6], all_step_smooth_pose[:, 6]], axis=1)
    # plt.plot(t, o, '-')
    # plt.show()
    #
    # plt.figure()
    # plt.subplot(711)
    # o = np.stack([pm[:, 0], pmr[:, 0]], axis=1)
    # plt.plot(t, o, '-', )
    # plt.subplot(712)
    # o = np.stack([pm[:, 1], pmr[:, 1]], axis=1)
    # plt.plot(t, o, '-')
    # plt.subplot(713)
    # o = np.stack([pm[:, 2], pmr[:, 2]], axis=1)
    # plt.plot(t, o, '-')
    # plt.subplot(714)
    # o = np.stack([pm[:, 3], pmr[:, 3]], axis=1)
    # plt.plot(t, o, '-')
    # plt.subplot(715)
    # o = np.stack([pm[:, 4], pmr[:, 4]], axis=1)
    # plt.plot(t, o, '-')
    # plt.subplot(716)
    # o = np.stack([pm[:, 5], pmr[:, 5]], axis=1)
    # plt.plot(t, o, '-')
    # # plt.subplot(717)
    # # o = np.stack([pm[:, 6], pmr[:, 6]], axis=1)
    # # plt.plot(t, o, '-')
    # plt.show()
    #
    # plt.figure()
    # plt.subplot(711)
    # o = np.stack([all_step_glb_pos_ori[:, 0], all_step_smooth_glb_pos_ori[:, 0]], axis=1)
    # plt.plot(t, o, '-', )
    # plt.subplot(712)
    # o = np.stack([all_step_glb_pos_ori[:, 1], all_step_smooth_glb_pos_ori[:, 1]], axis=1)
    # plt.plot(t, o, '-')
    # plt.subplot(713)
    # o = np.stack([all_step_glb_pos_ori[:, 2], all_step_smooth_glb_pos_ori[:, 2]], axis=1)
    # plt.plot(t, o, '-')
    # plt.subplot(714)
    # o = np.stack([all_step_glb_pos_ori[:, 3], all_step_smooth_glb_pos_ori[:, 3]], axis=1)
    # plt.plot(t, o, '-')
    # plt.subplot(715)
    # o = np.stack([all_step_glb_pos_ori[:, 4], all_step_smooth_glb_pos_ori[:, 4]], axis=1)
    # plt.plot(t, o, '-')
    # plt.subplot(716)
    # o = np.stack([all_step_glb_pos_ori[:, 5], all_step_smooth_glb_pos_ori[:, 5]], axis=1)
    # plt.plot(t, o, '-')
    # # plt.subplot(717)
    # # o = np.stack([all_step_glb_pos_ori[:, 6], all_step_smooth_glb_pos_ori[:, 6]], axis=1)
    # # plt.plot(t, o, '-')
    # plt.show()
    #
    # plt.figure()
    # plt.subplot(711)
    # o = np.stack([all_step_real_delta_direct[:, 0], all_step_smooth_real_delta_direct[:, 0]], axis=1)
    # plt.plot(t, o, '-', )
    # plt.subplot(712)
    # o = np.stack([all_step_real_delta_direct[:, 1], all_step_smooth_real_delta_direct[:, 1]], axis=1)
    # plt.plot(t, o, '-')
    # plt.subplot(713)
    # o = np.stack([all_step_real_delta_direct[:, 2], all_step_smooth_real_delta_direct[:, 2]], axis=1)
    # plt.plot(t, o, '-')
    # plt.subplot(714)
    # o = np.stack([all_step_real_delta_direct[:, 3], all_step_smooth_real_delta_direct[:, 3]], axis=1)
    # plt.plot(t, o, '-')
    # plt.subplot(715)
    # o = np.stack([all_step_real_delta_direct[:, 4], all_step_smooth_real_delta_direct[:, 4]], axis=1)
    # plt.plot(t, o, '-')
    # plt.subplot(716)
    # o = np.stack([all_step_real_delta_direct[:, 5], all_step_smooth_real_delta_direct[:, 5]], axis=1)
    # plt.plot(t, o, '-')
    # # plt.subplot(717)
    # # o = np.stack([all_step_real_delta_direct[:, 6], all_step_smooth_real_delta_direct[:, 6]], axis=1)
    # # plt.plot(t, o, '-')
    # plt.show()
