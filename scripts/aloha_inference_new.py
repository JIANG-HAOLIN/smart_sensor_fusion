import sys

sys.path.append("/home/jin4rng/Documents/tami_clap_candidate")

from tami_clap_candidate.rpc_interface.rpc_interface import RPCInterface

import cv2
from tami_clap_candidate.sensors.realsense import RealsenseRecorder, Preset

import os
import sys

import torch
import argparse
import numpy as np
import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import copy

import time
from utils.quaternion import q_exp_map, q_log_map, exp_map, log_map, q_to_rotation_matrix, q_from_rot_mat
from utils.inference_utils import load_checkpoints
import torchvision.transforms as T
from utils.video_collection import save_video
from src.datasets.dummy_robot_arm import get_loaders, Normalizer

GRIPPER_OPEN_STATE = 0.02

def limit_norm(arr, min, max):
    gap = max - min
    arr = arr - min
    arr = arr / gap * 2
    arr = arr - 1
    return arr


def inference(cfg: DictConfig, args: argparse.Namespace):
    # on robot side start,
    # (namespace is hardcoded to panda)
    # 1) launch robot control
    # 2) roslaunch tami_rpc_interface rpc_publisher_interface.launch topics:=[/panda/PandaStatePublisher/arm_states] interfaces:=[control/RobotState/getArmState] ports:=[34110] host:=10.87.172.60
    # 3) roslaunch tami_rpc_interface rpc_arm_state_interface.launch host:=10.87.172.60 port:=34410
    # 4) good to go

    cfgs = HydraConfig.get()
    cfg_path = cfgs.runtime['config_sources'][1]['path']
    normalizer = Normalizer.from_json(os.path.join(cfg_path
                                                   , "normalizer_config.json"))
    model = load_checkpoints(cfg, args)

    GENERATE_TRAJECTORY = False  # we send 1-step commands instead of a full trajectory

    rpc = RPCInterface("10.87.170.254")
    rpc.activate_cartesian_nonblocking_controller()
    # curr_target_pose = np.array([0.45, 0.03, 0.25, 0.0, 1.0, 0.0, 0.0])
    # success = rpc.open_gripper()
    # success = rpc.goto_cartesian_pose_blocking(curr_target_pose[:3], curr_target_pose[3:], True)

    ensemble = False
    # the namespacve is hardcoded to panda

    transform_cam = T.Compose([T.Resize((cfg.datasets.dataloader.args.resized_height_v,
                                         cfg.datasets.dataloader.args.resized_width_v), antialias=None), ])
    rs_recorder = RealsenseRecorder(
        height=480,
        width=640,
        fps=30,
        record_depth=False,
        depth_unit=0.001,
        preset=Preset.HighAccuracy,
        memory_first=True,
    )
    frame_list = []
    action_list = []
    image_list = []

    send_time = 0.10

    cv2.imshow("press q to exit", np.zeros((1, 1)))



    max_timesteps = 1000

    t = 0
    buffer = 0
    query_frequency = 5  # for chunk action

    all_time_position = torch.zeros([max_timesteps, max_timesteps + query_frequency, 4]).cuda()
    all_time_orientation = torch.zeros([max_timesteps, max_timesteps + query_frequency, 4]).cuda()
    while True:
        # if t % query_frequency != 0:
        #     t+=1
        #     continue
        print(t)
        time_start = time.time()

        # Get arm state
        arm_state = rpc.get_robot_state()
        gripper_state = arm_state.gripper[0]
        gripper_state = normalizer.normalize(gripper_state, "gripper")
        pose_loc = arm_state.pose.vec(quat_order="wxyz")
        qpos = log_map(pose_loc, np.array([0, 0, 0, 0, 1, 0, 0]))
        qpos = normalizer.normalize(qpos, "target_glb_pos_ori")
        qpos = np.concatenate([qpos, np.array(gripper_state)], axis=-1)
        # qpos = pose_loc
        qpos = torch.from_numpy(qpos).unsqueeze(0).to(args.device).float()

        realsense_frames = rs_recorder.get_frame()
        to_plot = rs_recorder._visualize_frame(realsense_frames).copy()
        cv2.imshow("direct from real sense", to_plot)
        frame_list.append(to_plot)
        image = np.stack(copy.deepcopy(frame_list[-1:]), axis=0)  # stack to get image sequence [seq_len, H, W, BRG]
        image = image[:, :, :,
                ::-1].copy()  # reverse channel as re_recorder use cv2 -> BRG and we want RBG --> [seq_len, H, W, RGB]
        image = torch.as_tensor(image)
        image = image.float()
        image = image.permute(0, 3, 1, 2)  # [seq_len, RGB, H, W]
        image = image / 255
        im_g = transform_cam(image[:, :, :, :640]).unsqueeze(0).to(
            args.device).float()  # unsqueeze to add batch dim -> [B, seq_len, RBG, H, W]
        im_f = transform_cam(image[:, :, :, 640:]).unsqueeze(0).to(
            args.device).float()  # unsqueeze to add batch dim -> [B, seq_len, RBG, H, W]


        cv2.imshow("sanity check gripper(color should reverse): gripper",
                   im_g[0][-1].permute(1, 2, 0).detach().cpu().numpy())  # sanity check

        cv2.imshow("sanity check fix(color should reverse)",
                   im_f[0][-1].permute(1, 2, 0).detach().cpu().numpy())  # sanity check

        image_list.append(im_f)
        if len(image_list) < 14:
            continue
        input_image_list = torch.cat(image_list[-9::2], dim=1)
        multimod_inputs = {"vision": {"v_fix": input_image_list}, }
        # multimod_inputs = {"vision": {"v_fix": im_f}}

        inference_type = cfg.pl_modules.pl_module.action
        if inference_type == "real_delta_target":
            action_type = "target_real_delta"

        elif inference_type == "position":
            action_type = "source_glb_pos_ori"

        elif inference_type == "real_delta_source":
            action_type = "source_real_delta"

        elif inference_type == "direct_vel":
            action_type = "direct_vel"

        all_action, raw_action, all_time_position, all_time_orientation, og_a_hat = model.rollout(
            qpos,
            multimod_inputs,
            env_state=None,
            actions=None,
            is_pad=None,
            all_time_position=all_time_position,
            all_time_orientation=all_time_orientation,
            t=t,
            args=args,
            v_scale=3,
            inference_type=inference_type,
            num_queries=query_frequency,
            normalizer=normalizer,
            action_type=action_type,
        )

        if ensemble:
            pose_new = raw_action[1:]
            start = time.time()

            gripper_command = raw_action[0]
        else:
            buffer = all_action[:query_frequency, :] if t % query_frequency == 0 else buffer
            # generate message
            start = time.time()
            pose_new = buffer[t % query_frequency][1:]

            gripper_command = buffer[t % query_frequency][0]

        rpc.goto_cartesian_pose_nonblocking(pose_new[:3], pose_new[3:], GENERATE_TRAJECTORY)
        loop_time = time.time() - start
        if loop_time < send_time:
            time.sleep(send_time - loop_time)

        # pose_new = raw_action
        # pose_new[3:] = np.array([0, 1, 0, 0])
        # position = Vec3f.new_message(x=float(pose_new[0]), y=float(pose_new[1]), z=float(pose_new[2]))
        # orientation = Quaternion.new_message(w=float(pose_new[3]), x=float(pose_new[4]), y=float(pose_new[5]),
        #                                      z=float(pose_new[6]))
        # pose_msg = Pose.new_message(position=position, orientation=orientation)
        #
        # client_send.moveToCartesianPose(pose=pose_msg, blocking=False, generateTraj=False)
        #
        # loop_time = time.time() - time_start
        # print(loop_time)
        # if loop_time < send_time:
        #     time.sleep(send_time - loop_time)

        if gripper_state >= GRIPPER_OPEN_STATE:  # if open
            if gripper_command < GRIPPER_OPEN_STATE:
                rpc.close_gripper()
        elif gripper_state < GRIPPER_OPEN_STATE:  # if close
            if gripper_command > GRIPPER_OPEN_STATE:
                rpc.open_gripper()

        print("iter time:", time.time() - time_start)
        key = cv2.waitKey(1)
        if key == ord("q"):
            cv2.destroyAllWindows()
            save_video(5, "test_cam_save", frame_list)
            break
        t += 1


if __name__ == "__main__":

    project_path = os.path.abspath(os.path.join(__file__, '..', '..'))
    sys.path.append(
        project_path)  # without this line:hydra.errors.InstantiationException: Error locating target
    # 'src.datasets.bautiro_drilling_dataset.get_loaders', set env var HYDRA_FULL_ERROR=1 to see chained exception.
    # full_key: datasets.dataloader

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        default='../checkpoints/cupboard/name=alohaname=vae_vanillaaction=positionname=coswarmuplr=4e-05weight_decay=0.0001kl_divergence=10source=Trueresized_height_v=240resized_width_v=320_04-26-10:32:06')
    parser.add_argument('--ckpt_path', type=str,
                        default='not needed anymore')
    parser.add_argument('--device', type=str,
                        default='cuda')

    args = parser.parse_args()

    if 'checkpoints' in os.listdir(args.config_path):
        hydra.initialize(config_path=args.config_path, version_base=None, )
        cfg = hydra.compose(config_name='config', return_hydra_config=True, )
        HydraConfig().cfg = cfg
        OmegaConf.resolve(cfg)

        inference(cfg=cfg, args=args)
    else:
        # inference on multiple checkpoints
        for sub_dir in os.listdir(args.config_path):
            sub_dir_path = os.path.join(args.config_path, sub_dir)
            if os.path.isdir(sub_dir_path) and sub_dir != '.hydra':
                print(f'current hydra config: {sub_dir}')
                hydra.initialize(config_path=sub_dir_path, version_base=None, )
                cfg = hydra.compose(config_name='config', return_hydra_config=True, )
                HydraConfig().cfg = cfg
                OmegaConf.resolve(cfg)

                inference(cfg=cfg, args=args)
                GlobalHydra.instance().clear()
