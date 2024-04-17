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

GRIPPER_OPEN_STATE = 0.02

denormalize = lambda x, arr: (x + 1) / 2 * (arr["max"] - arr["min"]) + arr["min"]


POSE_OFFSET = np.array([0.0, 0.0, -0.05, 1.0, 0.0, 0.0, 0.0])  # wxyz


def apply_offset(pose, pose_offset):
    # Computes T_pose * T_pose_offset => pose_new
    # Assuming quaternions are in wxyz format
    t = pose[:3]
    R = q_to_rotation_matrix(pose[3:])

    to = pose_offset[:3]
    Ro = q_to_rotation_matrix(pose_offset[3:])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    To = np.eye(4)
    To[:3, :3] = Ro
    To[:3, 3] = to

    T_new = np.matmul(T, To)
    pose_new = np.zeros((7,))
    pose_new[:3] = T_new[:3, 3]
    pose_new[3:] = q_from_rot_mat(T_new[:3, :3])
    return pose_new

def limit_norm(arr, min, max):
    gap = max - min
    arr = arr - min
    arr = arr / gap * 2
    arr = arr - 1
    return arr

norm_state = {
    'resample': {'target_real_delta': {'max': np.array([0.01833077, 0.01471558, 0.01419077, 0.0090178, 0.00767898,
                                                        0.01499568]),
                                       'min': np.array([-0.01137316, -0.01453483, -0.01489501, -0.00618764, -0.00964906,
                                                        -0.01807459]), 'mean': np.array(
            [3.13497064e-04, 4.33342494e-04, -1.08199064e-03, 4.00690490e-05,
             -2.47188444e-04, 2.91489864e-04]),
                                       'std': np.array([0.00257714, 0.00284448, 0.0027104, 0.0010101, 0.00154394,
                                                        0.00194711])},
                 'target_glb_pos_ori': {'max': np.array([0.60799567, 0.24106479, 0.37450424, 0.10115965, 0.04896518,
                                                         0.17713495]),
                                        'min': np.array([0.29311409, -0.06481589, 0.08938819, -0.02968255, -0.10085378,
                                                         -0.15162727]),
                                        'mean': np.array([0.45605428, 0.08565441, 0.174311, 0.02461957, -0.03768246,
                                                          0.01832523]),
                                        'std': np.array([0.068358, 0.07183608, 0.0707843, 0.02024454, 0.02290095,
                                                         0.04824965])},
                 'source_real_delta': {'max': np.array([0.05323321, 0.04156064, -0.01151191, 0.04448153, 0.05435663,
                                                        0.03082125]),
                                       'min': np.array([-0.05535634, -0.04399007, -0.08517262, -0.03585722, -0.04552806,
                                                        -0.03912532]),
                                       'mean': np.array([-0.00534913, 0.00288355, -0.05514399, 0.0020868, 0.01027822,
                                                         0.00162523]),
                                       'std': np.array([0.01629853, 0.0115366, 0.01200146, 0.01321739, 0.01472549,
                                                        0.00834285])},
                 'source_glb_pos_ori': {'max': np.array([0.60589246, 0.24761381, 0.32945419, 0.11862382, 0.06972952,
                                                         0.1831929]),
                                        'min': np.array([0.2776695, -0.06622958, 0.03222788, -0.02801248, -0.09814695,
                                                         -0.15366502]),
                                        'mean': np.array([0.4504151, 0.08810488, 0.12027499, 0.02617468, -0.0271667,
                                                          0.02004663]),
                                        'std': np.array([0.06902871, 0.07362574, 0.07155867, 0.02378412, 0.0213967,
                                                         0.04946909])},
                 'gripper': {'max': np.array([0.03066929]), 'min': np.array([0.00145977, ]),
                             'mean': np.array([0.01886992, ]), 'std': np.array([0.01372382, ])},
                 'direct_vel': {'max': np.array([0.04288793, 0.03173504, -0.01813139, 0.04065353, 0.04867868,
                                                 0.0184301]),
                                'min': np.array([-0.04977977, -0.03515093, -0.07595439, -0.03135369, -0.04040732,
                                                 -0.02396019]),
                                'mean': np.array([-0.00563918, 0.00245047, -0.05403601, 0.00200457, 0.01039245,
                                                  0.00130618]),
                                'std': np.array([0.01390834, 0.0089605, 0.00962011, 0.01280517, 0.0141328,
                                                 0.00703205])}},
    'smooth': {'target_real_delta': {'max': np.array([0.01769036, 0.01410254, 0.01244026, 0.00721367, 0.00660652,
                                                      0.01507445]),
                                     'min': np.array([-0.01034324, -0.01433304, -0.01295911, -0.00580396, -0.00849982,
                                                      -0.01739865]),
                                     'mean': np.array([3.13550801e-04, 4.33161468e-04, -1.08197222e-03, 4.08131623e-05,
                                                       -2.48204061e-04, 2.91412280e-04]),
                                     'std': np.array([0.00253795, 0.00280512, 0.00266423, 0.00096524, 0.00149994,
                                                      0.00189667])},
               'target_glb_pos_ori': {'max': np.array([0.60782946, 0.24107972, 0.37453574, 0.10093857, 0.04868671,
                                                       0.17745757]),
                                      'min': np.array([0.29317631, -0.06488244, 0.08923991, -0.02988128, -0.10058427,
                                                       -0.15150303]),
                                      'mean': np.array([0.45605428, 0.08565441, 0.174311, 0.02461957, -0.03768246,
                                                        0.01832523]),
                                      'std': np.array([0.06835725, 0.07183541, 0.07078369, 0.02024171, 0.02289883,
                                                       0.04824871])},
               'source_real_delta': {'max': np.array([0.0533458, 0.04051842, -0.01175208, 0.04351691, 0.05407896,
                                                      0.03038214]),
                                     'min': np.array([-0.05561119, -0.04391569, -0.08456834, -0.0348076, -0.04496127,
                                                      -0.03898464]),
                                     'mean': np.array([-0.0053492, 0.00288325, -0.05514397, 0.00208666, 0.01027814,
                                                       0.00162536]),
                                     'std': np.array([0.01628969, 0.01152728, 0.01199221, 0.0132063, 0.01471578,
                                                      0.00832774])},
               'source_glb_pos_ori': {'max': np.array([0.60580062, 0.24757468, 0.32923787, 0.11831453, 0.06992805,
                                                       0.1833786]),
                                      'min': np.array([0.27800035, -0.06643748, 0.03202078, -0.02810872, -0.09806573,
                                                       -0.15393454]),
                                      'mean': np.array([0.4504151, 0.08810488, 0.12027499, 0.02617468, -0.0271667,
                                                        0.02004663]),
                                      'std': np.array([0.06902801, 0.0736251, 0.07155798, 0.0237819, 0.02139435,
                                                       0.04946812])},
               'gripper': {'max': np.array([0.03066929]), 'min': np.array([0.00145977]), 'mean': np.array([0.01886992]),
                           'std': np.array([0.01372382])},
               'direct_vel': {'max': np.array([0.04246886, 0.03144862, -0.01852505, 0.04025868, 0.04797959,
                                               0.01792963]),
                              'min': np.array([-0.04955756, -0.03480149, -0.07592206, -0.03116205, -0.03996206,
                                               -0.02402754]),
                              'mean': np.array([-0.00563918, 0.00245047, -0.05403601, 0.00200456, 0.01039244,
                                                0.00130622]),
                              'std': np.array([0.01390194, 0.00895157, 0.00961124, 0.01279775, 0.01412746,
                                               0.00702197])}}}


def inference(cfg: DictConfig, args: argparse.Namespace):
    # on robot side start,
    # (namespace is hardcoded to panda)
    # 1) launch robot control
    # 2) roslaunch tami_rpc_interface rpc_publisher_interface.launch topics:=[/panda/PandaStatePublisher/arm_states] interfaces:=[control/RobotState/getArmState] ports:=[34110] host:=10.87.172.60
    # 3) roslaunch tami_rpc_interface rpc_arm_state_interface.launch host:=10.87.172.60 port:=34410
    # 4) good to go
    model = load_checkpoints(cfg, args)

    GENERATE_TRAJECTORY = False  # we send 1-step commands instead of a full trajectory

    rpc = RPCInterface("10.87.170.254")
    rpc.activate_cartesian_nonblocking_controller()

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
        gripper_state = limit_norm(gripper_state, norm_state["resample"]["gripper"]["min"], norm_state["resample"]["gripper"]["max"])
        pose_loc = arm_state.pose.vec(quat_order="wxyz")
        qpos = log_map(pose_loc, np.array([0, 0, 0, 0, 1, 0, 0]))
        qpos = limit_norm(qpos, norm_state["resample"]["target_glb_pos_ori"]["min"], norm_state["resample"]["target_glb_pos_ori"]["max"])
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
        cv2.imshow("sanity check gripper(color should reverse): gripper",
                   im_g[0][-1].permute(1, 2, 0).detach().cpu().numpy())  # sanity check
        im_f = transform_cam(image[:, :, :, 640:]).unsqueeze(0).to(
            args.device).float()  # unsqueeze to add batch dim -> [B, seq_len, RBG, H, W]
        cv2.imshow("sanity check fix(color should reverse)",
                   im_f[0][-1].permute(1, 2, 0).detach().cpu().numpy())  # sanity check
        multimod_inputs = {"vision": {"v_fix": im_f}, }
        # multimod_inputs = {"vision": im_g}

        inference_type = "real_delta_target"
        if inference_type == "real_delta_target":
            action_norm_state = norm_state["resample"]["target_real_delta"]

        elif inference_type == "position":
            action_norm_state = norm_state["resample"]["source_glb_pos_ori"]

        elif inference_type == "real_delta_source":
            action_norm_state = norm_state["resample"]["source_real_delta"]

        elif inference_type == "direct_vel":
            action_norm_state = norm_state["resample"]["direct_vel"]

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
            v_scale=2,
            inference_type=inference_type,
            num_queries=query_frequency,
            action_norm_state=action_norm_state,
            qpos_norm_state=norm_state["resample"]["target_glb_pos_ori"],
            gripper_norm_state=norm_state["resample"]["gripper"]
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

        if inference_type == "position":
            pose_new = apply_offset(pose_new, POSE_OFFSET)
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
                        default='../checkpoints/cupboard/name=alohaname=vae_vanillaaction=real_delta_targetname=coswarmuplr=5e-05weight_decay=0.0001kl_divergence=10hidden_dim=512output_layer_index=-1source=True_04-16-08:18:34')
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
