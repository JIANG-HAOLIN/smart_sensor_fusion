import sys

import matplotlib.pyplot as plt

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
from multiprocessing import Queue, Process, Event, Lock
import logging
import time
from utils.quaternion import q_exp_map, q_log_map, exp_map, log_map, q_to_rotation_matrix, q_from_rot_mat
from utils.inference_utils import load_checkpoints
import torchvision.transforms as T
from utils.video_collection import save_video
from src.datasets.vision_audio_robot_arm import get_loaders, Normalizer
import datetime
import pyaudio
from collections import deque
import torchaudio
from utils.quaternion import q_exp_map, q_log_map, recover_pose_from_quat_real_delta, exp_map_seq, log_map_seq, exp_map




GRIPPER_OPEN_STATE = 0.02
logger = logging.getLogger(__name__)


class Inference:

    def __init__(self, **kwargs) -> None:
        # Config related variables
        self.target_ip = kwargs.get("target_robot_ip", None)
        self.resolution = kwargs.get("resolution", (640, 480))
        self.exposure = kwargs.get("exposure_time", -1)
        self.image_freq = kwargs.get("image_frequency", 30)
        logger.info(f"Using target robot IP: {self.target_ip}")

        # Class variables
        self.finish_event = Event()
        self.target_state_list = []
        self.image_list = []
        self.audio_list = []

        self.state_lock = Lock()
        self.image_lock = Lock()

        self.realsense_config = {
            "height": self.resolution[1],
            "width": self.resolution[0],
            "fps": self.image_freq,
            "record_depth": False,
            "depth_unit": 0.001,
            "preset": Preset.HighAccuracy,
            "memory_first": True,
            "exposure_time": self.exposure
        }

        self.mdl = kwargs.get("model", None)
        self.normalizer = kwargs.get("normalizer", None)
        self.transform_cam = kwargs.get("transform_cam", None)
        self.loader = kwargs.get("loader", None)

    def start_demo_recording_inference(self, root_folder):
        if not os.path.exists(root_folder):
            os.mkdir(root_folder)

        date_string = datetime.datetime.now().isoformat()
        folder_suffix = date_string.replace(":", "-").replace(".", "-")
        folder_name = os.path.join(root_folder, "demo_" + folder_suffix + "/")
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        self.folder_name = folder_name
        camera_folder_name = os.path.join(folder_name, "camera")
        self.realsense_config["record_folder"] = camera_folder_name

        target_state_q = Queue()
        image_q = Queue()
        audio_q = Queue()
        action_q = Queue()
        obs_image = Queue(maxsize=1)
        obs_audio = Queue(maxsize=1)
        obs_qpos = Queue(maxsize=1)
        target_state_deque = deque(maxlen=100)
        inference_recording = []
        image_deque = deque(maxlen=100)
        audio_deque = deque(maxlen=100)

        processes = []
        processes.append(Process(target=self.mic_process, args=(audio_q,), daemon=True))
        processes.append(Process(target=self.realsense_process, args=(self.realsense_config, image_q), daemon=True))
        if self.target_ip != None:
            processes.append(Process(target=self.target_robot_process, args=(target_state_q,), daemon=True))
        # processes.append(Process(target=self.gather_obs_qpos, args=(target_state_q, image_q, audio_q, obs_qpos, obs_image, obs_audio), daemon=True))
        # processes.append(Process(target=self.gather_obs_image, args=(target_state_q, image_q, audio_q, obs_qpos, obs_image, obs_audio), daemon=True))
        # processes.append(Process(target=self.gather_obs_audio, args=(target_state_q, image_q, audio_q, obs_qpos, obs_image, obs_audio), daemon=True))


        for p in processes:
            p.start()


        rpc = RPCInterface("10.87.170.254")
        rpc.activate_cartesian_nonblocking_controller()
        init_target_pose = np.array([0.3, 0.0, 0.3, 0.0, 1.0, 0.0, 0.0])
        # success = rpc.open_gripper()
        success = rpc.goto_cartesian_pose_blocking(init_target_pose[:3], init_target_pose[3:], True)

        ensemble = False

        test_list1 = []
        test_list2 = []

        send_time = 0.1

        max_timesteps = 1000

        cur_step = 0
        buffer = 0
        query_frequency = 6  # for chunk action

        dataloader = iter(self.loader)
        while True:
            while target_state_q.qsize() > 0:
                target_state_deque.append(target_state_q.get())
            while image_q.qsize() > 0:
                image_deque.append(image_q.get())
            while audio_q.qsize() > 0:
                audio_deque.append(audio_q.get())

            # image_deque, audio_deque, qpos_deque = obs_image.get(), obs_audio.get(), obs_qpos.get()
            # print(len(target_state_deque))
            if len(image_deque) >= 99 and len(audio_deque) >= 99 and len(target_state_deque) >= 99:
                start_time = time.time()
                print(f"###########################inference started########################################")
                break

        while True:
            while target_state_q.qsize() > 0:
                target_state_deque.append(target_state_q.get())
            while image_q.qsize() > 0:
                image_deque.append(image_q.get())
                inference_recording.append(image_deque[-1][0])
            while audio_q.qsize() > 0:
                audio_deque.append(audio_q.get())

            cur_t = time.time()
            print(f"current step:{cur_step}, current time:{cur_t - start_time}")

            cv2.imshow("Press q to exit", np.zeros((1, 1)))
            key = cv2.waitKey(1)

            # image_deque, audio_deque, qpos_deque = obs_image.get(), obs_audio.get(), obs_qpos.get()
            print("image_delay", cur_t - image_deque[-1][-1])
            print("audio_delay", cur_t - audio_deque[-1][-1])
            print("qpos_delay", cur_t - target_state_deque[-1][-1])

            obs_time_stamps = np.array([[2.0, 1.5, 1.0, 0.5, 0.0]])
            if key == ord("q"):
                self.finish_event.set()
                cv2.destroyAllWindows()
                save_video(30, "test_cam_save", inference_recording)
                break

            ###################### start model computation ########################
            # continue
            # Get qpos observation
            qpos_time_stamp = np.stack([np.array([cur_t - qpos[-1]]) for qpos in target_state_deque], axis=0)
            # print(qpos_time_stamp.shape)
            qpos_pseudo_time_stamp = np.argmin(np.abs(qpos_time_stamp - obs_time_stamps), axis=0)
            print(qpos_pseudo_time_stamp)
            # print(qpos_time_stamp[qpos_pseudo_time_stamp])
            qpos_nongrip = np.stack([target_state_deque[t][0] for t in qpos_pseudo_time_stamp], axis=0)
            print("qpos_nogrop:",qpos_nongrip)
            real_qpos_6 = torch.from_numpy(qpos_nongrip).unsqueeze(0)[:, -1:, :].to(args.device)
            real_qpos_3 = torch.from_numpy(qpos_nongrip).unsqueeze(0)[:, -1:, ::2].to(args.device)
            print("real_qpos_6:", real_qpos_6)
            test_list1.append(real_qpos_6[0][0].detach().cpu().numpy())
            qpos_grip = np.stack([target_state_deque[t][1] for t in qpos_pseudo_time_stamp], axis=0)
            qpos_nongrip = self.normalizer.normalize(qpos_nongrip, "target_glb_pos_ori")
            qpos_grip = self.normalizer.normalize(qpos_grip, "gripper")
            qpos_nongrip = torch.from_numpy(qpos_nongrip).unsqueeze(0)
            print("normalized qpos observation:", qpos_nongrip[:, :, ::2])
            # print(qpos_nongrip.shape)
            # Get image observation
            imgs = np.stack([image_deque[t][0] for t in range(-61, 0, 15)], axis=0)  # [seq_len, H, W, RGB]
            image = torch.as_tensor(imgs).permute(0, 3, 1, 2)/255  # [seq_len, RGB, H, W]
            im_g = self.transform_cam(image[:, :, :, :640]).unsqueeze(0)  # unsqueeze to add batch dim -> [B, seq_len, RGB, H, W]
            im_f = self.transform_cam(image[:, :, :, 640:]).unsqueeze(0)  # unsqueeze to add batch dim -> [B, seq_len, RGB, H, W]
            # print(im_f.shape)
            cv2.imshow("sanity check gripper(color should reverse): gripper",
                       im_g[0][-1].permute(1, 2, 0).detach().cpu().numpy())  # sanity check

            cv2.imshow("sanity check fix(color should reverse)",
                       im_f[0][-1].permute(1, 2, 0).detach().cpu().numpy())  # sanity check

            # Get audio observation
            audio = [audio_deque[t][0] for t in range(-75, 0)]  # !! list only !! don't use numpy array here !!
            audio_data = torch.from_numpy(np.frombuffer(b''.join(audio), np.int16).copy())
            # print(audio_data.shape)
            audio_clip = self.normalizer.normalize_audio(audio_data)
            audio_clip = torchaudio.functional.resample(audio_clip, 44100, 16000).unsqueeze(0).unsqueeze(0)  # [1, 1, 40000]
            # print(audio_clip.shape)
            ###################################################################################################################
            # try:
            #     batch = next(dataloader)
            # except (StopIteration, KeyboardInterrupt):
            #     print("Asdfasfasdfasdfafasdfasdf")
            #     test_list1 = np.stack(test_list1, axis=0)
            #     test_list2 = np.stack(test_list2, axis=0)
            #     plt.subplot(3, 1, 1)
            #     plt.plot(np.arange(test_list2.shape[0]), test_list2[:, 0])
            #
            #     plt.plot(np.arange(test_list1.shape[0]), test_list1[:, 0])
            #     plt.subplot(3, 1, 2)
            #     plt.plot(np.arange(test_list2.shape[0]), test_list2[:, 2])
            #
            #     plt.plot(np.arange(test_list1.shape[0]), test_list1[:, 2])
            #     plt.subplot(3, 1, 3)
            #     plt.plot(np.arange(test_list2.shape[0]), test_list2[:, 4])
            #
            #     plt.plot(np.arange(test_list1.shape[0]), test_list1[:, 4])
            #     plt.show()
            # inp_data = batch["observation"]
            # for key, value in inp_data.items():
            #     inp_data[key] = value.to(args.device)
            # multimod_inputs = {
            #     "vision": inp_data,
            #     "qpos": batch["traj"]["target_glb_pos_ori"]["obs"][:, :, ::2].float().to(args.device),
            #     "audio": batch["observation"]["a_holebase"].to(args.device),
            # }
            # print("dataloader inputs denormalized:", self.normalizer.denormalize(
            #     torch.stack([multimod_inputs["qpos"][:, :, 0], torch.zeros_like(multimod_inputs["qpos"][:, :, 0]),
            #                multimod_inputs["qpos"][:, :, 1], torch.zeros_like(multimod_inputs["qpos"][:, :, 0]),
            #                multimod_inputs["qpos"][:, :, 2], torch.zeros_like(multimod_inputs["qpos"][:, :, 0]), ], dim=-1),
            #     "target_glb_pos_ori"))
            #######################################################################################################################
            multimod_inputs = {"vision": {"v_fix": im_f.float().to(args.device)},
                               "qpos": qpos_nongrip[:, :, ::2].float().to(args.device),
                               "audio": audio_clip.to(args.device),
                               }

            inference_type = cfg.pl_modules.pl_module.action
            if inference_type == "real_delta_target":
                action_type = "target_real_delta"

            elif inference_type == "position":
                action_type = "source_glb_pos_ori"

            elif inference_type == "real_delta_source":
                action_type = "source_real_delta"

            elif inference_type == "direct_vel":
                action_type = "direct_vel"
            out = self.mdl(multimod_inputs["qpos"][:, -1, :],
                        multimod_inputs,
                        actions=None,
                        is_pad=None,
                        mask=None,
                        mask_type="None",
                        task="repr",
                        mode="val",
                        env_state=None)
            a_hat, is_pad_hat, (mu, logvar) = out["vae_output"]
            # a_hat = batch["traj"]["source_glb_pos_ori"]["action"][:, :, ::2].float().to(args.device)
            a_hat = a_hat.detach()
            dim_pad = torch.zeros_like(a_hat[..., 0])
            dim_pad1 = torch.ones_like(a_hat[..., 0])
            all_action = torch.stack([a_hat[..., 0], dim_pad, a_hat[..., 1], dim_pad, a_hat[..., 2], dim_pad], dim=-1)
            if inference_type == "direct_vel" or inference_type == "real_delta_target":
                all_action = self.normalizer.denormalize(all_action, action_type)
                all_action = torch.stack([all_action[..., 0], dim_pad, all_action[..., 2], dim_pad, all_action[..., 4] * 4, dim_pad], dim=-1)
                print("all_action:", all_action)
                # continue
                base = torch.stack([real_qpos_6[..., 0], dim_pad[:, -1:], real_qpos_6[..., 2], dim_pad[:, -1:], real_qpos_6[..., 4], dim_pad[:, -1:]], dim=-1)
                base = exp_map(base.squeeze(0).squeeze(0).detach().cpu().numpy(), np.array([0, 0, 0, 0, 1, 0, 0]))
                print(base.shape)
                v = all_action.squeeze(0).detach().cpu().numpy() * 1.8
                all_action = recover_pose_from_quat_real_delta(v, base)
            elif inference_type == "position":
                all_action = self.normalizer.denormalize(all_action, "source_glb_pos_ori")
                delta = all_action[:, :, ::2] - real_qpos_3
                print("delta:",delta)
                rotate = delta[:, :, 2]
                new_euler = rotate + real_qpos_3[:, :, 2]
                all_action = torch.stack([all_action[..., 0], dim_pad1 * 0.00742, all_action[..., 2], dim_pad1 * 0.0051700705953731585, new_euler, dim_pad1 * -0.0031471225012891704], dim=-1)
                v = all_action.squeeze(0).cpu().numpy()
                print("all action pos ori:",v)
                all_action = exp_map_seq(v, np.array([0, 0, 0, 0, 1, 0, 0]))
            buffer = all_action[:query_frequency, :] if cur_step % query_frequency == 0 else buffer  # plz make sure cur_step start with 0
            # generate message
            start = time.time()
            pose_new = buffer[cur_step % query_frequency]
            print("pose_new_pos_ori:", log_map(pose_new, np.array([0, 0, 0, 0, 1, 0, 0])))
            test_list2.append(log_map(pose_new, np.array([0, 0, 0, 0, 1, 0, 0])))
            rpc.goto_cartesian_pose_blocking(pose_new[:3], pose_new[3:], False)
            loop_time = time.time() - cur_t
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

            # if gripper_state >= GRIPPER_OPEN_STATE:  # if open
            #     if gripper_command < GRIPPER_OPEN_STATE:
            #         rpc.close_gripper()
            # elif gripper_state < GRIPPER_OPEN_STATE:  # if close
            #     if gripper_command > GRIPPER_OPEN_STATE:
            #         rpc.open_gripper()

            print("iter time:", time.time() - cur_t)
            cur_step += 1
            ################ end model computation ########################################################


    def realsense_process(self, realsense_config, image_q):
        cache_imgs = deque(maxlen=50)
        logger.debug("Realsense process started")
        logger.debug("Starting realsense recorder ...")
        rs_recorder = RealsenseRecorder(
            **realsense_config,
        )
        rs_recorder._make_clean_folder(realsense_config["record_folder"])
        logger.debug("... realsense recorder intialized")
        while not self.finish_event.is_set():
            try:
                t0 = time.time()
                realsense_frames = rs_recorder.get_frame()
                t1 = time.time()
                to_plot = rs_recorder._visualize_frame(realsense_frames)[:, :, ::-1].copy()
                # print(f"deep copy time: {time.time() - t1}")
                image_q.put((to_plot, time.time()))
                # print(f"total image time: {time.time() - t0}")
            except (KeyboardInterrupt, SystemExit):
                break

        del rs_recorder

    def mic_process(self, audio_q):
        cache_audio = deque(maxlen=50)
        logger.debug("microphone process started")

        logger.debug("Starting microphone recorder ...")

        mic_idx = 6
        sr = 44100
        fps = 30
        CHUNK = int(sr / fps)
        p = pyaudio.PyAudio()
        audio_stream = p.open(format=pyaudio.paInt16,
                              channels=1,
                              rate=sr,
                              input=True,
                              input_device_index=mic_idx,  # Corrected variable name to microphone_index
                              frames_per_buffer=CHUNK)

        logger.debug("... microphone recorder intialized")
        while not self.finish_event.is_set():
            try:
                audio_frames = audio_stream.read(CHUNK, exception_on_overflow=False)
                audio_q.put((copy.deepcopy(audio_frames), time.time()))
                # print("mic")
            except (KeyboardInterrupt, SystemExit):
                break

        del audio_stream

    def target_robot_process(self, target_state_q):
        cache_pose = deque(maxlen=50)
        logger.debug(f"Real robot process started with RPC @ {self.target_ip}")
        rpc = RPCInterface(self.target_ip)
        while not self.finish_event.is_set():
            arm_state = rpc.get_robot_state()
            pose_loc = arm_state.pose.vec(quat_order="wxyz")
            qpos = log_map(pose_loc, np.array([0, 0, 0, 0, 1, 0, 0]))
            gripper_state = arm_state.gripper[0]
            target_state_q.put((qpos, gripper_state, time.time()))
            # print("robo")

    def gather_obs_image(self, target_state_q, image_q, audio_q, obs_qpos, obs_image, obs_audio):
        cache_img = deque(maxlen=50)
        while not self.finish_event.is_set():
            try:
                start_time = time.time()
                while image_q.qsize() > 0:
                    img = image_q.get()
                    cache_img.append(img)
                    # print(f"len imgs {len(img)}")

                obs_image.put(cache_img)
                print(f"get img obs time:", time.time() - start_time, "\n")


            except (KeyboardInterrupt, SystemExit):
                break



    def gather_obs_audio(self, target_state_q, image_q, audio_q, obs_qpos, obs_image, obs_audio):
        cache_audio = deque(maxlen=50)
        while not self.finish_event.is_set():
            try:
                start_time = time.time()
                while audio_q.qsize() > 0:
                    aud = audio_q.get()
                    cache_audio.append(aud)
                    # print(f"len aud {len(aud)}")
                obs_audio.put(cache_audio)
                # print(f"get audio obs time:", time.time() - start_time, "\n")

            except (KeyboardInterrupt, SystemExit):
                break

    def gather_obs_qpos(self, target_state_q, image_q, audio_q, obs_qpos, obs_image, obs_audio):
        cache_qpos = deque(maxlen=50)

        while not self.finish_event.is_set():
            try:
                start_time = time.time()
                while target_state_q.qsize() > 0:
                    state_loc = target_state_q.get()
                    cache_qpos.append(state_loc)
                    # print(f"len state loc {len(state_loc)}")
                obs_qpos.put(cache_qpos)
                # print(f"get qpos obs time:", time.time() - start_time, "\n")


            except (KeyboardInterrupt, SystemExit):
                break

    def action_execution(self, action_q):
        while not self.finish_event.is_set():
            try:

                while action_q.qsize() > 0:
                    aud = action_q.get()
                    cache_action.append(aud)
                    # print(f"len aud {len(aud)}")
                obs_audio.put(cache_audio)
                # print(f"get audio obs time:", time.time() - start_time, "\n")

                out_chunk = np.concatenate([gripper, out_chunk, ], axis=-1)
                out_position = torch.from_numpy(out_chunk[:, :4])
                out_orientation = torch.from_numpy(out_chunk[:, 4:])
                all_time_orientation[[t], t:t + num_queries] = out_orientation.float().to(args.device)
                orientation_for_curr_step = all_time_orientation[:, t]
                actions_populated = torch.all(orientation_for_curr_step != 0, axis=1)
                orientation_for_curr_step = orientation_for_curr_step[actions_populated]

                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(orientation_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = (exp_weights[::-1]).copy()  # [::-1] could lead to negative strides

                weights = np.expand_dims(exp_weights, axis=0)
                raw_orientation = orientation_for_curr_step[0].detach().cpu().numpy()
                orientation = orientation_for_curr_step.permute(1, 0).detach().cpu().numpy()
                for i in range(5):
                    tangent_space_vector = q_log_map(orientation, raw_orientation)
                    tangent_space_vector = np.sum(tangent_space_vector * weights, axis=1, keepdims=True)
                    raw_orientation = q_exp_map(tangent_space_vector, raw_orientation)[:, 0]

                all_time_position[[t], t:t + num_queries] = out_position.float().to(args.device)
                position_for_curr_step = all_time_position[:, t]
                actions_populated = torch.all(position_for_curr_step != 0, axis=1)
                position_for_curr_step = position_for_curr_step[actions_populated]
                weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                raw_action = (position_for_curr_step * weights).sum(dim=0, keepdim=True)
                raw_position = raw_action.squeeze(0).cpu().numpy()


            except (KeyboardInterrupt, SystemExit):
                break



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

    transform_cam = T.Compose([T.Resize((cfg.datasets.dataloader.args.resized_height_v,
                                         cfg.datasets.dataloader.args.resized_width_v), antialias=None), ])

    train_loaders, val_loaders, train_inference_loaders = get_loaders(**cfg.datasets.dataloader, debug=True,
                                                                      load_json=os.path.join(cfg_path,
                                                                                             "normalizer_config.json"))

    demo_config = {
        "target_robot_ip": "10.87.170.254",
        "resampling_time": 0.1,
        "resolution": (640, 480),
        "exposure_time": 40,
        "image_freqeuncy": 30,
        "model": model,
        "normalizer": normalizer,
        "transform_cam": transform_cam,
        "loader": val_loaders,
    }

    inference_mdl = Inference(**demo_config)
    inference_mdl.start_demo_recording_inference("/tmp/robot_demos/")


if __name__ == "__main__":

    project_path = os.path.abspath(os.path.join(__file__, '..', '..'))
    sys.path.append(
        project_path)  # without this line:hydra.errors.InstantiationException: Error locating target
    # 'src.datasets.bautiro_drilling_dataset.get_loaders', set env var HYDRA_FULL_ERROR=1 to see chained exception.
    # full_key: datasets.dataloader

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        default='../checkpoints/pouring/name=aloha_pouringname=vae_resnet_qposaction=positionname=coswarmuplr=4e-05source=Trueresized_height_v=240resized_width_v=320batch_size=64_05-01-19:18:15')
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
