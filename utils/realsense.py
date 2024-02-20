# pyrealsense2 is required.
# Please see instructions in https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python
import copy
import json
import os
import shutil
from enum import IntEnum
from os import makedirs
from os.path import exists

import cv2
import numpy as np
import pyrealsense2 as rs
from tqdm import tqdm


class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5


def colorize_depth(img: np.ndarray, return_as_uint8=True) -> np.ndarray:
    """Creats a colored RGB heatmap from input image
    Args:
        img (np.ndarray): heatmap with values in [0, 1], size (H, W)
    Returns:
        np.ndarray: colored heatmap (cold -> red), size (H, W, 3)
    """
    img_loc = np.asarray(img / img.max(), dtype=float)
    heat_map = np.zeros((img.shape[0], img.shape[1], 3))
    heat_map[:, :, 2] = (img_loc - 1) ** 2
    heat_map[:, :, 1] = -((2 * img_loc - 1) ** 2) + 1
    heat_map[:, :, 0] = img_loc**2
    u, v = np.where(img_loc == 0.0)
    heat_map[u, v, :] = 0.0
    if return_as_uint8:
        return np.asarray(heat_map * 255.0, dtype=np.uint8)
    else:
        return heat_map


class RealsenseRecorder:
    """Multi-Realsense-camera recorder class
    Args:
        height (int): the height of the recorded images
        width (int): the width of the recorded images
        fps (int): the framerate of the recording
        record_depth (bool): to record the depth images, or only RGB
        depth_unit (float): the resolution of the depth (0.001 => 1 mm)
        preset (int): camera preset setting
        memory_first (bool): if you intend to save frames to hard drive it is a good idea to record first to memory and then write to file at the end
        record_folder (str): to where the images should be recorded
    Functionalities:
        get_frame(): records a single frame from all cameras
        run_and_visualize(): visualizes all the camera streams
        run_and_stroe(): visualizes and saves files
    """

    def __init__(
        self,
        height: int = 480,
        width: int = 640,
        fps: int = 30,
        record_depth: bool = True,
        depth_unit: float = 0.001,
        preset: int = Preset.HighAccuracy,
        memory_first: bool = True,
        record_folder: str = "/tmp/realsense_recording",
    ):
        # Generic camera settings
        self._h = height
        self._w = width
        self._fps = fps
        self._depth_unit = depth_unit
        self._preset = preset
        self._record_depth = record_depth

        # Settings related to recording to folder
        self._memory_first = memory_first
        self._record_folder = record_folder

        self._pipelines = {}
        self._depth_scales = {}
        self._setup_pipelines()

    def __del__(self):
        """Close all the pipelines"""
        if len(self._pipelines) > 0:
            for p_key in list(self._pipelines.keys()):
                self._pipelines[p_key].stop()

    def _setup_pipelines(self):
        """Set up all the camera pipelines"""
        ctx = rs.context()
        self._serials = []
        num_devices = len(ctx.devices)

        for i in range(num_devices):
            sn = ctx.devices[i].get_info(rs.camera_info.serial_number)
            self._serials.append(sn)

        pipelines = {}
        for i in range(num_devices):
            # Create a pipeline
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(self._serials[i])
            config.enable_stream(rs.stream.color, self._w, self._h, rs.format.bgr8, self._fps)
            if self._record_depth:
                config.enable_stream(rs.stream.depth, self._w, self._h, rs.format.z16, self._fps)

            profile = pipeline.start(config)

            sensor = pipeline.get_active_profile().get_device().query_sensors()[1]
            sensor.set_option(rs.option.enable_auto_exposure, 1)

            if self._record_depth:
                depth_sensor = profile.get_device().first_depth_sensor()
                depth_sensor.set_option(rs.option.depth_units, self._depth_unit)
                depth_sensor.set_option(rs.option.visual_preset, self._preset)
                self._depth_scales[self._serials[i]] = depth_sensor.get_depth_scale()

            pipelines[self._serials[i]] = pipeline

        self._align = rs.align(rs.stream.color)
        self._pipelines = pipelines

    def get_num_cameras(self):
        return len(self._pipelines)

    def get_serial_numbers(self):
        return self._serials

    def _make_clean_folder(self, path_folder):
        """Setup the folders to where the images will be stored"""
        if not exists(path_folder):
            makedirs(path_folder)
        else:
            user_input = input("%s not empty. Overwrite? (y/n) : " % path_folder)
            if user_input.lower() == "y":
                shutil.rmtree(path_folder)
                makedirs(path_folder)
            else:
                exit()

        for s in self._serials:
            makedirs(os.path.join(path_folder, s))
            makedirs(os.path.join(path_folder, s, "rgb"))
            if self._record_depth:
                makedirs(os.path.join(path_folder, s, "depth"))

    def _save_frame(self, frames, folder, id):
        """Write a single image from all the camera frames to the specific folders"""
        for s in self._serials:
            serial_folder = os.path.join(folder, s)
            rgb_folder = os.path.join(serial_folder, "rgb")
            depth_folder = os.path.join(serial_folder, "depth")
            if id == 0:
                self._save_intrinsic_as_json(os.path.join(serial_folder, "intrinsics.json"), frames[s]["intrinsic"])
                self._save_intrinsic_as_dso_calib(
                    os.path.join(serial_folder, "intrinsics_dso.txt"), frames[s]["intrinsic"]
                )
            cv2.imwrite("%s/%06d.jpg" % (rgb_folder, id), frames[s]["rgb"])
            if self._record_depth:
                cv2.imwrite("%s/%06d.png" % (depth_folder, id), frames[s]["depth"])

    def _save_intrinsic_as_json(self, filename, intrinsics):
        """Saves the intrinsic parameters as a json file"""
        with open(filename, "w") as outfile:
            obj = json.dump(
                {
                    "width": intrinsics["width"],
                    "height": intrinsics["height"],
                    "intrinsic_matrix": [
                        intrinsics["fx"],
                        0,
                        0,
                        0,
                        intrinsics["fy"],
                        0,
                        intrinsics["cx"],
                        intrinsics["cy"],
                        1,
                    ],
                },
                outfile,
                indent=4,
            )

    def _save_intrinsic_as_dso_calib(self, filename, intrinsics):
        """Saves the intrinsic parameters to a text file, which is compatible with DSO"""
        to_write = f'{intrinsics["fx"]:.3f} {intrinsics["fy"]:.3f} {intrinsics["cx"]:.3f} {intrinsics["fy"]:.3f} 0.0\n'
        to_write += f'{intrinsics["width"]} {intrinsics["height"]}\ncrop\n{intrinsics["width"]} {intrinsics["height"]}'
        with open(filename, "w") as f:
            f.write(to_write)

    def get_frame(self):
        """Returns a single RGBD frame for all the camera frames"""
        frames = {}
        loc_frame_dict = {}
        # Get frame from every camera
        for s in self._serials:
            loc_frame_dict[s] = self._pipelines[s].wait_for_frames()
    
        for s in self._serials:
            frames[s] = {}
            # Align images in each frame
            frame = self._align.process(loc_frame_dict[s])
            color_frame = frame.get_color_frame()
            if self._record_depth:
                aligned_depth_frame = frame.get_depth_frame()
                # Validate that both frames are valid
                if not aligned_depth_frame or not color_frame:
                    continue
            else:
                if not color_frame:
                    continue

            # If everything is good => save as normal images
            frames[s]["rgb"] = np.asanyarray(color_frame.get_data())
            if self._record_depth:
                frames[s]["depth"] = np.asanyarray(aligned_depth_frame.get_data()) * self._depth_scales[s]

            intrinsics = frame.profile.as_video_stream_profile().intrinsics
            intrinsic = {
                "width": intrinsics.width,
                "height": intrinsics.height,
                "fx": intrinsics.fx,
                "fy": intrinsics.fy,
                "cx": intrinsics.ppx,
                "cy": intrinsics.ppy,
                "intrinsic_matrix": np.array(
                    [[intrinsics.fx, 0, intrinsics.ppx], [0, intrinsics.fy, intrinsics.ppy], [0.0, 0.0, 1.0]]
                ),
            }
            frames[s]["intrinsic"] = intrinsic
        return frames

    def _visualize_frame(self, frames):
        """returns an 8-bit numpy array image for visualization"""
        num_devices = self.get_num_cameras()
        num_img = 2 if self._record_depth else 1
        to_plot = np.zeros((num_img * self._h, num_devices * self._w, 3), dtype=np.uint8)
        for i, s in enumerate(self._serials):
            to_plot[: self._h, i * self._w : (i + 1) * self._w, :] = frames[s]["rgb"]
            if self._record_depth:
                to_plot[self._h :, i * self._w : (i + 1) * self._w, :] = colorize_depth(frames[s]["depth"])
        return to_plot

    def run_and_visualize(self, store: bool = False, store_folder: str = ""):
        """Visualizes all the camera streams"""
        print("Press ESC or q to stop visualizing (and recording)!")

        if store:
            if len(store_folder) == 0:
                store_folder = self._record_folder
            self._make_clean_folder(store_folder)

        frames_list = []
        frame_id = 0
        while True:
            frames = self.get_frame()
            to_plot = self._visualize_frame(frames)

            if store:
                if self._memory_first:
                    frames_list.append(copy.deepcopy(frames))
                else:
                    self._save_frame(frames, store_folder, frame_id)
                    frame_id += 1

            cv2.imshow("RealsenseRecorder", to_plot)
            key = cv2.waitKey(1)

            # if 'esc' button pressed, escape loop and exit program
            if key == 27 or key == ord("q"):
                cv2.destroyAllWindows()
                break

        if self._memory_first and store:
            pbar = tqdm(range(len(frames_list)), desc="Writing images to file ...")
            for frame_count in pbar:
                self._save_frame(frames_list[frame_count], store_folder, frame_count)
        if store:
            print(f"Files saved under {store_folder}")

    def run_and_save(self, store_folder: str = ""):
        """Visualizes and saves all the camera streams"""
        self.run_and_visualize(True, store_folder)


if __name__ == "__main__":

    recorder = RealsenseRecorder(
        height=480,
        width=640,
        fps=30,
        record_depth=False,
        depth_unit=0.001,
        preset=Preset.HighAccuracy,
        memory_first=True,
        record_folder="/tmp/dummy",
    )
    recorder.run_and_save()
    # recorder.run_and_visualize()