import numpy as np
import os


def get_pseudo_image_index(traj_path):
    # get pseudo time stamp from resampled real time stamp and original real time stamps
    resample_time_step = np.expand_dims(np.loadtxt(os.path.join(traj_path, f"resampled_image_time_stamps.txt")), axis=1)
    og_time_step = np.expand_dims(np.loadtxt(os.path.join(traj_path, f"image_time_stamps.txt")), axis=0)
    og_time_step = np.tile(og_time_step, (resample_time_step.shape[0], 1))
    diff = np.abs(og_time_step - resample_time_step)
    resample_pseudo_time_stamps = np.argmin(diff, axis=1, keepdims=False)
    return resample_pseudo_time_stamps