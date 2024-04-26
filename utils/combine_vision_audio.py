import os
import wave
import cv2
import numpy as np
from datetime import datetime
import subprocess
import soundfile as sf
### use subprocess to run outside command in terminal

def demo2video(folder="/home/jin4rng/Documents/robodemo_4_24/demo_2024-04-24T17-39-18-7779"):
    cur_t = datetime.now().strftime("%m-%d-%H:%M:%S")
    audio_path = os.path.join(folder + "/audio.wav")
    sound = sf.read(audio_path)
    a = wave.open(audio_path, "rb")
    print(a.getparams())
    audio_len = a.getparams().nframes
    audio_data = a.readframes(nframes=audio_len)
    audio_data = np.frombuffer(audio_data, dtype=np.int16)
    audio_data = audio_data.reshape(audio_len, 1)
    max, min = np.max(audio_data), np.min(audio_data)
    audio_data = (audio_data - min) / (max - min) * 2 - 1
    audio_data = audio_data.copy().astype(np.float32)
    audio_data = (audio_data + 1) / 2 * (max - min) + min

    print(audio_data.shape)

    image_folder = os.path.join(folder, "resampled_camera/220322060186/rgb/")
    image_list = []
    ims = sorted(os.listdir(image_folder))
    for im in ims:
        image_list.append(cv2.imread(image_folder + im).copy())
    frequency = 30
    video_path = os.path.join(folder, "fix_cam.mp4")
    out = cv2.VideoWriter(video_path,
                          cv2.VideoWriter_fourcc(*"MJPG"), frequency,
                          (image_list[0].shape[1], image_list[0].shape[0]))  # careful of the size, should be W, H
    for frame in image_list:
        out.write(frame.astype('uint8').copy())
    out.release()
    mkv_save_path = os.path.join(folder, "output.mkv")
    cmd = f'ffmpeg -i {video_path} -i {audio_path} -c copy {mkv_save_path}'  # only mkv!!
    subprocess.call(cmd, shell=True)

if __name__ == "__main__":
    demo2video()