import os

import cv2
from cv2 import VideoWriter
from datetime import datetime
import os
from multiprocessing import Process
import wave
import pyaudio
import numpy as np
def collect_from_webcam():
    fs = []
    webcam = cv2.VideoCapture(0)  # 0 for hd cam 2 for ir cam 4 for logi cam

    while True:
        stream_ok, frame = webcam.read()

        if stream_ok:
            cv2.imshow('webcam', frame)
            frame = frame[:, :, ::-1].copy()
            fs.append(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    webcam.release()
    return fs


def save_video(frequency, folder_name, frame_list, parent_path="./"):
    ## input should be in [H, W, RGB] form!!
    save_folder_path = os.path.join(parent_path, folder_name)
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    out = cv2.VideoWriter(f'{save_folder_path}/{datetime.now().strftime("%m-%d-%H:%M:%S")}_{int(frequency)}HZ.avi',
                          cv2.VideoWriter_fourcc(*"MJPG"), frequency,
                          (frame_list[0].shape[1], frame_list[0].shape[0]))  # careful of the size, should be W, H
    for frame in frame_list:
        out.write(frame[:, :, ::-1].astype('uint8').copy())
    out.release()


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


def record_audio():
    mic_frame_list = []
    while True:
        audio_frames = audio_stream.read(CHUNK, exception_on_overflow=False)
        mic_frame_list.append(audio_frames)

        cv2.imshow("test_cam_save", np.zeros((1, 1)))
        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            break

    wf = wave.open("/tmp/dummy/" + "mic.wav", 'wb')
    wf.setnchannels(1)
    # print(audio.get_format_from_width(FORMAT).size)
    wf.setsampwidth(2)
    wf.setframerate(44100)
    wf.writeframes(b''.join(mic_frame_list))
    wf.close()

if __name__ == "__main__":
    fs = collect_from_webcam()
    a = Process(target=save_video, args=(30, "test_cam_save", fs))
    b = Process(target=record_audio, args=())
    # a.start()
    b.start()
    # a.join()
    # b.join()
