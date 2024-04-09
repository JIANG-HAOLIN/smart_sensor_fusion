import os

import cv2
from cv2 import VideoWriter
from datetime import datetime
import os


def collect_from_webcam():
    fs = []
    webcam = cv2.VideoCapture(4)  # 0 for hd cam 2 for ir cam 4 for logi cam

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


if __name__ == "__main__":
    fs = collect_from_webcam()
    save_video(30, "test_cam_save", fs)
