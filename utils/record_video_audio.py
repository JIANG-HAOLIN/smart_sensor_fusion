import cv2
import pyaudio
import pyaudio
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import cv2
import librosa
from threading import Thread
import time
import wave


def list_devices():
    print("Available devices:")
    audio = pyaudio.PyAudio()
    devices = audio.get_device_count()
    for i in range(devices):
        device_info = audio.get_device_info_by_index(i)
        print(
            f"#{i}, in/out: {device_info.get('maxInputChannels')}/{device_info.get('maxOutputChannels')}, name: {device_info.get('name')}")



def choose_device(device_type):
    index = int(input(f"Choose {device_type} device (enter the device number): "))
    return index


def record_video_audio(output_file, webcam_index, microphone_index):
    cap = cv2.VideoCapture(webcam_index)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (640, 480))

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=44100,
                    input=True,
                    input_device_index=microphone_index,  # Corrected variable name to microphone_index
                    frames_per_buffer=1024)

    print("Recording video and audio. Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Video", frame)
        out.write(frame)

        data = stream.read(1024)
        # out.write(data)  # Save audio data along with video frames

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    list_devices()

    webcam_index = choose_device("webcam")
    microphone_index = choose_device("microphone")

    output_file = "data_collection.mp4"
    record_video_audio(output_file, webcam_index, microphone_index)

    print(f"Video and audio recorded successfully. Saved as {output_file}")
