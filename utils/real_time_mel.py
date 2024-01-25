import pyaudio
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import cv2
import librosa
from threading import Thread
import time
import wave

# Initialize pyaudio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 4096
RECORD_SECONDS = 4
FMAX = 8000
MEL_HOP = 256
MEL_BINS = 64

DEVICE = [15]
# if no connected micro: 15/16 if conected micro: 14/16/17
audio = pyaudio.PyAudio()
CHUNK_TIME = CHUNK / (RATE * 1.0)
frame_size = int(RECORD_SECONDS / CHUNK_TIME)


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


def record_video(output_file, webcam_index):
    cap = cv2.VideoCapture(webcam_index)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (640, 480))

    print("Recording video and audio. Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Video", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


class VideoRecorder:

    def __init__(self, webcam_index, output_file):
        self.cap = cv2.VideoCapture(webcam_index)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_file, fourcc, 20.0, (640, 480))

    def read_frame(self):
        print("Recording video and audio. Press 'q' to stop.")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            cv2.imshow("Video", frame)
            self.out.write(frame)

            if cv2.waitKey(1) == ord('q'):
                break

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()


class AudioRecorder:

    def __init__(self, device, format, channels, rate, chunk, record_second):
        self.streams = {}
        self.frames = {}
        self.device = device
        self.chunk = chunk
        for device_id in device:
            self.streams[device_id] = audio.open(format=format, channels=channels,
                                                 rate=rate, input=True, frames_per_buffer=chunk,
                                                 input_device_index=device_id)
            self.frames[device_id] = []
        self.max_frames = rate / chunk * record_second

    def read_chunk(self):
        while True:
            for device_id in self.device:
                data = self.streams[device_id].read(self.chunk)
                self.frames[device_id].append(data)

            # if len(self.frames) > self.max_frames:
            #     self.frames.pop(0)
            # print("updated chunk")

            # if cv2.waitKey(1) == ord('q'):
            #     break


class VARecorder:

    def __init__(self, output_file, device, format, channels, rate, chunk, record_second):
        self.cap = cv2.VideoCapture(4)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_file, fourcc, 10.7666, (640, 480))
        self.streams = {}
        self.frames = {}
        self.device = device
        self.chunk = chunk
        for device_id in device:
            self.streams[device_id] = audio.open(format=format, channels=channels,
                                                 rate=rate, input=True, frames_per_buffer=chunk,
                                                 input_device_index=device_id)
            self.frames[device_id] = []
        self.max_frames = rate / chunk * record_second

    def read_chunk(self):
        while True:
            for device_id in self.device:
                data = self.streams[device_id].read(self.chunk, exception_on_overflow=False)
                self.frames[device_id].append(data)
            ret, frame = self.cap.read()
            if not ret:
                break

            cv2.imshow("Video", frame)
            self.out.write(frame)

            if cv2.waitKey(1) == ord('q'):
                break

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

        # if len(self.frames) > self.max_frames:
        #     self.frames.pop(0)
        # print("updated chunk")

        # if cv2.waitKey(1) == ord('q'):
        #     break


def frames_analyser():
    while True:
        for device_id in DEVICE:
            if len(rec.frames[device_id]) > frame_size:
                audio_data = np.frombuffer(b''.join(rec.frames[device_id][-frame_size:]), np.int16)
                # print(audio_data.max(), audio_data.min())
                # plt.figure()
                # plt.plot(audio_data)
                # plt.show()
                audio_data_f = librosa.util.buf_to_float(audio_data)
                S = librosa.feature.melspectrogram(y=audio_data_f, sr=RATE, n_mels=MEL_BINS, fmax=FMAX,
                                                   hop_length=MEL_HOP)
                S_db = librosa.power_to_db(S, ref=np.max)
                S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min())
                cv2.imshow(f"Mel spectogram of source {device_id}", np.flipud(S_db))
                key = cv2.waitKey(1)
                if key == ord("q"):
                    return


if __name__ == "__main__":
    list_devices()

    # webcam_index = choose_device("webcam")
    # microphone_index = choose_device("microphone")

    output_file = "data_collection.mp4"
    #
    # rec = AudioRecorder(device=DEVICE,
    #                     format=FORMAT,
    #                     channels=CHANNELS,
    #                     rate=RATE,
    #                     chunk=CHUNK,
    #                     record_second=RECORD_SECONDS)
    # vrec = VideoRecorder(webcam_index, output_file)
    rec0 = VARecorder(output_file,
                      device=DEVICE,
                      format=FORMAT,
                      channels=CHANNELS,
                      rate=RATE,
                      chunk=CHUNK,
                      record_second=RECORD_SECONDS)
    # image_thread = Thread(target=vrec.read_frame)
    # image_thread.start()
    # recording_thread = Thread(target=rec.read_chunk, daemon=True)
    # recording_thread.start()
    # image_thread.join()
    # analysis_thread = Thread(target=frames_analyser)
    # analysis_thread.start()
    # analysis_thread.join()
    rec0.read_chunk()
    for device_id in DEVICE:
        filename = f"recorded_audio_source_{device_id}.wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        # print(audio.get_format_from_width(FORMAT).size)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(b''.join(rec0.frames[device_id]))
        wf.close()

    print(f"Video and audio recorded successfully. Saved as {output_file}")
