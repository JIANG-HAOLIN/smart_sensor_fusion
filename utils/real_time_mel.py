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
DEVICE = [6]
# if no connected micro: 15/16 if conected micro: 14/16/17
audio = pyaudio.PyAudio()
CHUNK_TIME = CHUNK / (RATE * 1.0)
frame_size = int(RECORD_SECONDS / CHUNK_TIME)


class AudioRecorder:

    def __init__(self):
        self.streams = {}
        self.frames = {}
        for device_id in DEVICE:
            self.streams[device_id] = audio.open(format=FORMAT, channels=CHANNELS,
                                                 rate=RATE, input=True, frames_per_buffer=CHUNK,
                                                 input_device_index=device_id)
            self.frames[device_id] = []
        self.max_frames = RATE / CHUNK * RECORD_SECONDS

    def read_chunk(self):
        while True:
            for device_id in DEVICE:
                data = self.streams[device_id].read(CHUNK)
                self.frames[device_id].append(data)

            # if len(self.frames) > self.max_frames:
            #     self.frames.pop(0)
            # print("updated chunk")


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


rec = AudioRecorder()

recording_thread = Thread(target=rec.read_chunk, daemon=True)
recording_thread.start()
analysis_thread = Thread(target=frames_analyser)
analysis_thread.start()
analysis_thread.join()

for device_id in DEVICE:
    filename = f"recorded_audio_source_{device_id}.wav"
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    # print(audio.get_format_from_width(FORMAT).size)
    wf.setsampwidth(2)
    wf.setframerate(RATE)
    wf.writeframes(b''.join(rec.frames[device_id]))
    wf.close()

print("done")