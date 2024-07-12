import os
import wave
import cv2
import numpy as np
from datetime import datetime
import subprocess
import soundfile as sf
import matplotlib.pyplot as plt

from PIL import Image
import torch, torchaudio



def show_images(folder="/fs/scratch/rb_bd_dlp_rng-dl01_cr_ROB_employees/students/jin4rng/data/6_6_rotate_cup/demo_2024-06-06T16-46-47-091936", cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    image_folder = os.path.join(folder, "camera/220322060186/resample_rgb/")
    images_path = sorted(os.listdir(image_folder))
    num_images = len(images_path)
    images = [np.array(Image.open(os.path.join(image_folder, im)).convert('RGB')) for im in images_path[110:140:num_images//36]]

    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['{:.4}'.format(f"{(i-1)*num_images}")+" s" for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)).astype("int"), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.axis("off")
    plt.show()


def visualize_wav(folder="/fs/scratch/rb_bd_dlp_rng-dl01_cr_ROB_employees/students/jin4rng/data/6_6_rotate_cup/demo_2024-06-06T15-51-19-077878"):
    cur_t = datetime.now().strftime("%m-%d-%H:%M:%S")
    audio_path = os.path.join(folder + "/audio.wav")
    sound = sf.read(audio_path)
    a = wave.open(audio_path, "rb")
    print(a.getparams())
    audio_len = a.getparams().nframes
    audio_data = a.readframes(nframes=audio_len)
    audio_data = np.frombuffer(audio_data, dtype=np.int16).astype("float64")
    audio_data = audio_data.reshape(audio_len, 1)
    max, min = np.max(audio_data), np.min(audio_data)
    audio_data_norm = (audio_data - min) / (max - min) * 2 - 1
    audio_data_norm = audio_data_norm.copy().astype(np.float64)
    # audio_data = (audio_data + 1) / 2 * (max - min) + min
    plt.figure()
    plt.subplot(211)
    plt.plot(np.arange(audio_data.shape[0])/44100, audio_data)
    plt.xlabel("Time/s")
    plt.title("Raw Audio Input")
    plt.subplot(212)
    plt.plot(np.arange(audio_data_norm.shape[0]), audio_data_norm)
    plt.title("Normalized Audio Input")
    plt.show()
    return audio_data_norm


class MelSpec(torch.nn.Module):
    """Compute mel spectrogram for audio signal"""
    def __init__(self,
                 windows_size: float = 0.025,
                 length: int = 40000,
                 sr: int = 16000,  # originally 44.1k, resampled to 16k
                 n_mels: int = 64,
                 norm_audio: bool = False,
                 hop: float = 0.01):
        """
        length - number of sample in an input sequence
        sr - sampling rate for mel spectrogram
        n_mel - Number of mel filterbanks
        norm_audio - whether do we normalize the audio signal
        """
        super().__init__()
        self.norm_audio = norm_audio
        self.n_mel = n_mels
        hop_length = int(sr * hop)
        n_fft = int(sr * windows_size)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        self.out_size = (n_mels, int(length/hop_length)+1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
                waveform - the input sound signal in waveform
        Return:
                log_spec - the logarithm form of mel spectrogram of sound signal
        """
        eps = 1e-8
        spec = self.mel(waveform.float())
        log_spec = torch.log(spec + eps)  # logarithm, eps to avoid zero denominator
        assert log_spec.size(-2) == self.n_mel
        if self.norm_audio:
            log_spec /= log_spec.sum(dim=-2, keepdim=True)
        return log_spec

if __name__ == "__main__":
    show_images()
    mel = MelSpec(windows_size=0.01, length=1246560, sr=44100, n_mels=64, hop=0.01)
    wave = visualize_wav()
    mel_spectrogram = mel(torch.from_numpy(wave).unsqueeze(0).squeeze(-1)).squeeze(0)

    mel_spectrogram_np = mel_spectrogram.numpy()

    # Plot the mel spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spectrogram_np, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Intensity (dB)')
    plt.title('Mel Spectrogram')
    plt.xlabel('Time Frame')
    plt.ylabel('Mel Filter Banks')
    plt.tight_layout()
    plt.show()