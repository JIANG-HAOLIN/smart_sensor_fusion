import torch
import torchaudio


class MelSpec(torch.nn.Module):
    """Compute mel spectrogram for audio signal"""
    def __init__(self, length: int = 40000,
                 sr: int = 16000,
                 n_mels: int = 64,
                 norm_audio: bool = False,
                 hop_ratio: float = 0.01):
        """
        length - number of sample in an input sequence
        sr - sampling rate for mel spectrogram
        n_mel - Number of mel filterbanks
        norm_audio - whether do we normalize the audio signal
        """
        super().__init__()
        self.norm_audio = norm_audio
        self.n_mel = n_mels
        hop_length = int(sr * hop_ratio)
        n_fft = int(sr * 0.025)
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
