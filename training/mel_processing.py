"""Spectrogram and mel-spectrogram extraction in PyTorch."""

from __future__ import annotations

import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn

# Cache mel basis across calls
_MEL_BASIS = {}
_HANN_WINDOW = {}


def dynamic_range_compression_torch(x: torch.Tensor, C: float = 1.0, clip_val: float = 1e-5) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x: torch.Tensor, C: float = 1.0) -> torch.Tensor:
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    return dynamic_range_compression_torch(magnitudes)


def spectrogram_torch(
    y: torch.Tensor,
    n_fft: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    center: bool = False,
) -> torch.Tensor:
    """Extract linear spectrogram [B, n_fft // 2 + 1, T] from waveform [B, T]."""
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global _HANN_WINDOW
    dtype_device = f"{y.dtype}_{y.device}"
    wnsize_dtype_device = f"{win_size}_{dtype_device}"
    if wnsize_dtype_device not in _HANN_WINDOW:
        _HANN_WINDOW[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    pad_size = int((n_fft - hop_size) / 2)
    y = torch.nn.functional.pad(
        y.unsqueeze(1), (pad_size, pad_size), mode="reflect"
    ).squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=_HANN_WINDOW[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


def spec_to_mel_torch(
    spec: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    fmin: float,
    fmax: float | None,
) -> torch.Tensor:
    """Convert linear spectrogram to mel spectrogram."""
    global _MEL_BASIS
    dtype_device = f"{spec.dtype}_{spec.device}"
    fmax_dtype_device = f"{n_fft}_{num_mels}_{sampling_rate}_{fmin}_{fmax}_{dtype_device}"
    if fmax_dtype_device not in _MEL_BASIS:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        _MEL_BASIS[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=spec.dtype, device=spec.device
        )
    mel_spec = torch.matmul(_MEL_BASIS[fmax_dtype_device], spec)
    mel_spec = spectral_normalize_torch(mel_spec)
    return mel_spec


def mel_spectrogram_torch(
    y: torch.Tensor,
    n_fft: int = 1024,
    num_mels: int = 80,
    sampling_rate: int = 22050,
    hop_size: int = 256,
    win_size: int = 1024,
    fmin: float = 0.0,
    fmax: float | None = None,
    center: bool = False,
) -> torch.Tensor:
    """Extract mel spectrogram directly from audio waveform."""
    spec = spectrogram_torch(
        y, n_fft, sampling_rate, hop_size, win_size, center=center
    )
    mel = spec_to_mel_torch(
        spec, n_fft, num_mels, sampling_rate, fmin, fmax
    )
    return mel
