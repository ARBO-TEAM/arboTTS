"""Dataset loading and batch collation for ArboTTS VITS training."""

from __future__ import annotations

import os
from pathlib import Path
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader

from arbotts.frontend.g2p import phonemes_to_ids
from training.mel_processing import spectrogram_torch, mel_spectrogram_torch


class TextAudioLoader(Dataset):
    """Loads audio waveforms and phoneme texts from train_pilot.txt."""

    def __init__(self, metadata_path: str, root_dir: str = "D:/kerjaan/belajar/tts/arboTTS") -> None:
        self.root_dir = Path(root_dir)
        self.items = []
        with open(self.root_dir / metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    wav_rel, phonemes = parts[0], parts[1]
                    wav_full = self.root_dir / wav_rel
                    if wav_full.is_file():
                        self.items.append((str(wav_full), phonemes))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wav_path, phonemes = self.items[index]
        # 1. Phonemes to IDs
        text_ids = phonemes_to_ids(phonemes)
        if len(text_ids) == 0:
            text_ids = [1]  # pad/blank fallback
        text_tensor = torch.tensor(text_ids, dtype=torch.long)

        # 2. Audio loading
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)

        # 3. Spectrogram extraction
        spec = spectrogram_torch(
            audio_tensor,
            n_fft=1024,
            sampling_rate=22050,
            hop_size=256,
            win_size=1024,
            center=False,
        ).squeeze(0)

        return text_tensor, spec, audio_tensor.squeeze(0)


class TextAudioCollate:
    """Zero-pads phonemes, spectrograms, and waveforms into uniform batch tensors."""

    def __call__(self, batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, ...]:
        # Filter empty
        batch = [b for b in batch if b[0].size(0) > 0 and b[1].size(1) > 0]
        if not batch:
            return ()

        _, ids_sorted_desc = torch.sort(
            torch.tensor([x[1].size(1) for x in batch], dtype=torch.long),
            descending=True,
        )

        max_text_len = max([x[0].size(0) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(0) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))

        text_padded = torch.zeros(len(batch), max_text_len, dtype=torch.long)
        spec_padded = torch.zeros(len(batch), batch[0][1].size(0), max_spec_len, dtype=torch.float32)
        wav_padded = torch.zeros(len(batch), max_wav_len, dtype=torch.float32)

        for i in range(len(ids_sorted_desc)):
            idx = ids_sorted_desc[i]
            text, spec, wav = batch[idx]

            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav_padded[i, :wav.size(0)] = wav
            wav_lengths[i] = wav.size(0)

        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths
