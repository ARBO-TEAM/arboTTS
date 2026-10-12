"""Acoustic Morphing & Mastering Tool for ArboTTS Speaker Profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import butter, sosfilt
import soundfile as sf


def load_profiles(config_path: Path | str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_presence_boost(audio: np.ndarray, sr: int, freq_hz: float = 3500, gain: float = 0.25) -> np.ndarray:
    sos = butter(2, freq_hz, "highpass", fs=sr, output="sos")
    boost = sosfilt(sos, audio) * gain
    return audio + boost


def apply_lowpass_warmth(audio: np.ndarray, sr: int, freq_hz: float = 5500) -> np.ndarray:
    sos = butter(2, freq_hz, "lowpass", fs=sr, output="sos")
    return sosfilt(sos, audio)


def peak_normalize(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    m = np.max(np.abs(audio))
    return (audio / m * target_peak) if m > 1e-4 else audio


def morph_audio(
    audio: np.ndarray,
    sr: int,
    recipe: dict,
) -> tuple[np.ndarray, int]:
    pitch_semitones = recipe.get("pitch_shift_semitones", 0.0)
    if abs(pitch_semitones) > 1e-3:
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_semitones)

    if "presence_boost_hz" in recipe:
        audio = apply_presence_boost(
            audio,
            sr,
            freq_hz=recipe["presence_boost_hz"],
            gain=recipe.get("presence_gain", 0.25),
        )

    if "lowpass_warmth_hz" in recipe:
        audio = apply_lowpass_warmth(
            audio,
            sr,
            freq_hz=recipe["lowpass_warmth_hz"],
        )

    audio = peak_normalize(audio, target_peak=0.95)
    return audio, sr


def main() -> None:
    parser = argparse.ArgumentParser(description="Morph input audio to an ArboTTS speaker profile.")
    parser.add_argument("input_audio", type=Path, help="Input WAV/MP3 path")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output WAV path")
    parser.add_argument("-s", "--speaker", type=str, default="arba", choices=["arbo", "arba", "sita", "maya"])
    parser.add_argument("-c", "--config", type=Path, default=Path(__file__).resolve().parent.parent / "configs" / "speakers.json")

    args = parser.parse_args()
    profiles = load_profiles(args.config)

    if args.speaker not in profiles.get("speakers", {}):
        raise ValueError(f"Speaker '{args.speaker}' not found in {args.config}")

    recipe = profiles["speakers"][args.speaker]["dsp_recipe"]
    target_sr = profiles.get("sampling_rate", 22050)

    audio, sr = librosa.load(str(args.input_audio), sr=target_sr)
    morphed, out_sr = morph_audio(audio, sr, recipe)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), morphed, out_sr, subtype="PCM_16")
    print(f"Morphed [{args.input_audio.name}] -> [{args.speaker}] ({args.output})")


if __name__ == "__main__":
    main()
