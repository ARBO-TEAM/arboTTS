"""Speaker morphing and acoustic transformation for ArboTTS profiles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, sosfilt

try:
    import librosa
except ImportError:
    librosa = None  # type: ignore

DEFAULT_SPEAKERS = {
    "version": "1.0",
    "sampling_rate": 22050,
    "speakers": {
        "arbo": {
            "id": 0,
            "gender": "male",
            "persona": "Formal Broadcast Announcer",
            "description": "Mature, authoritative, crisp consonants and balanced resonance.",
            "dsp_recipe": {},
        },
        "arba": {
            "id": 1,
            "gender": "female",
            "persona": "Professional News Presenter",
            "description": "Clear, assertive, polite, warm mid-range articulation.",
            "dsp_recipe": {
                "pitch_shift_semitones": -0.8,
                "presence_boost_hz": 3500,
                "presence_gain": 0.25,
            },
        },
        "sita": {
            "id": 2,
            "gender": "female",
            "persona": "Smart Interactive Assistant",
            "description": "Bright, cheerful, engaging conversational tempo.",
            "dsp_recipe": {
                "pitch_shift_semitones": 1.2,
                "presence_boost_hz": 4000,
                "presence_gain": 0.25,
            },
        },
        "maya": {
            "id": 3,
            "gender": "female",
            "persona": "Calm Storyteller & Audiobook Narration",
            "description": "Gentle, relaxed, velvety warmth with soft cadence.",
            "dsp_recipe": {
                "pitch_shift_semitones": -1.4,
                "lowpass_warmth_hz": 5500,
            },
        },
    },
}


def load_profiles(config_path: Path | str | None = None) -> dict[str, Any]:
    if config_path is not None:
        p = Path(config_path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)

    # Check repository configs/speakers.json
    repo_cfg = Path(__file__).resolve().parent.parent / "configs" / "speakers.json"
    if repo_cfg.exists():
        with open(repo_cfg, "r", encoding="utf-8") as f:
            return json.load(f)

    return DEFAULT_SPEAKERS


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


def morph_audio(audio: np.ndarray, sr: int, recipe: dict[str, Any]) -> tuple[np.ndarray, int]:
    if not recipe:
        return peak_normalize(audio), sr

    pitch_semitones = recipe.get("pitch_shift_semitones", 0.0)
    if abs(pitch_semitones) > 1e-3 and librosa is not None:
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_semitones, n_fft=512)

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
