"""ArboTTS: Ultra-lightweight Indonesian Text-to-Speech library."""

from __future__ import annotations

from arbotts.clone import AutoClone, StudioClone, rewrite_indonesian_phonetics
from arbotts.downloader import download_model, get_model_path
from arbotts.kokoro_engine import KOKORO_PRESET_VOICES, KokoroONNXSynthesizer
from arbotts.synthesizer import ArboTTS

__version__ = "2.3.0"
__all__ = [
    "ArboTTS",
    "AutoClone",
    "StudioClone",
    "KokoroONNXSynthesizer",
    "KOKORO_PRESET_VOICES",
    "rewrite_indonesian_phonetics",
    "get_model_path",
    "download_model",
    "__version__",
]
