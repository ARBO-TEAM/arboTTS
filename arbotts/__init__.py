"""ArboTTS: Ultra-lightweight Indonesian Text-to-Speech library."""

from __future__ import annotations

from arbotts.clone import StudioClone, rewrite_indonesian_phonetics
from arbotts.downloader import download_model, get_model_path
from arbotts.synthesizer import ArboTTS

__version__ = "2.1.0"
__all__ = [
    "ArboTTS",
    "StudioClone",
    "rewrite_indonesian_phonetics",
    "get_model_path",
    "download_model",
    "__version__",
]
