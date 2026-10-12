"""ArboTTS: Ultra-lightweight Indonesian Text-to-Speech library."""

from __future__ import annotations

from arbotts.downloader import download_model, get_model_path
from arbotts.synthesizer import ArboTTS

__version__ = "1.0.0"
__all__ = ["ArboTTS", "get_model_path", "download_model", "__version__"]
