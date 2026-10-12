"""Bilingual (Indonesian & English) phonemization utilities for ArboTTS."""

from __future__ import annotations

import json
import logging
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from arbotts.frontend.lexicon import resolve_indonesian_phonemes

logger = logging.getLogger("arbotts.frontend")

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

LANG_TO_ESPEAK = {
    "id": "id",
    "indonesia": "id",
    "indonesian": "id",
    "en": "en-us",
    "english": "en-us",
    "en-us": "en-us",
    "en-gb": "en-gb",
}


@dataclass(frozen=True)
class PhonemeTable:
    """Voice phoneme -> token ID mapping."""
    espeak_voice: str
    id_map: dict[str, int]


def load_phoneme_table(config_path: Path | str) -> PhonemeTable:
    """Load phoneme table from *.onnx.json."""
    p = Path(config_path)
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    espeak_voice = (cfg.get("espeak") or {}).get("voice", "id")
    raw_map = cfg.get("phoneme_id_map", {})
    id_map: dict[str, int] = {}
    for k, ids in raw_map.items():
        if ids:
            id_map[k] = ids[0]

    return PhonemeTable(espeak_voice=espeak_voice, id_map=id_map)


class EspeakFrontend:
    """Lazily initialized phonemizer backend supporting multiple languages."""

    def __init__(self) -> None:
        self._backends: dict[str, Any] = {}

    def _get_backend(self, voice: str = "id") -> Any:
        if voice in self._backends:
            return self._backends[voice]

        try:
            import espeakng_loader
            from phonemizer.backend import EspeakBackend
            from phonemizer.backend.espeak.wrapper import EspeakWrapper

            EspeakWrapper.set_library(espeakng_loader.get_library_path())
            EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
            backend = EspeakBackend(
                language=voice,
                preserve_punctuation=True,
                with_stress=True,
                tie=False,
            )
            self._backends[voice] = backend
            return backend
        except Exception:
            pass

        try:
            from phonemizer.backend import EspeakBackend
            backend = EspeakBackend(
                language=voice,
                preserve_punctuation=True,
                with_stress=True,
                tie=False,
            )
            self._backends[voice] = backend
            return backend
        except Exception as e:
            raise RuntimeError(
                f"Could not initialize espeak-ng backend for voice '{voice}': {e}. "
                f"Please ensure 'espeakng-loader' or 'espeak-ng' is installed."
            ) from e

    def phonemize(self, text: str, voice: str = "id") -> str:
        backend = self._get_backend(voice)
        out = backend.phonemize([text], strip=False)
        return out[0].rstrip() if out else ""


_FRONTEND = EspeakFrontend()


def text_to_phoneme_ids(text: str, table: PhonemeTable, lang: str = "id") -> np.ndarray:
    """Convert text string into phoneme token IDs for ONNX model.

    Supports:
    - 'id' (Indonesian, default)
    - 'en' (English / en-us)
    """
    clean = text.strip()
    if not clean:
        raise ValueError("Text is empty")

    voice = LANG_TO_ESPEAK.get(lang.lower(), table.espeak_voice)
    if lang.lower() in ("id", "indonesia", "indonesian"):
        words = clean.split()
        if words:
            backend = _FRONTEND._get_backend(voice)
            raw_phs = backend.phonemize(words, strip=False)
            fixed_phs = [
                resolve_indonesian_phonemes(w, ph.strip())
                for w, ph in zip(words, raw_phs)
            ]
            raw_phon = " ".join(fixed_phs)
        else:
            raw_phon = _FRONTEND.phonemize(clean, voice)
    else:
        raw_phon = _FRONTEND.phonemize(clean, voice)
    phonemes = list(unicodedata.normalize("NFD", raw_phon))

    ids: list[int] = [BOS_ID, PAD_ID]
    for p in phonemes:
        pid = table.id_map.get(p)
        if pid is not None:
            ids.append(pid)
            ids.append(PAD_ID)
    ids.append(EOS_ID)

    return np.asarray(ids, dtype=np.int64)
