"""Phonemization utilities for ArboTTS."""

from __future__ import annotations

import json
import logging
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("arbotts.frontend")

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2


@dataclass(frozen=True)
class PhonemeTable:
    """Voice phoneme -> Piper token ID mapping."""
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
    """Lazily initialized phonemizer backend."""

    def __init__(self) -> None:
        self._backend = None

    def _init_backend(self, voice: str = "id") -> Any:
        if self._backend is not None:
            return self._backend

        try:
            import espeakng_loader
            from phonemizer.backend import EspeakBackend
            from phonemizer.backend.espeak.wrapper import EspeakWrapper

            EspeakWrapper.set_library(espeakng_loader.get_library_path())
            EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
            self._backend = EspeakBackend(
                language=voice,
                preserve_punctuation=True,
                with_stress=True,
                tie=False,
            )
            return self._backend
        except Exception:
            pass

        try:
            from phonemizer.backend import EspeakBackend
            self._backend = EspeakBackend(
                language=voice,
                preserve_punctuation=True,
                with_stress=True,
                tie=False,
            )
            return self._backend
        except Exception as e:
            raise RuntimeError(
                f"Could not initialize espeak-ng backend: {e}. "
                f"Please ensure 'espeakng-loader' or 'espeak-ng' is installed."
            ) from e

    def phonemize(self, text: str, voice: str = "id") -> str:
        backend = self._init_backend(voice)
        out = backend.phonemize([text], strip=False)
        return out[0].rstrip() if out else ""


_FRONTEND = EspeakFrontend()


def text_to_phoneme_ids(text: str, table: PhonemeTable) -> np.ndarray:
    """Convert text string into phoneme token IDs for ONNX model."""
    clean = text.strip()
    if not clean:
        raise ValueError("Text is empty")

    raw_phon = _FRONTEND.phonemize(clean, table.espeak_voice)
    phonemes = list(unicodedata.normalize("NFD", raw_phon))

    ids: list[int] = [BOS_ID, PAD_ID]
    for p in phonemes:
        pid = table.id_map.get(p)
        if pid is not None:
            ids.append(pid)
            ids.append(PAD_ID)
    ids.append(EOS_ID)

    return np.asarray(ids, dtype=np.int64)
