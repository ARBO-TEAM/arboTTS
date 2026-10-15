"""ArboTTS Bilingual MCU Tier (~1.55 MB INT8).

Ultra-compact, embedded C99 Text-to-Speech using distilled Kokoro-82M & VITS-Arbo
student models (Duration Student + Acoustic Latent Student + Piperlite iSTFT Decoder)
quantized to INT8 for microcontrollers (ESP32-S3 without PSRAM) and ultra-lightweight edge devices.
Supports native Indonesian and English speech with zero cloud dependencies.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import soundfile as sf

from arbotts.frontend import phonemizer

logger = logging.getLogger("arbotts.mcu")


def detect_lang(text: str) -> str:
    """Detect whether text is predominantly English or Indonesian."""
    text_clean = text.lower()
    words = set(text_clean.split())

    en_markers = {
        "the", "is", "are", "this", "that", "with", "have", "from", "you",
        "good", "welcome", "hello", "please", "morning", "afternoon", "evening",
        "artificial", "intelligence", "today", "speech", "synthesizer", "devices",
        "drink", "water", "short", "break", "world", "future", "running"
    }
    id_markers = {
        "dan", "yang", "di", "ke", "dari", "ini", "itu", "untuk", "pada",
        "adalah", "dengan", "saya", "kamu", "dia", "mereka", "kita", "kami",
        "selamat", "pagi", "siang", "sore", "malam", "terima", "kasih", "bisa",
        "akan", "sudah", "mikrokontroler", "sangat", "jernih", "suara", "semuanya"
    }

    en_hits = len(words.intersection(en_markers))
    id_hits = len(words.intersection(id_markers))
    return "en" if en_hits > id_hits else "id"


class BilingualMCUSynthesizer:
    """Standalone ultra-compact Bilingual MCU Synthesizer (~1.55 MB INT8).

    Features:
    - Entire model weights and architecture distilled into ~1.55 MB INT8 bundle.
    - Dual native language support: Indonesian (id) & English (en).
    - Sub-1M parameter student model: 136k Duration + 697k Latent + 730k Decoder.
    - Default speed: 1.25 for relaxed English articulation, 1.0 for Indonesian.
    - 2nd-order Butterworth HPF + frame-based silence gating + 1.25x volume boost.
    - 0 KB PSRAM required on ESP32-S3 (fits in internal SRAM ~184 KB).
    """

    def __init__(
        self,
        bundle_path: Optional[Union[str, Path]] = None,
        models_dir: Optional[Union[str, Path]] = None,
        e2e_bin: Optional[Union[str, Path]] = None,
    ) -> None:
        default_dir = Path(__file__).resolve().parent / "models"
        self.models_dir = Path(models_dir) if models_dir else default_dir
        self.bundle_path = Path(bundle_path) if bundle_path else self.models_dir / "mcu_bilingual_arbo_kokoro_745k_q8.bin"

        # Check for extracted front-q8 and dec-q8 components
        self.front_dir = self.models_dir / "front-q8"
        self.dec_dir = self.models_dir / "dec-q8"
        if not (self.front_dir.exists() and self.dec_dir.exists()) and self.bundle_path.exists():
            self._unpack_bundle()

        # Executable loader
        self.e2e_bin = Path(e2e_bin) if e2e_bin else self.models_dir / ("piperlite_e2e.exe" if os.name == "nt" else "piperlite_e2e")
        if not self.e2e_bin.exists():
            # Check PATH or parent directory
            alt_bin = shutil.which("piperlite_e2e")
            if alt_bin:
                self.e2e_bin = Path(alt_bin)

        # Load phoneme table
        phoneme_map_path = self.models_dir / "phoneme_map.json"
        if not phoneme_map_path.exists():
            # Search repo root
            root_map = Path(__file__).resolve().parents[1] / "arbotts_v1.onnx.json"
            if root_map.exists():
                phoneme_map_path = root_map

        if phoneme_map_path.exists():
            self.table = phonemizer.load_phoneme_table(phoneme_map_path)
        else:
            raise FileNotFoundError(f"Phoneme table mapping not found in {self.models_dir}")

    @property
    def sample_rate(self) -> int:
        """Output audio sampling rate (22,050 Hz)."""
        return 22050

    def _unpack_bundle(self) -> None:
        """Unpack single-blob .bin bundle into front-q8 and dec-q8 components."""
        if not self.bundle_path.exists():
            return
        data = self.bundle_path.read_bytes()
        if len(data) < 64:
            return
        hdr = struct.unpack_from("<10I", data, 0)
        magic, version, total_bytes, _, off_fm, len_fm, off_fw, len_fw, off_dm, len_dm = hdr
        off_dw, len_dw = struct.unpack_from("<2I", data, 40)

        self.front_dir.mkdir(parents=True, exist_ok=True)
        self.dec_dir.mkdir(parents=True, exist_ok=True)

        (self.front_dir / "front_meta_q8.bin").write_bytes(data[off_fm:off_fm + len_fm])
        (self.front_dir / "front_weights_q8.bin").write_bytes(data[off_fw:off_fw + len_fw])
        (self.dec_dir / "meta_q8.bin").write_bytes(data[off_dm:off_dm + len_dm])
        (self.dec_dir / "weights_q8.bin").write_bytes(data[off_dw:off_dw + len_dw])
        logger.info("Unpacked bilingual MCU bundle into %s and %s", self.front_dir, self.dec_dir)

    def synthesize(
        self,
        text: str,
        speed: Optional[float] = None,
        lang: str = "id",
    ) -> tuple[np.ndarray, float]:
        """Synthesize text into 22,050 Hz numpy float32 audio waveform.

        Parameters
        ----------
        text : str
            Input text to synthesize.
        speed : float, optional
            Speaking speed / length_scale. Defaults to 1.25 for English (relaxed pacing)
            and 1.0 for Indonesian.
        lang : str
            Language code: 'id' for Indonesian, 'en' for English, or 'auto' for auto-detection.

        Returns
        -------
        audio : np.ndarray
            Synthesized audio samples (float32, 22.05 kHz).
        latency_ms : float
            Latency of synthesis in milliseconds.
        """
        clean_text = text.strip()
        if not clean_text:
            return np.zeros((0,), dtype=np.float32), 0.0

        target_lang = detect_lang(clean_text) if lang.lower() == "auto" else lang.lower()

        # Default speed: 1.25 for English, 1.0 for Indonesian
        if speed is None or speed <= 0:
            actual_speed = 1.25 if target_lang.startswith("en") else 1.0
        else:
            actual_speed = float(speed)

        t0 = time.perf_counter()

        # Phonemize to token IDs
        ids = phonemizer.text_to_phoneme_ids(clean_text, self.table, lang=target_lang)
        if len(ids) == 0:
            return np.zeros((0,), dtype=np.float32), 0.0

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ids_bin = tmp_path / "input_ids.bin"
            np.array(ids, dtype=np.int32).tofile(ids_bin)

            out_wav = tmp_path / "input_ids-int8.wav"

            if self.e2e_bin and self.e2e_bin.exists():
                cmd = [
                    str(self.e2e_bin),
                    str(self.front_dir),
                    str(self.dec_dir),
                    str(actual_speed),
                    str(tmp_path),
                    str(ids_bin),
                ]
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode != 0:
                    raise RuntimeError(f"MCU piperlite synthesis failed: {res.stderr}")

                if out_wav.exists():
                    audio, sr = sf.read(out_wav, dtype="float32")
                else:
                    raise FileNotFoundError(f"Expected synthesis output not generated: {out_wav}")
            else:
                raise RuntimeError("MCU piperlite binary not found. Please compile or provide piperlite_e2e.")

        latency_ms = (time.perf_counter() - t0) * 1000
        return audio, latency_ms

    def save(
        self,
        text: str,
        output_path: Union[str, Path],
        speed: Optional[float] = None,
        lang: str = "id",
    ) -> float:
        """Synthesize text and write directly to WAV file."""
        audio, latency_ms = self.synthesize(text, speed=speed, lang=lang)
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_p), audio, self.sample_rate)
        return latency_ms


# Alias for convenience
MCUSynthesizer = BilingualMCUSynthesizer
