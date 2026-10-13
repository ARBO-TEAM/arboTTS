"""ArboTTS Studio Clone Tier (~550MB).

Zero-shot in-context speech language model with automatic prosody, intonation,
and vocal timbre cloning. Powered by Qwen-backbone 1.58-bit ternary codec LM
(emb4 ~469 MB + S3Codec ~60 MB + SpeakerEncoder ~16.6 MB), Apache 2.0.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import soundfile as sf
import torch

from arbotts.frontend.normalizer import normalize_id_text

# Default HF cache on D: drive to protect disk C:
DEFAULT_CACHE_DIR = Path(r"D:\hf_cache")
if DEFAULT_CACHE_DIR.exists() and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(DEFAULT_CACHE_DIR)


def rewrite_indonesian_phonetics(text: str) -> str:
    """Rewrite Indonesian text to ensure correct articulation on English-tokenized speech LMs.

    1. Expands Indonesian numbers, decimals, and currency (e.g., Rp 750.000).
    2. Rewrites 'c' to 'ch' so words like 'contoh' or 'kecerdasan' are pronounced /tʃ/ instead of /k/.
    3. Hyphenates weak unstressed syllables (e.g. 'be-lum') for crisp, clear consonant-vowel separation.
    """
    # 1. Expand currency, numbers, decimals
    t = normalize_id_text(text)

    # 2. Phonetic 'c' -> 'ch' (unless already preceded by 's' or followed by 'h')
    t = re.sub(r"(?<![sS])c(?![hH])", "ch", t)
    t = re.sub(r"(?<![sS])C(?![hH])", "Ch", t)

    # 3. Weak unstressed syllables & compound boundary articulation
    t = re.sub(r"\bbelum\b", "be-lum", t, flags=re.IGNORECASE)
    t = re.sub(r"\bsebentar\b", "se-bentar", t, flags=re.IGNORECASE)
    # Fix subword tokenization splitting 'ban-get' -> 'bang-et' so /ŋ/ is grouped correctly
    t = re.sub(r"\bbanget\b", "bang-et", t, flags=re.IGNORECASE)
    # Fix suffix -an on root 'buat' to prevent nasalization to 'buatang'
    t = re.sub(r"\bbuatan\b", "buat-an", t, flags=re.IGNORECASE)

    return t


class StudioClone:
    """Studio Clone engine for zero-shot in-context voice cloning."""

    def __init__(
        self,
        model_id: str = "KittenML/kitten-tts-2",
        weights: str = "emb4",
        device: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_id = model_id
        self.weights = weights
        self.cache_dir = str(cache_dir or (DEFAULT_CACHE_DIR if DEFAULT_CACHE_DIR.exists() else None))

        self._tts = None
        self._vc = None

    def _get_engine(self):
        if self._tts is None:
            # Monkey-patch KittenTTS2._load_voices for Windows UTF-8 compatibility
            try:
                import json
                from kittenml.kittentts2.model import KittenTTS2
                def _safe_load_voices(self_model):
                    voices_path = self_model.repo_dir / self_model.config.get("voices", "voices/voices.json")
                    if not voices_path.is_file():
                        return {}
                    with open(voices_path, encoding="utf-8") as f:
                        entries = json.load(f)
                    root = voices_path.parent
                    return {name: {**spec,
                                   "reference": str(root / spec["reference"]),
                                   "artifacts": str(root / spec["artifacts"])}
                            for name, spec in entries.items()}
                KittenTTS2._load_voices = _safe_load_voices
            except Exception:
                pass

            from kittenml import KittenTTS

            # Enforce cache_dir
            if self.cache_dir and "HF_HOME" not in os.environ:
                os.environ["HF_HOME"] = self.cache_dir

            self._tts = KittenTTS(
                self.model_id,
                weights=self.weights,
                device=self.device,
                cache_dir=self.cache_dir,
            )
        return self._tts

    def _get_vc(self):
        if self._vc is None:
            from arbotts.vc import ToneColorConverter

            self._vc = ToneColorConverter(device=self.device)
        return self._vc

    @property
    def sample_rate(self) -> int:
        return self._get_engine().sample_rate

    def clone(
        self,
        text: str,
        reference: Union[str, Path],
        reference_text: Optional[str] = None,
        speed: float = 1.0,
        phonetic_preprocess: bool = True,
        polish_timbre: bool = False,
        tau: float = 0.2,
    ) -> tuple[np.ndarray, int, float]:
        """Synthesize text using zero-shot in-context cloning from reference audio.

        Parameters
        ----------
        text : str
            Input text to synthesize.
        reference : str | Path
            Path to reference audio file (.wav, .ogg, .mp3, .m4a).
        reference_text : str, optional
            Transcript of the reference audio. If None, Whisper auto-transcribes it.
        speed : float
            Speaking speed (default: 1.0).
        phonetic_preprocess : bool
            Whether to apply Indonesian phonetic rewriting (c -> ch, be-lum) (default: True).
        polish_timbre : bool
            Whether to run ToneColorConverter flow-matching resonance refinement (default: False).
        tau : float
            ToneColorConverter blend ratio if polish_timbre is True (default: 0.2).

        Returns
        -------
        audio : np.ndarray
            1D float32 numpy array of synthesized speech.
        sample_rate : int
            Sampling rate of the audio (typically 24000 Hz).
        latency_ms : float
            Synthesis latency in milliseconds.
        """
        t0 = time.perf_counter()
        engine = self._get_engine()

        ref_path = Path(reference).resolve()
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_path}")

        processed_text = rewrite_indonesian_phonetics(text) if phonetic_preprocess else text

        # Step 1: In-context generative speech synthesis
        audio = engine.generate(
            processed_text,
            reference=str(ref_path),
            reference_text=reference_text,
            normalize=False,
        )

        # Step 2: Optional ToneColorConverter resonance polish
        if polish_timbre:
            vc = self._get_vc()
            target_se = vc.extract_se(ref_path)
            audio, _ = vc.convert(audio, target_se=target_se, tau=tau)

        # Step 3: Peak normalization to 0.95
        max_val = np.abs(audio).max()
        if max_val > 0.01:
            audio = audio / max_val * 0.95

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return audio, engine.sample_rate, latency_ms
