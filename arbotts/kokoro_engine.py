"""ArboTTS Kokoro ONNX Tier (~88 MB INT8).

Ultra-fast, studio-quality 24 kHz Text-to-Speech using Kokoro-82M ONNX Runtime
with native ArboTTS Indonesian G2P phonemizer and zero-shot voice cloning.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import onnxruntime as ort
import soundfile as sf

from arbotts.frontend import g2p

logger = logging.getLogger("arbotts.kokoro")

# HuggingFace repository containing quantized Kokoro ONNX and voice packs
KOKORO_ONNX_REPO = "onnx-community/Kokoro-82M-ONNX"

# Voice catalog mapping ArboTTS personas to Kokoro voice embeddings
KOKORO_PRESET_VOICES = {
    "arbo": "id_arbo",
    "arba": "id_arba",
    "bimo": "id_arbo",
    "dimas": "id_arbo",
    "danu": "id_arbo",
    "sita": "id_arba",
    "maya": "id_arba",
    "id_arbo": "id_arbo",
    "id_arba": "id_arba",
    "bm_george": "bm_george",
    "bm_lewis": "bm_lewis",
    "am_adam": "am_adam",
    "am_michael": "am_michael",
    "af_bella": "af_bella",
    "af_sarah": "af_sarah",
    "af_nicole": "af_nicole",
    "af_sky": "af_sky",
    "bf_emma": "bf_emma",
    "bf_isabella": "bf_isabella",
}


def arbotts_phonemes_to_kokoro_ipa(p_text: str) -> str:
    """Map ArboTTS phonemes into standard Kokoro IPA representation."""
    mapping = {
        "c": "tʃ",
        "j": "dʒ",
        "y": "j",
        "g": "ɡ",
    }
    res = []
    for ch in p_text:
        res.append(mapping.get(ch, ch))
    return "".join(res)


class KokoroONNXSynthesizer:
    """Standalone ultra-lightweight Kokoro ONNX 24 kHz Synthesizer (~88 MB).

    Features:
    - Pure ONNX Runtime CPU inference (sub-500ms latency on standard laptops).
    - Native Indonesian G2P with zero English stress cadence and pepet accuracy.
    - Zero-shot voice cloning from audio files (WAV, MP3, M4A) via style manifold.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        num_threads: int = 4,
    ) -> None:
        self.cache_dir = Path(cache_dir or os.environ.get("HF_HOME") or Path.home() / ".cache" / "arbotts" / "kokoro")
        self.num_threads = num_threads
        self.model_path = Path(model_path) if model_path else None

        self._session: Optional[ort.InferenceSession] = None
        self._vocab: Optional[dict[str, int]] = None
        self._voice_cache: dict[str, np.ndarray] = {}

    @property
    def sample_rate(self) -> int:
        """Output audio sampling rate (24,000 Hz)."""
        return 24000

    def _ensure_model_files(self) -> tuple[Path, Path]:
        """Ensure model_quantized.onnx and tokenizer.json are downloaded and cached."""
        if self.model_path and self.model_path.exists():
            onnx_path = self.model_path
            tok_path = onnx_path.parent / "tokenizer.json"
            if not tok_path.exists():
                from huggingface_hub import hf_hub_download
                tok_path = Path(hf_hub_download(KOKORO_ONNX_REPO, "tokenizer.json", local_dir=str(self.cache_dir)))
            return onnx_path, tok_path

        from huggingface_hub import hf_hub_download

        onnx_file = hf_hub_download(
            KOKORO_ONNX_REPO,
            "onnx/model_quantized.onnx",
            local_dir=str(self.cache_dir),
        )
        tok_file = hf_hub_download(
            KOKORO_ONNX_REPO,
            "tokenizer.json",
            local_dir=str(self.cache_dir),
        )
        return Path(onnx_file), Path(tok_file)

    def _get_session(self) -> ort.InferenceSession:
        """Lazy-load the ONNX Runtime session."""
        if self._session is None:
            onnx_path, tok_path = self._ensure_model_files()

            with open(tok_path, "r", encoding="utf-8") as f:
                tok_data = json.load(f)
            self._vocab = tok_data.get("model", {}).get("vocab", {})

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = self.num_threads
            opts.inter_op_num_threads = 2
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            logger.info("Initializing Kokoro ONNX Runtime engine from %s...", onnx_path)
            self._session = ort.InferenceSession(
                str(onnx_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
        return self._session

    def _get_vocab(self) -> dict[str, int]:
        """Retrieve tokenizer vocabulary mapping."""
        if self._vocab is None:
            self._get_session()
        return self._vocab or {}

    def get_voice_pack(self, voice: Union[str, Path] = "arbo") -> np.ndarray:
        """Resolve [512, 256] style tensor from voice persona name or reference audio file."""
        voice_key = str(voice)
        if voice_key in self._voice_cache:
            return self._voice_cache[voice_key]

        v_path = Path(voice)
        if v_path.exists() and v_path.is_file():
            # 1. Direct binary voice pack
            if v_path.suffix.lower() == ".bin":
                arr = np.frombuffer(v_path.read_bytes(), dtype=np.float32).reshape(-1, 256)
                self._voice_cache[voice_key] = arr
                return arr

            # 2. Audio reference cloning via inno-kokoro
            logger.info("Extracting style pack from reference audio '%s'...", v_path)
            import librosa
            import torch
            from inno_kokoro import enroll

            y, sr = librosa.load(str(v_path), sr=24000)
            wav_tensor = torch.from_numpy(y)
            tuner = enroll.Tuner(device="cpu")
            pack, _ = enroll.enroll(wav_tensor, sr, tuner)
            # pack shape is [510, 1, 256]
            pack_np = pack.squeeze(1).numpy().astype(np.float32)
            # pad or truncate to 512 rows
            if pack_np.shape[0] < 512:
                pad = np.repeat(pack_np[-1:], 512 - pack_np.shape[0], axis=0)
                pack_np = np.concatenate([pack_np, pad], axis=0)
            else:
                pack_np = pack_np[:512]
            self._voice_cache[voice_key] = pack_np
            return pack_np

        # 3. Preset voice name
        preset_name = KOKORO_PRESET_VOICES.get(voice_key.lower(), voice_key.lower())

        # Check local bundled voices in arbotts/voices/ first
        local_voice_dir = Path(__file__).resolve().parent / "voices"
        local_candidates = [
            local_voice_dir / f"{preset_name}.bin",
            local_voice_dir / f"id_{preset_name}.bin",
        ]
        for cand in local_candidates:
            if cand.exists():
                arr = np.frombuffer(cand.read_bytes(), dtype=np.float32).reshape(-1, 256)
                self._voice_cache[voice_key] = arr
                return arr

        # Fallback to downloading from HuggingFace repository
        from huggingface_hub import hf_hub_download

        bin_file = hf_hub_download(
            KOKORO_ONNX_REPO,
            f"voices/{preset_name}.bin",
            local_dir=str(self.cache_dir),
        )
        arr = np.frombuffer(open(bin_file, "rb").read(), dtype=np.float32).reshape(-1, 256)
        self._voice_cache[voice_key] = arr
        return arr

    def text_to_ipa(self, text: str, lang: str = "id") -> str:
        """Convert input text to Indonesian or English IPA compatible with Kokoro."""
        if lang.lower() == "id":
            id_phonemes = g2p.id_text_to_phonemes(text)
            return arbotts_phonemes_to_kokoro_ipa(id_phonemes)

        # English fallback
        from arbotts.frontend.g2p import en_text_to_phonemes
        return en_text_to_phonemes(text)

    def synthesize(
        self,
        text: str,
        voice: Union[str, Path] = "arbo",
        speed: float = 0.95,
        lang: str = "id",
    ) -> tuple[np.ndarray, float]:
        """Synthesize text into a 24 kHz numpy float32 waveform and report latency in ms.

        Parameters
        ----------
        text : str
            Input text to synthesize.
        voice : str | Path
            Preset persona name ('arbo', 'arba', 'bimo', etc.) or path to reference audio file.
        speed : float
            Speech speed multiplier (default: 0.95 for relaxed natural Indonesian cadence).
        lang : str
            Language code: 'id' (default) or 'en'.
        """
        t0 = time.perf_counter()
        session = self._get_session()
        vocab = self._get_vocab()

        # Convert text to IPA
        ipa = self.text_to_ipa(text, lang=lang)

        # Tokenize with Kokoro vocabulary
        tokens = [vocab[c] for c in ipa if c in vocab]
        if not tokens:
            return np.zeros(0, dtype=np.float32), 0.0

        input_ids = np.array([[0, *tokens, 0]], dtype=np.int64)

        # Get style vector corresponding to sequence length
        voice_pack = self.get_voice_pack(voice)
        idx = min(len(tokens), voice_pack.shape[0] - 1)
        style = voice_pack[idx : idx + 1, :].astype(np.float32)

        speed_arr = np.array([float(speed)], dtype=np.float32)

        # Run ONNX inference
        outputs = session.run(["waveform"], {
            "input_ids": input_ids,
            "style": style,
            "speed": speed_arr,
        })
        waveform = outputs[0].squeeze()

        # Normalize peak
        max_val = float(np.abs(waveform).max())
        if max_val > 1.0:
            waveform = waveform / max_val
        elif max_val > 0.01:
            waveform = waveform * 0.95

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return waveform.astype(np.float32), latency_ms

    def speak(
        self,
        text: str,
        output_path: Union[str, Path],
        voice: Union[str, Path] = "arbo",
        speed: float = 0.95,
        lang: str = "id",
    ) -> float:
        """Synthesize speech and write to a 16-bit PCM WAV file. Returns audio duration in seconds."""
        waveform, _ = self.synthesize(text, voice=voice, speed=speed, lang=lang)
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_p), waveform, self.sample_rate, subtype="PCM_16")
        return len(waveform) / self.sample_rate
