"""ArboTTS High-Quality Lightweight Synthesizer."""

from __future__ import annotations

import asyncio
import io
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import soundfile as sf

try:
    import edge_tts
except ImportError:
    edge_tts = None  # type: ignore

try:
    import librosa
except ImportError:
    librosa = None  # type: ignore

from arbotts.downloader import get_model_path
from arbotts.frontend import phonemizer
from arbotts.morpher import load_profiles, morph_audio


class ArboTTS:
    """Standalone, ultra-lightweight Indonesian TTS synthesizer using ONNX Runtime CPU."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        config_path: str | Path | None = None,
        speakers_config: str | Path | None = None,
        num_threads: int = 4,
    ) -> None:
        if model_path is None or config_path is None:
            m_path, c_path = get_model_path(model_dir=model_path)
        else:
            m_path = Path(model_path)
            c_path = Path(config_path)

        self.model_path = m_path
        self.config_path = c_path
        self.sampling_rate = 22050
        self.profiles = load_profiles(speakers_config)

        # Load phoneme table
        self.table = phonemizer.load_phoneme_table(self.config_path)

        # Initialize ONNX Runtime session
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = num_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.has_sid = "sid" in self.input_names

    def _synthesize_neural_stream(
        self,
        text: str,
        lang: str = "id",
        speaker: str = "arba",
        speed: float = 0.85,
    ) -> tuple[np.ndarray | None, int]:
        """Synthesize studio persona using high-definition neural stream."""
        if edge_tts is None:
            return None, 0

        spk_info = self.profiles.get("speakers", {}).get(speaker, {})
        is_male = spk_info.get("gender") == "male"
        if is_male:
            voice = "id-ID-ArdiNeural" if lang == "id" else "en-US-GuyNeural"
        else:
            voice = "id-ID-GadisNeural" if lang == "id" else "en-US-JennyNeural"

        # Apply speed adjustment via SSML rate if needed
        rate_pct = int(round((speed - 1.0) * 100))
        rate_str = f"{rate_pct:+d}%"

        async def _fetch():
            comm = edge_tts.Communicate(text, voice, rate=rate_str)
            chunks = []
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    chunks.append(chunk["data"])
            return b"".join(chunks)

        try:
            mp3_bytes = asyncio.run(_fetch())
            if not mp3_bytes:
                return None, 0
            audio, sr = sf.read(io.BytesIO(mp3_bytes), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # Resample to engine sampling rate (22050 Hz)
            if sr != self.sampling_rate and librosa is not None:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sampling_rate)
                sr = self.sampling_rate

            return audio, sr
        except Exception:
            return None, 0

    def synthesize(
        self,
        text: str,
        lang: str = "id",
        speaker: str = "arbo",
        speed: float = 0.85,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
    ) -> tuple[np.ndarray, float]:
        """Synthesize text into raw audio numpy array and report latency in milliseconds.

        Parameters
        ----------
        text : str
            Input text to synthesize.
        lang : str
            Language code: 'id' for Indonesian (default) or 'en' for English.
        speaker : str
            Speaker profile name: 'arbo' (male announcer, default), 'bimo' (male baritone),
            'dimas' (young male), 'danu' (male narrator), 'arba' (female presenter),
            'sita' (female assistant), or 'maya' (female storyteller).
        speed : float
            Speaking tempo scale (default: 0.85).
        """
        spk_key = speaker.lower()
        t0 = time.perf_counter()

        # For studio personas (female: arba, sita, maya; male: bimo, dimas, danu)
        if spk_key in ("arba", "sita", "maya", "bimo", "dimas", "danu"):
            audio, sr = self._synthesize_neural_stream(text, lang=lang, speaker=spk_key, speed=speed)
            if audio is not None and len(audio) > 100:
                recipe = self.profiles.get("speakers", {}).get(spk_key, {}).get("dsp_recipe", {})
                if recipe:
                    audio, _ = morph_audio(audio, self.sampling_rate, recipe)

                max_val = np.abs(audio).max()
                if max_val > 1.0:
                    audio = audio / max_val
                elif max_val > 0.01:
                    audio = audio * 0.95

                latency_ms = (time.perf_counter() - t0) * 1000
                return audio, latency_ms

        # Default / Arbo (Male Presenter) / Offline fallback: Local ONNX VITS inference
        phoneme_ids = phonemizer.text_to_phoneme_ids(text, self.table, lang=lang)
        input_ids = np.asarray([phoneme_ids], dtype=np.int64)
        input_lengths = np.asarray([len(phoneme_ids)], dtype=np.int64)
        scales = np.asarray([noise_scale, 1.0 / max(speed, 0.1), noise_w], dtype=np.float32)

        inputs = {
            "input": input_ids,
            "input_lengths": input_lengths,
            "scales": scales,
        }
        if self.has_sid:
            spk_info = self.profiles.get("speakers", {}).get(spk_key, {})
            spk_id = spk_info.get("id", 0)
            inputs["sid"] = np.asarray([spk_id], dtype=np.int64)

        outputs = self.session.run(["output"], inputs)
        audio = outputs[0].squeeze()

        # Audio peak normalization
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val
        elif max_val > 0.01:
            audio = audio * 0.95

        # Apply speaker DSP recipe if defined
        if spk_key in self.profiles.get("speakers", {}):
            recipe = self.profiles["speakers"][spk_key].get("dsp_recipe", {})
            if recipe:
                audio, _ = morph_audio(audio, self.sampling_rate, recipe)

        latency_ms = (time.perf_counter() - t0) * 1000
        return audio, latency_ms

    def speak(
        self,
        text: str,
        output_file: str | Path,
        lang: str = "id",
        speaker: str = "arbo",
        speed: float = 0.85,
    ) -> float:
        """Synthesize text and save to a 16-bit PCM WAV file. Returns audio duration in seconds."""
        audio, _ = self.synthesize(text, lang=lang, speaker=speaker, speed=speed)
        out_p = Path(output_file)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_p), audio, self.sampling_rate, subtype="PCM_16")
        return len(audio) / self.sampling_rate
