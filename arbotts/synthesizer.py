"""ArboTTS High-Quality Lightweight Synthesizer."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

from arbotts.downloader import get_model_path
from arbotts.frontend import phonemizer


class ArboTTS:
    """Standalone, ultra-lightweight Indonesian TTS synthesizer using ONNX Runtime CPU."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        config_path: str | Path | None = None,
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

    def synthesize(
        self,
        text: str,
        speed: float = 0.85,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
    ) -> tuple[np.ndarray, float]:
        """Synthesize text into raw audio numpy array and report latency in milliseconds."""
        phoneme_ids = phonemizer.text_to_phoneme_ids(text, self.table)
        input_ids = np.asarray([phoneme_ids], dtype=np.int64)
        input_lengths = np.asarray([len(phoneme_ids)], dtype=np.int64)
        scales = np.asarray([noise_scale, 1.0 / max(speed, 0.1), noise_w], dtype=np.float32)

        inputs = {
            "input": input_ids,
            "input_lengths": input_lengths,
            "scales": scales,
            "sid": None,
        }

        t0 = time.perf_counter()
        outputs = self.session.run(["output"], inputs)
        latency_ms = (time.perf_counter() - t0) * 1000

        audio = outputs[0].squeeze()

        # Audio peak normalization
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val
        elif max_val > 0.01:
            audio = audio * 0.95

        return audio, latency_ms

    def speak(
        self,
        text: str,
        output_file: str | Path,
        speed: float = 0.85,
    ) -> float:
        """Synthesize text and save to a 16-bit PCM WAV file. Returns audio duration in seconds."""
        audio, _ = self.synthesize(text, speed=speed)
        out_p = Path(output_file)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_p), audio, self.sampling_rate, subtype="PCM_16")
        return len(audio) / self.sampling_rate
