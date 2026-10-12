"""Synthesize speech using the standalone exported ArboTTS ONNX model.

Uses pure onnxruntime with sanoTTS frontend phonemization.
No PyTorch required during inference.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path("D:/kerjaan/belajar/tts/arboTTS").resolve()
SANOTTS_DIR = Path("D:/kerjaan/belajar/tts/sanoTTS/pypkg").resolve()

for p in [REPO_ROOT, SANOTTS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import onnxruntime as ort
import soundfile as sf
from sanotts import frontend


def synthesize_onnx(
    model_path: str | Path,
    text: str,
    output_wav: str | Path,
    speed: float = 0.85,
    noise_scale: float = 0.667,
    noise_w: float = 0.8,
) -> tuple[np.ndarray, float]:
    model_p = Path(model_path)
    cfg_p = model_p.with_suffix(".onnx.json")

    table = frontend.load_phoneme_table(cfg_p)

    # 1. Phonemize
    phoneme_ids = frontend.text_to_phoneme_ids(text, table)
    input_ids = np.asarray([phoneme_ids], dtype=np.int64)
    input_lengths = np.asarray([len(phoneme_ids)], dtype=np.int64)
    scales = np.asarray([noise_scale, 1.0 / speed, noise_w], dtype=np.float32)

    # 2. ONNX Inference Session
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 2
    opts.intra_op_num_threads = 4
    session = ort.InferenceSession(str(model_p), sess_options=opts, providers=["CPUExecutionProvider"])

    inputs = {
        "input": input_ids,
        "input_lengths": input_lengths,
        "scales": scales,
        "sid": None,
    }

    t0 = time.perf_counter()
    outputs = session.run(["output"], inputs)
    latency_ms = (time.perf_counter() - t0) * 1000

    audio = outputs[0].squeeze()

    # Normalize audio
    max_val = np.abs(audio).max()
    if max_val > 1.0:
        audio = audio / max_val
    elif max_val > 0.01:
        audio = audio * 0.95

    out_p = Path(output_wav)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_p), audio, 22050, subtype="PCM_16")

    duration = len(audio) / 22050
    rtf = (latency_ms / 1000) / duration
    rms = float(np.sqrt(np.mean(audio**2)))
    peak = float(np.max(np.abs(audio)))
    zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2)

    print(f"Generated: {out_p.name}")
    print(f" - Duration  : {duration:.2f} s")
    print(f" - Latency   : {latency_ms:.1f} ms (RTF: {rtf:.3f})")
    print(f" - RMS / Peak: {rms:.4f} / {peak:.4f}")
    print(f" - ZCR       : {zcr:.4f}")

    return audio, latency_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ArboTTS ONNX Inference")
    parser.add_argument("-m", "--model", type=str, default="D:/kerjaan/belajar/tts/arboTTS/arbotts_v1.onnx")
    parser.add_argument("-t", "--text", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("-s", "--speed", type=float, default=0.85, help="Speech speed scale (default 0.85 for relaxed broadcast pace)")
    args = parser.parse_args()

    synthesize_onnx(args.model, args.text, args.output, speed=args.speed)
