"""Audition and benchmarking suite for ArboTTS fine-tuned VITS model.

Synthesizes benchmark test sentences and computes broadcast acoustic metrics:
- RMS speech energy (target: 0.10 - 0.15)
- Peak amplitude (target: 0.50 - 0.95)
- Zero Crossing Rate / ZCR (target: 0.06 - 0.10 for crisp natural consonants)
- Silence ratio (target: 15% - 25% natural cadence)
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from pathlib import Path

# Windows PosixPath unpickle fix
pathlib.PosixPath = pathlib.WindowsPath

REPO_ROOT = Path("D:/kerjaan/belajar/tts/arboTTS").resolve()
PIPER_DIR = Path("D:/kerjaan/belajar/tts/piper/src/python").resolve()
SANOTTS_DIR = Path("D:/kerjaan/belajar/tts/sanoTTS/pypkg").resolve()

for p in [REPO_ROOT, PIPER_DIR, SANOTTS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import soundfile as sf
import torch

from piper_train.vits.models import SynthesizerTrn
from sanotts import frontend

TEST_SENTENCES = {
    "intro": "Halo, selamat datang di ArboTTS. Suara ini dihasilkan oleh kecerdasan buatan.",
    "harga": "Harga beras medium di pasar tradisional mencapai empat belas ribu rupiah per kilogram hari ini.",
    "presiden": "Presiden Joko Widodo meresmikan jembatan baru sepanjang dua ratus meter di Jawa Barat.",
    "english": "Good morning, this is ArboTTS speaking English fluently.",
}


def compute_metrics(audio: np.ndarray, sr: int = 22050) -> dict[str, float]:
    duration = len(audio) / sr
    rms = float(np.sqrt(np.mean(audio**2)))
    peak = float(np.max(np.abs(audio)))
    zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2)

    frame_len = int(sr * 0.01)
    frames = np.array([audio[i : i + frame_len] for i in range(0, len(audio) - frame_len, frame_len)])
    frame_rms = np.sqrt(np.mean(frames**2, axis=1))
    silence = float(np.mean(frame_rms < (0.05 * rms)))

    return {
        "duration": duration,
        "rms": rms,
        "peak": peak,
        "zcr": zcr,
        "silence": silence,
    }


def benchmark_checkpoint(
    ckpt_path: str | Path,
    base_ckpt: str = "D:/kerjaan/belajar/tts/sanoTTS/experiments/epoch199.ckpt",
    cfg_json: str = "D:/kerjaan/belajar/tts/sanoTTS/experiments/_pilot/teacher.onnx.json",
    out_dir: str | Path = "D:/kerjaan/belajar/tts/arboTTS/benchmark_samples",
    device_str: str = "cuda:0",
) -> dict[str, dict[str, float]]:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Hyperparameters & phoneme table
    ckpt_meta = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    hp = ckpt_meta["hyper_parameters"]
    table = frontend.load_phoneme_table(Path(cfg_json))

    # 2. Build model
    net_g = SynthesizerTrn(
        n_vocab=hp["num_symbols"],
        spec_channels=hp["filter_length"] // 2 + 1,
        segment_size=hp["segment_size"] // hp["hop_length"],
        inter_channels=hp["inter_channels"],
        hidden_channels=hp["hidden_channels"],
        filter_channels=hp["filter_channels"],
        n_heads=hp["n_heads"],
        n_layers=hp["n_layers"],
        kernel_size=hp["kernel_size"],
        p_dropout=hp["p_dropout"],
        resblock=str(hp["resblock"]),
        resblock_kernel_sizes=hp["resblock_kernel_sizes"],
        resblock_dilation_sizes=hp["resblock_dilation_sizes"],
        upsample_rates=hp["upsample_rates"],
        upsample_initial_channel=hp["upsample_initial_channel"],
        upsample_kernel_sizes=hp["upsample_kernel_sizes"],
        n_speakers=hp["num_speakers"],
        gin_channels=hp["gin_channels"],
        use_sdp=hp["use_sdp"],
    ).to(device)

    # 3. Load weights
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    if "model_g" in next(iter(sd.keys())):
        sd = {k.replace("model_g.", ""): v for k, v in sd.items() if k.startswith("model_g.")}
    net_g.load_state_dict(sd, strict=False)
    net_g.eval()

    ckpt_name = Path(ckpt_path).stem
    results = {}

    print(f"\n========================================================")
    print(f" Audition Benchmarking for Checkpoint: {ckpt_name}")
    print(f"========================================================")

    for key, text in TEST_SENTENCES.items():
        phoneme_ids = frontend.text_to_phoneme_ids(text, table)
        x = torch.tensor(phoneme_ids, dtype=torch.long, device=device).unsqueeze(0)
        x_lengths = torch.tensor([len(phoneme_ids)], dtype=torch.long, device=device)

        with torch.no_grad():
            audio_tensor = net_g.infer(x, x_lengths, noise_scale=0.667, length_scale=1.0, noise_scale_w=0.8)[0][0, 0]
            audio = audio_tensor.cpu().numpy()

        # Normalize
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val
        elif max_val > 0.01:
            audio = audio * 0.95

        wav_file = out_path / f"{ckpt_name}_{key}.wav"
        sf.write(str(wav_file), audio, 22050, subtype="PCM_16")

        m = compute_metrics(audio, 22050)
        results[key] = m
        print(
            f"[{key:8s}] Dur: {m['duration']:5.2f}s | RMS: {m['rms']:6.4f} | "
            f"Peak: {m['peak']:6.4f} | ZCR: {m['zcr']:6.4f} | Silence: {m['silence']*100:4.1f}% | "
            f"File: {wav_file.name}"
        )

    print("========================================================\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="Path to checkpoint (.pth or .ckpt)")
    parser.add_argument("-o", "--out-dir", type=str, default="D:/kerjaan/belajar/tts/arboTTS/benchmark_samples")
    args = parser.parse_args()

    benchmark_checkpoint(args.checkpoint, out_dir=args.out_dir)
