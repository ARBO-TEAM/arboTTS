"""Export ArboTTS Tiered (Small, Tiny, Standard) VITS PyTorch weights to standalone ONNX format.

Compatible with Piper / SanoTTS / ArboTTS ONNX runtime.
Supports multi-speaker conditioning tensor [sid].
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from pathlib import Path

# Fix Windows PosixPath unpickling
pathlib.PosixPath = pathlib.WindowsPath

REPO_ROOT = Path("D:/kerjaan/belajar/tts/arboTTS").resolve()
PIPER_DIR = Path("D:/kerjaan/belajar/tts/piper/src/python").resolve()
SANOTTS_DIR = Path("D:/kerjaan/belajar/tts/sanoTTS/pypkg").resolve()

for p in [REPO_ROOT, PIPER_DIR, SANOTTS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
from piper_train.vits.models import SynthesizerTrn


def export_tier_to_onnx(
    ckpt_path: str | Path,
    cfg_json: str | Path = "D:/kerjaan/belajar/tts/arboTTS/arbotts_small.onnx.json",
    base_ckpt: str = "D:/kerjaan/belajar/tts/sanoTTS/experiments/epoch199.ckpt",
    out_onnx: str | Path = "D:/kerjaan/belajar/tts/arboTTS/arbotts_small.onnx",
    n_speakers: int = 2,
    gin_channels: int = 256,
    opset: int = 15,
) -> Path:
    out_path = Path(out_onnx)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading metadata from base checkpoint: {base_ckpt}")
    ckpt_meta = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    hp = ckpt_meta["hyper_parameters"]

    cfg_p = Path(cfg_json)
    with open(cfg_p, "r", encoding="utf-8") as f:
        cfg_data = json.load(f)

    # Read architectural tier parameters
    m_cfg = cfg_data.get("model", {})
    hidden_channels = m_cfg.get("hidden_channels", hp["hidden_channels"])
    filter_channels = m_cfg.get("filter_channels", hp["filter_channels"])
    n_heads = m_cfg.get("n_heads", hp["n_heads"])
    n_layers = m_cfg.get("n_layers", hp["n_layers"])
    upsample_initial_channel = m_cfg.get("upsample_initial_channel", hp["upsample_initial_channel"])
    resblock = str(m_cfg.get("resblock", hp["resblock"]))

    print(f"\n=======================================================")
    print(f"📦 Exporting ArboTTS Tiered Model to ONNX: {out_path.name}")
    print(f"   Hidden Channels: {hidden_channels}")
    print(f"   Filter Channels: {filter_channels}")
    print(f"   Layers         : {n_layers}")
    print(f"   ResBlock       : {resblock}")
    print(f"   Speakers       : {n_speakers} (gin_channels={gin_channels})")
    print(f"=======================================================\n")

    model = SynthesizerTrn(
        n_vocab=hp["num_symbols"],
        spec_channels=hp["filter_length"] // 2 + 1,
        segment_size=hp["segment_size"] // hp["hop_length"],
        inter_channels=hidden_channels,
        hidden_channels=hidden_channels,
        filter_channels=filter_channels,
        n_heads=n_heads,
        n_layers=n_layers,
        kernel_size=hp["kernel_size"],
        p_dropout=hp["p_dropout"],
        resblock=resblock,
        resblock_kernel_sizes=hp["resblock_kernel_sizes"],
        resblock_dilation_sizes=hp["resblock_dilation_sizes"],
        upsample_rates=hp["upsample_rates"],
        upsample_initial_channel=upsample_initial_channel,
        upsample_kernel_sizes=hp["upsample_kernel_sizes"],
        n_speakers=n_speakers,
        gin_channels=gin_channels,
        use_sdp=hp["use_sdp"],
    )

    print(f"Loading weights from checkpoint: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("model_g.", ""): v for k, v in sd.items() if k.startswith("model_g.") or not k.startswith("model_d.")}
    model.load_state_dict(sd, strict=False)
    model.eval()

    with torch.no_grad():
        model.dec.remove_weight_norm()

    def infer_forward(text, text_lengths, scales, sid=None):
        noise_scale = scales[0]
        length_scale = scales[1]
        noise_scale_w = scales[2]
        audio = model.infer(
            text,
            text_lengths,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
            sid=sid,
        )[0].unsqueeze(1)
        return audio

    model.forward = infer_forward

    num_symbols = model.n_vocab
    dummy_input_length = 50
    sequences = torch.randint(low=0, high=num_symbols, size=(1, dummy_input_length), dtype=torch.long)
    sequence_lengths = torch.LongTensor([sequences.size(1)])
    scales = torch.FloatTensor([0.667, 1.0, 0.8])
    sid = torch.LongTensor([0])

    dummy_input = (sequences, sequence_lengths, scales, sid)

    print(f"Exporting ONNX model graph to {out_path} (opset {opset})...")
    torch.onnx.export(
        model=model,
        args=dummy_input,
        f=str(out_path),
        verbose=False,
        opset_version=opset,
        input_names=["input", "input_lengths", "scales", "sid"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "phonemes"},
            "input_lengths": {0: "batch_size"},
            "output": {0: "batch_size", 1: "time"},
        },
        dynamo=False,
    )

    # Save matching ONNX JSON config
    out_cfg = out_path.with_suffix(".onnx.json")
    cfg_data["num_speakers"] = n_speakers
    cfg_data["speaker_id_map"] = {
        "arbo": 0,
        "arba": 1,
    }

    with open(out_cfg, "w", encoding="utf-8") as f:
        json.dump(cfg_data, f, indent=2, ensure_ascii=False)

    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\n✅ Successfully exported {out_path.name} ({file_size_mb:.2f} MB) and {out_cfg.name}!")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ArboTTS Tier Model to ONNX")
    parser.add_argument("checkpoint", type=str, help="Path to PyTorch checkpoint (G_*.pth)")
    parser.add_argument("--config", type=str, default="D:/kerjaan/belajar/tts/arboTTS/arbotts_small.onnx.json", help="Path to tier config JSON")
    parser.add_argument("-o", "--output", type=str, default="D:/kerjaan/belajar/tts/arboTTS/arbotts_small.onnx", help="Output ONNX path")
    parser.add_argument("--speakers", type=int, default=2, help="Number of speakers")
    parser.add_argument("--gin-channels", type=int, default=256, help="Speaker embedding channels")
    parser.add_argument("--opset", type=int, default=15, help="ONNX opset version")
    args = parser.parse_args()

    export_tier_to_onnx(
        ckpt_path=args.checkpoint,
        cfg_json=args.config,
        out_onnx=args.output,
        n_speakers=args.speakers,
        gin_channels=args.gin_channels,
        opset=args.opset,
    )
