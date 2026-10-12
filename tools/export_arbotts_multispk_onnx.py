"""Export ArboTTS fine-tuned Multi-Speaker VITS PyTorch weights to standalone ONNX format.

Supports Speaker ID (sid) conditioning tensor for Arbo (0) and Arba (1).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from pathlib import Path

# Windows PosixPath fix
pathlib.PosixPath = pathlib.WindowsPath

REPO_ROOT = Path("D:/kerjaan/belajar/tts/arboTTS").resolve()
PIPER_DIR = Path("D:/kerjaan/belajar/tts/piper/src/python").resolve()

for p in [REPO_ROOT, PIPER_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
from piper_train.vits.models import SynthesizerTrn


def export_to_onnx(
    ckpt_path: str | Path,
    base_ckpt: str = "D:/kerjaan/belajar/tts/sanoTTS/experiments/epoch199.ckpt",
    cfg_json: str = "D:/kerjaan/belajar/tts/sanoTTS/experiments/_pilot/teacher.onnx.json",
    out_onnx: str | Path = "D:/kerjaan/belajar/tts/arboTTS/arbotts_v2.onnx",
    n_speakers: int = 2,
    gin_channels: int = 256,
    opset: int = 15,
) -> Path:
    out_path = Path(out_onnx)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading metadata from base checkpoint: {base_ckpt}")
    ckpt_meta = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    hp = ckpt_meta["hyper_parameters"]

    print(f"Instantiating Multi-Speaker SynthesizerTrn (n_speakers={n_speakers}, gin_channels={gin_channels})...")
    model = SynthesizerTrn(
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
        n_speakers=n_speakers,
        gin_channels=gin_channels,
        use_sdp=hp["use_sdp"],
    )

    print(f"Loading tuned multi-speaker weights from: {ckpt_path}")
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

    print(f"Exporting ONNX model to {out_path} (opset {opset})...")
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

    # Save multi-speaker config JSON
    out_cfg = out_path.with_suffix(".onnx.json")
    with open(cfg_json, "r", encoding="utf-8") as f:
        cfg_data = json.load(f)
    cfg_data["dataset"] = "arbotts_v2"
    cfg_data["num_speakers"] = n_speakers
    cfg_data["speaker_id_map"] = {
        "arbo": 0,
        "arba": 1,
    }

    with open(out_cfg, "w", encoding="utf-8") as f:
        json.dump(cfg_data, f, indent=4)

    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Successfully exported {out_path.name} ({file_size_mb:.2f} MB) and {out_cfg.name}!")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, default="D:/kerjaan/belajar/tts/arboTTS/checkpoints_vits_multispk/G_ft_1000.pth")
    parser.add_argument("-o", "--output", type=str, default="D:/kerjaan/belajar/tts/arboTTS/arbotts_v2.onnx")
    parser.add_argument("--n-speakers", type=int, default=2)
    parser.add_argument("--gin-channels", type=int, default=256)
    args = parser.parse_args()

    export_to_onnx(
        args.checkpoint,
        out_onnx=args.output,
        n_speakers=args.n_speakers,
        gin_channels=args.gin_channels,
    )
