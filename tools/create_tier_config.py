"""Tool to generate tier configuration files (Standard, Small, Tiny) for ArboTTS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def generate_tier_config(tier_name: str, out_path: Path | None = None) -> dict:
    tiers_path = REPO_ROOT / "configs" / "tiers.json"
    base_cfg_path = REPO_ROOT / "arbotts_v1.onnx.json"

    with open(tiers_path, "r", encoding="utf-8") as f:
        tiers_data = json.load(f)["tiers"]

    if tier_name not in tiers_data:
        raise ValueError(f"Unknown tier: {tier_name}. Choose from {list(tiers_data.keys())}")

    tier_spec = tiers_data[tier_name]

    with open(base_cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Inject tier architecture hyperparameters
    cfg["dataset"] = f"arbotts_{tier_name}"
    cfg["tier"] = tier_name
    cfg["tier_info"] = {
        "name": tier_spec["name"],
        "description": tier_spec["description"],
        "model_size_mb": tier_spec["model_size_mb"],
        "parameters_m": tier_spec["parameters_m"],
        "target_rtf_cpu": tier_spec["target_rtf_cpu"],
        "target_platform": tier_spec["target_platform"],
    }
    cfg["model"] = {
        "hidden_channels": tier_spec["hidden_channels"],
        "filter_channels": tier_spec["filter_channels"],
        "n_heads": tier_spec["n_heads"],
        "n_layers": tier_spec["n_layers"],
        "upsample_initial_channel": tier_spec["upsample_initial_channel"],
        "resblock": tier_spec["resblock"],
    }

    if out_path is None:
        out_path = REPO_ROOT / f"arbotts_{tier_name}.onnx.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"[ArboTTS Tier] Generated '{tier_name}' config -> {out_path}")
    print(f"  • Parameters : ~{tier_spec['parameters_m']} M")
    print(f"  • Est. Size  : ~{tier_spec['model_size_mb']} MB")
    print(f"  • Target RTF : ~{tier_spec['target_rtf_cpu']}x (CPU)")
    print(f"  • Platform   : {tier_spec['target_platform']}")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ArboTTS model tier configuration")
    parser.add_argument("--tier", type=str, default="all", choices=["standard", "small", "tiny", "all"], help="Tier to generate")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output JSON path (optional)")
    args = parser.parse_args()

    if args.tier == "all":
        for t in ["standard", "small", "tiny"]:
            generate_tier_config(t)
    else:
        out = Path(args.output) if args.output else None
        generate_tier_config(args.tier, out)


if __name__ == "__main__":
    main()
