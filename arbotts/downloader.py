"""Model download and cache utilities for ArboTTS."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

DEFAULT_CACHE_DIR = Path(os.environ.get("ARBOTTS_CACHE_DIR", Path.home() / ".cache" / "arbotts"))
RELEASE_TAG = "v2.0.0"
BASE_URL = f"https://github.com/ARBO-TEAM/arboTTS/releases/download/{RELEASE_TAG}"

MODEL_FILES_V2 = {
    "model": "arbotts_v2.onnx",
    "config": "arbotts_v2.onnx.json",
}
MODEL_FILES_V1 = {
    "model": "arbotts_v1.onnx",
    "config": "arbotts_v1.onnx.json",
}
MODEL_FILES = MODEL_FILES_V1

TIER_FILES = {
    "standard": {"model": "arbotts_v1.onnx", "config": "arbotts_v1.onnx.json"},
    "small": {"model": "arbotts_small.onnx", "config": "arbotts_small.onnx.json"},
    "tiny": {"model": "arbotts_tiny.onnx", "config": "arbotts_tiny.onnx.json"},
}


def get_model_path(model_dir: str | Path | None = None, tier: str = "standard") -> tuple[Path, Path]:
    """Get paths to the ONNX model and config.

    Checks in:
    1. Specified `model_dir`
    2. Local working directory / repository root
    3. User cache directory (~/.cache/arbotts)

    Supports tiers: 'standard', 'small', 'tiny'.
    """
    candidates = []
    if model_dir is not None:
        candidates.append(Path(model_dir))

    cwd = Path.cwd()
    candidates.append(cwd)
    candidates.append(cwd / "models")
    candidates.append(Path(__file__).resolve().parent.parent)
    candidates.append(DEFAULT_CACHE_DIR)

    # Check for requested tier first
    tier_info = TIER_FILES.get(tier, TIER_FILES["standard"])
    for base in candidates:
        m_path = base / tier_info["model"]
        c_path = base / tier_info["config"]
        if m_path.is_file() and c_path.is_file():
            return m_path, c_path

    # Fallback to standard / v1
    for base in candidates:
        m_path = base / MODEL_FILES_V1["model"]
        c_path = base / MODEL_FILES_V1["config"]
        if m_path.is_file() and c_path.is_file():
            return m_path, c_path

    # Check for v2 if v1 not present
    for base in candidates:
        m_path = base / MODEL_FILES_V2["model"]
        c_path = base / MODEL_FILES_V2["config"]
        if m_path.is_file() and c_path.is_file():
            return m_path, c_path

    # If not found, download to cache directory
    return download_model(DEFAULT_CACHE_DIR, tier=tier)


def download_model(target_dir: str | Path = DEFAULT_CACHE_DIR, tier: str = "standard") -> tuple[Path, Path]:
    """Download ArboTTS ONNX model and configuration from GitHub Releases."""
    out_dir = Path(target_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tier_files = TIER_FILES.get(tier, TIER_FILES["standard"])
    m_path = out_dir / tier_files["model"]
    c_path = out_dir / tier_files["config"]

    for name, filename in tier_files.items():
        dst = out_dir / filename
        if dst.is_file() and dst.stat().st_size > 1000:
            continue

        url = f"{BASE_URL}/{filename}"
        print(f"[ArboTTS] Downloading {filename} from {url}...")
        try:
            urllib.request.urlretrieve(url, str(dst))
            print(f"[ArboTTS] Saved to {dst}")
        except Exception as e:
            if dst.exists():
                dst.unlink()
            raise RuntimeError(
                f"Failed to download ArboTTS model asset '{filename}' from {url}. "
                f"Error: {e}. You can download it manually and place in {out_dir}."
            ) from e

    return m_path, c_path
