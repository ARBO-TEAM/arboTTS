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


def get_model_path(model_dir: str | Path | None = None) -> tuple[Path, Path]:
    """Get paths to the ONNX model and config.

    Checks in:
    1. Specified `model_dir`
    2. Local working directory / repository root
    3. User cache directory (~/.cache/arbotts)

    Prefers v2 (multi-speaker) if present, falls back to v1.
    """
    candidates = []
    if model_dir is not None:
        candidates.append(Path(model_dir))

    # Check current directory & repo root
    cwd = Path.cwd()
    candidates.append(cwd)
    candidates.append(cwd / "models")
    candidates.append(Path(__file__).resolve().parent.parent)

    # Check cache directory
    candidates.append(DEFAULT_CACHE_DIR)

    # Check for v1 broadcast-quality model first
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

    # If not found, download v1 to cache directory
    return download_model(DEFAULT_CACHE_DIR)


def download_model(target_dir: str | Path = DEFAULT_CACHE_DIR) -> tuple[Path, Path]:
    """Download ArboTTS ONNX model and configuration from GitHub Releases."""
    out_dir = Path(target_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    m_path = out_dir / MODEL_FILES["model"]
    c_path = out_dir / MODEL_FILES["config"]

    for name, filename in MODEL_FILES.items():
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
