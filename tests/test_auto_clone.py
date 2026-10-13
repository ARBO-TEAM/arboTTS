"""Tests for ArboTTS AutoClone (~96MB)."""

from pathlib import Path
import pytest
import numpy as np

from arbotts.clone import AutoClone
from arbotts.synthesizer import ArboTTS

REPO_ROOT = Path(__file__).resolve().parent.parent
SUARA_M4A = Path(r"D:\kerjaan\belajar\tts\sanoTTS\docs\suara.m4a")


def test_auto_clone_pitch_detection():
    """Verify that AutoClone correctly identifies male pitch and maps to 'arbo'."""
    if not SUARA_M4A.exists():
        pytest.skip("Reference file suara.m4a not found")

    pitch, persona = AutoClone.detect_pitch_and_persona(SUARA_M4A)
    # suara.m4a has male baritone pitch around 110-120 Hz
    assert 80.0 <= pitch <= 150.0, f"Unexpected pitch: {pitch} Hz"
    assert persona == "arbo", f"Expected male persona 'arbo', got '{persona}'"


def test_auto_clone_synthesis_smoke(tmp_path):
    """Smoke test running AutoClone end-to-end if checkpoints exist."""
    ckpt_dir = REPO_ROOT / "checkpoints_vc"
    if not ckpt_dir.exists() or not SUARA_M4A.exists():
        pytest.skip("checkpoints_vc or suara.m4a not available")

    engine = AutoClone(tier="standard")
    text = "Halo, selamat pagi Indonesia."
    audio, sr, latency_ms = engine.clone(
        text=text,
        reference=SUARA_M4A,
        speed=0.85,
    )
    assert isinstance(audio, np.ndarray)
    assert len(audio) > 0
    assert sr == 22050
    assert latency_ms > 0


def test_arbotts_clone_routing_to_autoclone():
    """Verify ArboTTS.clone routes to AutoClone by default."""
    tts = ArboTTS(tier="standard")
    assert tts._studio_clone is None
    assert tts._auto_clone is None
