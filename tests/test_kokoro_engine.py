"""Tests for ArboTTS Kokoro ONNX Tier (~88 MB)."""

from pathlib import Path
import pytest
import numpy as np

from arbotts.kokoro_engine import KokoroONNXSynthesizer, arbotts_phonemes_to_kokoro_ipa
from arbotts.synthesizer import ArboTTS

REPO_ROOT = Path(__file__).resolve().parent.parent
SUARA_M4A = Path(r"D:\kerjaan\belajar\tts\sanoTTS\docs\suara.m4a")


def test_phoneme_mapping():
    """Verify conversion from ArboTTS phonemes to Kokoro IPA."""
    mapped = arbotts_phonemes_to_kokoro_ipa("cəpat jərnih yaŋ")
    assert "tʃ" in mapped
    assert "dʒ" in mapped
    assert "j" in mapped


def test_kokoro_onnx_synthesis_smoke():
    """Smoke test running Kokoro ONNX synthesis."""
    engine = KokoroONNXSynthesizer()
    text = "Halo, selamat pagi Indonesia."
    audio, lat_ms = engine.synthesize(text, voice="arbo", speed=1.0)
    assert isinstance(audio, np.ndarray)
    assert len(audio) > 0
    assert engine.sample_rate == 24000
    assert lat_ms > 0


def test_kokoro_onnx_cloning_smoke():
    """Smoke test running Kokoro ONNX zero-shot cloning from audio reference."""
    if not SUARA_M4A.exists():
        pytest.skip("Reference file suara.m4a not found")

    tts = ArboTTS(tier="kokoro")
    audio, lat_ms = tts.clone("Ini suara saya sendiri.", reference=SUARA_M4A)
    assert isinstance(audio, np.ndarray)
    assert len(audio) > 0
    assert tts.sampling_rate == 24000
    assert lat_ms > 0
