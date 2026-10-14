"""Tests for ArboTTS Paradee-8M Nano Tier (~8.6 MB INT8)."""

from pathlib import Path
import numpy as np
import pytest

from arbotts.paradee_engine import (
    ParadeeONNXSynthesizer,
    resolve_pepet,
    text_to_id_ipa,
    word_to_id_ipa,
)
from arbotts.synthesizer import ArboTTS


def test_paradee_g2p_rules():
    """Verify Indonesian pepet resolution and IPA mapping for Paradee."""
    # Pepet vs taling
    assert resolve_pepet("bebek") == "bebek"
    assert "ə" in resolve_pepet("dengan")
    assert "ə" in resolve_pepet("membeli")

    # Word to IPA
    ipa_word = word_to_id_ipa("cepat")
    assert "tʃ" in ipa_word
    assert "ə" in ipa_word

    # Full text to IPA
    ipa_full = text_to_id_ipa("Halo, selamat pagi Indonesia!")
    assert "tʃ" in word_to_id_ipa("cepat")
    assert "!" in ipa_full


def test_paradee_onnx_synthesis_smoke(tmp_path: Path):
    """Smoke test running standalone Paradee ONNX synthesis and file save."""
    engine = ParadeeONNXSynthesizer()
    assert engine.sample_rate == 24000

    text = "Selamat pagi Indonesia."
    audio, lat_ms = engine.synthesize(text, speed=1.0)
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert len(audio) > 0
    assert lat_ms > 0

    out_file = tmp_path / "test_paradee.wav"
    save_lat = engine.save(text, out_file, speed=1.0)
    assert out_file.exists()
    assert out_file.stat().st_size > 0
    assert save_lat > 0


def test_arbotts_tier_nano_integration(tmp_path: Path):
    """Verify ArboTTS unified class integration with tier='nano' and 'paradee'."""
    tts = ArboTTS(tier="nano")
    assert tts.sampling_rate == 24000

    text = "Uji coba arboTTS tier nano delapan megabita."
    audio, lat_ms = tts.synthesize(text, speed=1.0)
    assert isinstance(audio, np.ndarray)
    assert len(audio) > 0
    assert lat_ms > 0

    out_file = tmp_path / "test_nano.wav"
    dur = tts.speak(text, out_file, speed=1.0)
    assert out_file.exists()
    assert dur > 0


def test_paradee_bilingual_english_synthesis(tmp_path: Path):
    """Verify bilingual English synthesis through Paradee tier."""
    engine = ParadeeONNXSynthesizer()
    en_text = "Good morning, and welcome to ArboTTS."
    audio, lat_ms = engine.synthesize(en_text, speed=1.0, lang="en")
    assert isinstance(audio, np.ndarray)
    assert len(audio) > 0
    assert lat_ms > 0

    # Auto-detection
    audio_auto, _ = engine.synthesize("Hello world, this is a test.", lang="auto")
    assert len(audio_auto) > 0

