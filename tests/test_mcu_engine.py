"""Tests for ArboTTS Bilingual MCU Tier (~1.55 MB INT8)."""

from pathlib import Path
import numpy as np
import pytest

from arbotts.mcu_engine import (
    BilingualMCUSynthesizer,
    MCUSynthesizer,
    detect_lang,
)
from arbotts.synthesizer import ArboTTS


def test_mcu_lang_detector():
    """Verify bilingual language detection."""
    assert detect_lang("Hello world, this is English speech synthesis.") == "en"
    assert detect_lang("Good morning! How are you doing today?") == "en"
    assert detect_lang("Halo selamat pagi semuanya di Indonesia.") == "id"
    assert detect_lang("Sistem ini berjalan dengan sangat baik dan jernih.") == "id"


def test_mcu_synthesis_smoke(tmp_path: Path):
    """Smoke test running standalone Bilingual MCU synthesis and file save."""
    engine = BilingualMCUSynthesizer()
    assert engine.sample_rate == 22050

    # Test Indonesian synthesis
    id_text = "Halo selamat pagi Indonesia."
    audio_id, lat_id = engine.synthesize(id_text, lang="id")
    assert isinstance(audio_id, np.ndarray)
    assert audio_id.dtype == np.float32
    assert len(audio_id) > 0
    assert lat_id > 0

    # Test English synthesis (default length_scale = 1.25)
    en_text = "Hello world, this is ArboTTS bilingual MCU synthesis."
    audio_en, lat_en = engine.synthesize(en_text, lang="en")
    assert isinstance(audio_en, np.ndarray)
    assert audio_en.dtype == np.float32
    assert len(audio_en) > 0
    assert lat_en > 0

    out_file = tmp_path / "test_mcu.wav"
    save_lat = engine.save(id_text, out_file, lang="id")
    assert out_file.exists()
    assert out_file.stat().st_size > 0
    assert save_lat > 0


def test_arbotts_tier_mcu_integration(tmp_path: Path):
    """Verify ArboTTS unified class integration with tier='mcu' and tier='bilingual'."""
    tts = ArboTTS(tier="mcu")
    assert tts.sampling_rate == 22050

    text = "Uji coba arboTTS tier MCU satu koma lima megabita."
    audio, lat_ms = tts.synthesize(text, lang="id")
    assert isinstance(audio, np.ndarray)
    assert len(audio) > 0
    assert lat_ms > 0

    # Test bilingual alias
    tts_bilingual = ArboTTS(tier="bilingual")
    assert tts_bilingual.sampling_rate == 22050
    audio_bi, lat_bi = tts_bilingual.synthesize("Welcome to the bilingual tier.", lang="en")
    assert len(audio_bi) > 0
    assert lat_bi > 0
