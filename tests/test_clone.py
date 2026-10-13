"""Tests for ArboTTS Studio Clone Tier (~550MB in-context voice cloning)."""

from pathlib import Path
import numpy as np
import pytest

from arbotts.clone import rewrite_indonesian_phonetics, StudioClone
from arbotts import ArboTTS

REPO_ROOT = Path(__file__).resolve().parent.parent
REF_SAMPLE = REPO_ROOT / "audition_clone" / "sample.wav"


def test_indonesian_phonetic_rewriting():
    # 1. 'c' -> 'ch' mapping
    assert rewrite_indonesian_phonetics("contoh") == "chontoh"
    assert rewrite_indonesian_phonetics("kecerdasan") == "kecherdasan"
    assert rewrite_indonesian_phonetics("baca") == "bacha"
    assert rewrite_indonesian_phonetics("kucing") == "kuching"

    # Preserves existing 'ch' or 'sch'
    assert rewrite_indonesian_phonetics("chart") == "chart"
    assert rewrite_indonesian_phonetics("schema") == "schema"

    # 2. Number and currency expansion
    assert "tujuh ratus lima puluh ribu rupiah" in rewrite_indonesian_phonetics("Rp 750.000")

    # 3. Weak unstressed syllable & compound boundary articulation
    assert rewrite_indonesian_phonetics("belum") == "be-lum"
    assert rewrite_indonesian_phonetics("banget") == "bang-et"
    assert rewrite_indonesian_phonetics("buatan") == "buat-an"


@pytest.mark.skipif(not REF_SAMPLE.exists(), reason="Reference audio sample not found")
def test_arbotts_clone_integrated():
    tts = ArboTTS(tier="standard")
    text = "Halo, ini pengetesan resmi arbo TTS studio clone."
    ref_text = "Ini adalah contoh rekaman suara referensi untuk pengujian."

    audio, latency_ms = tts.clone(
        text=text,
        reference=REF_SAMPLE,
        reference_text=ref_text,
    )

    assert isinstance(audio, np.ndarray)
    assert len(audio) > 1000
    assert latency_ms > 0
    assert np.abs(audio).max() > 0.05
