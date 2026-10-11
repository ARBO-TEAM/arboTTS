import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arbotts.frontend.normalizer import normalize_id_text, normalize_en_text, normalize_text
from arbotts.frontend.g2p import text_to_phonemes, phonemes_to_ids, ids_to_phonemes, UNIFIED_VOCAB

def test_normalize_id_numbers_and_currency():
    raw = "Harga tiket Rp 750.000 pada 16 November 2004."
    norm = normalize_id_text(raw)
    assert "tujuh ratus lima puluh ribu rupiah" in norm
    assert "enam belas" in norm
    assert "dua ribu empat" in norm

def test_phonemize_indonesian():
    text = "bebek makan beras di tepi sawah"
    ph = text_to_phonemes(text, lang="id")
    assert isinstance(ph, str)
    assert len(ph) > 0
    ids = phonemes_to_ids(ph)
    assert len(ids) > 0
    assert all(isinstance(i, int) for i in ids)
    # Check vocabulary round-trip
    decoded = ids_to_phonemes(ids)
    assert len(decoded) > 0

def test_phonemize_english():
    text = "Artificial intelligence is fast."
    ph = text_to_phonemes(text, lang="en")
    assert isinstance(ph, str)
    assert len(ph) > 0
    ids = phonemes_to_ids(ph)
    assert len(ids) > 0
    assert all(isinstance(i, int) for i in ids)

def test_vocabulary_size():
    assert len(UNIFIED_VOCAB) == 128
    assert UNIFIED_VOCAB[0] == "_"

if __name__ == "__main__":
    test_normalize_id_numbers_and_currency()
    print("PASS: test_normalize_id_numbers_and_currency")
    test_phonemize_indonesian()
    print("PASS: test_phonemize_indonesian")
    test_phonemize_english()
    print("PASS: test_phonemize_english")
    test_vocabulary_size()
    print("PASS: test_vocabulary_size")
    print("ALL FRONTEND TESTS PASSED!")

