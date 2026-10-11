"""Unified Bilingual (Indonesian + English) Grapheme-to-Phoneme (G2P) engine."""

from __future__ import annotations

import json
import re
from pathlib import Path

# Static 128-symbol unified IPA vocabulary table for VITS
# ID 0 is pad symbol '_'
# ID 1..7 are punctuation symbols
# ID 8..127 are Indonesian and English IPA phonemes
UNIFIED_VOCAB: list[str] = [
    "_", " ", ".", ",", "!", "?", "-", ":", ";",
    # Indonesian & English Vowels & Diphthongs
    "a", "i", "u", "e", "o", "ə", "ɛ", "ɔ", "ɪ", "ʊ", "æ", "ʌ", "ɑ", "ɒ",
    "aj", "aw", "oj", "eɪ", "aɪ", "ɔɪ", "aʊ", "oʊ", "ɪə", "eə", "ʊə",
    # Indonesian & English Consonants
    "b", "d", "dʒ", "f", "g", "h", "j", "k", "l", "m", "n", "ɲ", "ŋ",
    "p", "r", "s", "ʃ", "t", "tʃ", "v", "w", "z", "ʔ", "x", "θ", "ð", "ʒ"
]

# Pad vocabulary to exactly 128 tokens for clean tensor alignment
while len(UNIFIED_VOCAB) < 128:
    UNIFIED_VOCAB.append(f"<pad_{len(UNIFIED_VOCAB)}>")

VOCAB_INDEX: dict[str, int] = {sym: idx for idx, sym in enumerate(UNIFIED_VOCAB)}

# Known Indonesian words where 'e' is pepet /ə/ vs taling /e/
# Common prefixes that are consistently pepet: be-, me-, pe-, te-, se-, ke-
PEPET_PREFIXES = ("ber", "bel", "be", "men", "mem", "meng", "me", "per", "pel", "pe", "ter", "te", "se", "ke")

KNOWN_TALING = {
    "bebek", "meja", "sore", "capek", "sate", "tempe", "lele", "tante",
    "nenek", "kakek", "kolega", "model", "modern", "konser", "paket", "tiket",
    "presiden", "hotel", "kafe", "periode", "tema", "teks", "pesta", "desa",
    "daerah", "merah", "lewat", "pena", "belok", "tenda", "level", "elemen"
}

def _id_word_to_phonemes(w: str) -> str:
    """Convert an Indonesian word to IPA phonemes."""
    w_lower = w.lower()

    # Glottal stop on word-final 'k': bapak -> bapaʔ, tidak -> tidaʔ
    # Except if preceded by 'k' or part of prefix
    is_final_k = w_lower.endswith("k") and len(w_lower) > 2
    if is_final_k:
        w_lower = w_lower[:-1] + "ʔ"

    # Digraphs
    w_lower = w_lower.replace("ng", "ŋ")
    w_lower = w_lower.replace("ny", "ɲ")
    w_lower = w_lower.replace("sy", "ʃ")
    w_lower = w_lower.replace("kh", "x")

    # Diphthongs
    w_lower = w_lower.replace("ai", "aj")
    w_lower = w_lower.replace("au", "aw")
    w_lower = w_lower.replace("oi", "oj")

    # Pepet vs Taling resolution for 'e'
    # If the word is in KNOWN_TALING, keep /e/
    # If not, resolve prefixes as pepet /ə/
    if w.lower() in KNOWN_TALING:
        pass  # keep 'e'
    else:
        # Check prefixes
        chars = list(w_lower)
        for pref in PEPET_PREFIXES:
            if w.lower().startswith(pref):
                # The 'e' in prefix is pepet
                idx = pref.find("e")
                if idx != -1 and idx < len(chars):
                    chars[idx] = "ə"
                break

        # Word-initial or medial 'e' followed by consonant + vowel often pepet in native words
        # e.g., besar -> bəsar, beras -> bəras, empat -> əmpat, enam -> ənam, tepi -> təpi
        for i, c in enumerate(chars):
            if c == "e":
                # Default native Indonesian resolution: treat untagged 'e' as pepet /ə/
                chars[i] = "ə"
        w_lower = "".join(chars)

    return w_lower


def id_text_to_phonemes(text: str) -> str:
    """Convert normalized Indonesian text to IPA string."""
    tokens = re.findall(r"[\wʔ]+|[^\w\s]|\s+", text)
    result = []
    for tok in tokens:
        if tok.isspace() or tok in ".,!?-:;":
            result.append(tok)
        elif re.match(r"^\w+$", tok):
            result.append(_id_word_to_phonemes(tok))
        else:
            result.append(tok)
    return "".join(result)


def en_text_to_phonemes(text: str) -> str:
    """Convert English text to basic IPA phonemes."""
    # Simple rule-based English IPA mapper for core bilingual support
    en_rules = {
        "the": "ðə", "is": "ɪz", "are": "ɑr", "and": "ænd", "to": "tu", "of": "əv",
        "in": "ɪn", "that": "ðæt", "it": "ɪt", "for": "fɔr", "on": "ɑn", "with": "wɪð",
        "as": "æz", "at": "æt", "by": "baɪ", "this": "ðɪs", "fast": "fæst",
        "speech": "spitʃ", "clear": "klɪr", "natural": "nætʃərəl",
        "intelligence": "ɪntɛlɪdʒəns", "artificial": "ɑrtəfɪʃəl", "model": "mɑdəl"
    }
    words = text.split()
    res = []
    for w in words:
        punct = ""
        clean_w = w.lower()
        if clean_w and clean_w[-1] in ".,!?;:":
            punct = clean_w[-1]
            clean_w = clean_w[:-1]

        ph = en_rules.get(clean_w, clean_w)
        res.append(ph + punct)
    return " ".join(res)


def text_to_phonemes(text: str, lang: str = "id") -> str:
    """Dispatch text to phoneme generator by language."""
    if lang.lower().startswith("en"):
        return en_text_to_phonemes(text)
    return id_text_to_phonemes(text)


def phonemes_to_ids(phonemes: str) -> list[int]:
    """Map IPA phoneme string into integer IDs using VOCAB_INDEX."""
    ids = []
    i = 0
    symbols_sorted = sorted([k for k in VOCAB_INDEX.keys() if not k.startswith("<pad_")], key=len, reverse=True)

    while i < len(phonemes):
        matched = False
        for sym in symbols_sorted:
            if phonemes.startswith(sym, i):
                ids.append(VOCAB_INDEX[sym])
                i += len(sym)
                matched = True
                break
        if not matched:
            # Drop or map unknown symbol to blank space
            if phonemes[i] in VOCAB_INDEX:
                ids.append(VOCAB_INDEX[phonemes[i]])
            i += 1
    return ids


def ids_to_phonemes(ids: list[int]) -> str:
    """Map integer IDs back to IPA symbols."""
    return "".join(UNIFIED_VOCAB[i] for i in ids if 0 <= i < len(UNIFIED_VOCAB) and not UNIFIED_VOCAB[i].startswith("<pad_") and UNIFIED_VOCAB[i] != "_")


def save_vocab_config(config_path: str | Path = "config/phonemes.json") -> None:
    """Save the unified 128-token vocabulary to config/phonemes.json."""
    p = Path(config_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"vocab": UNIFIED_VOCAB, "size": len(UNIFIED_VOCAB)}, f, indent=2, ensure_ascii=False)


# Auto-export phonemes.json on import if config directory exists
try:
    save_vocab_config()
except Exception:
    pass
