"""Text normalization for Indonesian and English speech synthesis."""

from __future__ import annotations

import re

# Indonesian number words
_ID_ONES = [
    "", "satu", "dua", "tiga", "empat", "lima",
    "enam", "tujuh", "delapan", "sembilan", "sepuluh", "sebelas"
]

def _id_num_to_words(n: int) -> str:
    """Convert integer up to trillions into Indonesian words."""
    if n < 0:
        return "minus " + _id_num_to_words(-n)
    if n == 0:
        return "nol"
    if n < 12:
        return _ID_ONES[n]
    if n < 20:
        return _id_num_to_words(n - 10) + " belas"
    if n < 100:
        tens, rem = divmod(n, 10)
        return (_ID_ONES[tens] + " puluh" + (" " + _id_num_to_words(rem) if rem else "")).strip()
    if n < 200:
        return ("seratus" + (" " + _id_num_to_words(n - 100) if n > 100 else "")).strip()
    if n < 1000:
        hundreds, rem = divmod(n, 100)
        return (_ID_ONES[hundreds] + " ratus" + (" " + _id_num_to_words(rem) if rem else "")).strip()
    if n < 2000:
        return ("seribu" + (" " + _id_num_to_words(n - 1000) if n > 1000 else "")).strip()
    if n < 1_000_000:
        thousands, rem = divmod(n, 1000)
        return (_id_num_to_words(thousands) + " ribu" + (" " + _id_num_to_words(rem) if rem else "")).strip()
    if n < 1_000_000_000:
        millions, rem = divmod(n, 1_000_000)
        return (_id_num_to_words(millions) + " juta" + (" " + _id_num_to_words(rem) if rem else "")).strip()
    if n < 1_000_000_000_000:
        billions, rem = divmod(n, 1_000_000_000)
        return (_id_num_to_words(billions) + " miliar" + (" " + _id_num_to_words(rem) if rem else "")).strip()
    trillions, rem = divmod(n, 1_000_000_000_000)
    return (_id_num_to_words(trillions) + " triliun" + (" " + _id_num_to_words(rem) if rem else "")).strip()


def normalize_id_text(text: str) -> str:
    """Normalize Indonesian text: expand currency, numbers, dates, punctuation."""
    t = text.strip()

    # Expand Indonesian Currency: Rp 750.000, Rp. 1.500.000, Rp20000
    def _expand_rupiah(m: re.Match) -> str:
        num_str = m.group(1).replace(".", "").replace(",", "")
        try:
            val = int(num_str)
            return _id_num_to_words(val) + " rupiah"
        except ValueError:
            return m.group(0)

    t = re.sub(r"(?:Rp\.?|rupiah\s*)\s*(\d[\d\.,]*)", _expand_rupiah, t, flags=re.IGNORECASE)

    # Decimal numbers: 3,5 -> tiga koma lima
    def _expand_decimal(m: re.Match) -> str:
        whole, frac = m.group(1), m.group(2)
        try:
            w_w = _id_num_to_words(int(whole))
            f_w = " ".join(_ID_ONES[int(c)] if int(c) < 12 else c for c in frac)
            return f"{w_w} koma {f_w}"
        except Exception:
            return m.group(0)

    t = re.sub(r"\b(\d+),(\d+)\b", _expand_decimal, t)

    # Standard thousand-separated or plain integers: 16, 2004, 750.000
    def _expand_integer(m: re.Match) -> str:
        clean = m.group(0).replace(".", "")
        try:
            return _id_num_to_words(int(clean))
        except ValueError:
            return m.group(0)

    t = re.sub(r"\b\d{1,3}(?:\.\d{3})+\b", _expand_integer, t)
    t = re.sub(r"\b\d+\b", _expand_integer, t)

    # Clean redundant whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


# English basic number conversion
_EN_ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
            "seventeen", "eighteen", "nineteen"]
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

def _en_num_to_words(n: int) -> str:
    if n < 0:
        return "minus " + _en_num_to_words(-n)
    if n == 0:
        return "zero"
    if n < 20:
        return _EN_ONES[n]
    if n < 100:
        tens, rem = divmod(n, 10)
        return (_EN_TENS[tens] + (" " + _EN_ONES[rem] if rem else "")).strip()
    if n < 1000:
        hundreds, rem = divmod(n, 100)
        return (_EN_ONES[hundreds] + " hundred" + (" " + _en_num_to_words(rem) if rem else "")).strip()
    if n < 1_000_000:
        thousands, rem = divmod(n, 1000)
        return (_en_num_to_words(thousands) + " thousand" + (" " + _en_num_to_words(rem) if rem else "")).strip()
    millions, rem = divmod(n, 1_000_000)
    return (_en_num_to_words(millions) + " million" + (" " + _en_num_to_words(rem) if rem else "")).strip()


def normalize_en_text(text: str) -> str:
    """Normalize English text: numbers and common symbols."""
    t = text.strip()

    # Currency: $50 -> fifty dollars
    def _expand_dollar(m: re.Match) -> str:
        num_str = m.group(1).replace(",", "")
        try:
            return _en_num_to_words(int(num_str)) + " dollars"
        except ValueError:
            return m.group(0)

    t = re.sub(r"\$(\d[\d,]*)", _expand_dollar, t)

    def _expand_en_int(m: re.Match) -> str:
        clean = m.group(0).replace(",", "")
        try:
            return _en_num_to_words(int(clean))
        except ValueError:
            return m.group(0)

    t = re.sub(r"\b\d+\b", _expand_en_int, t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalize_text(text: str, lang: str = "id") -> str:
    """Normalize text dispatching on language code."""
    if lang.lower().startswith("en"):
        return normalize_en_text(text)
    return normalize_id_text(text)
