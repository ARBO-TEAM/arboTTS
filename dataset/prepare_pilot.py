"""Prepare the 850-clip ArboTTS pilot dataset.

Reads mp3 clips from the sanoTTS id-male corpus zips, applies the Arbo v4
broadcast-male profile (mono 22,050 Hz, -2.0 semitone pitch shift, 0.95 peak
normalization), writes 16-bit PCM WAVs to dataset/wavs/, and emits
dataset/train_pilot.txt (800 rows) and dataset/val_pilot.txt (50 rows) in the
`dataset/wavs/{clip_id}.wav|{phonemes}` format.
"""

from __future__ import annotations

import io
import json
import random
import re
import sys
import zipfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SANOTTS = Path(r"D:\kerjaan\belajar\tts\sanoTTS")
FULL_ZIP = SANOTTS / "id-male-corpus-full.zip"
PILOT_ZIP = SANOTTS / "id-male-corpus-pilot.zip"
APSIPA_JSONL = SANOTTS / "data" / "textsets" / "apsipa-distill-v1" / "id_ID.train.jsonl"
MC4_JSONL = SANOTTS / "data" / "textsets" / "mc4-id-clean-v2.jsonl"

OUT_DIR = REPO_ROOT / "dataset"
WAV_DIR = OUT_DIR / "wavs"
TRAIN_FILE = OUT_DIR / "train_pilot.txt"
VAL_FILE = OUT_DIR / "val_pilot.txt"

N_CLIPS = 850
N_TRAIN = 800
SEED = 42
TARGET_SR = 22050
PITCH_SEMITONES = -2.0
PEAK_TARGET = 0.95
MIN_DURATION_S = 0.4

# Basic Indonesian orthography -> IPA fallback (used until
# arbotts.frontend.g2p ships; longest keys first so digraphs win).
_IPA_RULES = {
    "ai": "aj", "au": "aw", "oi": "oj", "ng": "ŋ", "ny": "ɲ", "sy": "ʃ",
    "kh": "x", "dj": "dʒ", "c": "tʃ", "j": "dʒ", "q": "k", "x": "ks",
    "y": "j", "a": "a", "i": "i", "u": "u", "e": "ə", "o": "o", "b": "b",
    "d": "d", "f": "f", "g": "g", "h": "h", "k": "k", "l": "l", "m": "m",
    "n": "n", "p": "p", "r": "r", "s": "s", "t": "t", "v": "v", "w": "w",
    "z": "z",
}
_IPA_RE = re.compile("|".join(re.escape(k) for k in sorted(_IPA_RULES, key=len, reverse=True)))
_STRIP_RE = re.compile(r"[^a-z\s']+")


def basic_ipa(text: str) -> str:
    """Minimal rule-based IPA pass over Indonesian orthography."""
    cleaned = _STRIP_RE.sub(" ", text.lower())
    words = []
    for word in cleaned.split():
        words.append(_IPA_RE.sub(lambda m: _IPA_RULES[m.group(0)], word))
    return " ".join(words)


def _load_g2p():
    try:
        from arbotts.frontend.g2p import text_to_phonemes  # noqa: PLC0415
        return text_to_phonemes
    except ImportError:
        return None


def phonemize(text: str, g2p) -> str:
    if g2p is not None:
        try:
            return g2p(text, lang="id")
        except Exception:
            pass
    return basic_ipa(text)


def load_transcripts() -> dict[str, str]:
    """Merge id_ID (apsipa) and mc4 transcripts keyed by clip id."""
    transcripts: dict[str, str] = {}
    for path in (APSIPA_JSONL, MC4_JSONL):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    transcripts[rec["id"]] = rec["text"]
    return transcripts


def pick_zip() -> Path:
    if FULL_ZIP.exists():
        return FULL_ZIP
    if PILOT_ZIP.exists():
        return PILOT_ZIP
    raise FileNotFoundError(f"no corpus zip found: {FULL_ZIP} or {PILOT_ZIP}")


def decode_mono22k(data: bytes) -> np.ndarray | None:
    """Decode mp3 bytes to mono float32 at 22,050 Hz."""
    try:
        y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception:
        y, sr = librosa.load(io.BytesIO(data), sr=None, mono=False)
        y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1 if y.shape[1] <= y.shape[0] else 0)
    if sr != TARGET_SR:
        y = librosa.resample(np.asarray(y, dtype=np.float32), orig_sr=sr, target_sr=TARGET_SR)
    return np.asarray(y, dtype=np.float32)


def morph(y: np.ndarray) -> np.ndarray | None:
    """Pitch-shift -2.0 semitones (Arbo v4 male profile), peak-normalize 0.95."""
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    if len(y) < int(MIN_DURATION_S * TARGET_SR):
        return None
    y = librosa.effects.pitch_shift(y, sr=TARGET_SR, n_steps=PITCH_SEMITONES)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak < 1e-6:
        return None
    return (y * (PEAK_TARGET / peak)).astype(np.float32)


def process_clip(zf: zipfile.ZipFile, clip_id: str, text: str, g2p) -> tuple[str, str] | None:
    wav_path = WAV_DIR / f"{clip_id}.wav"
    if not (wav_path.exists() and wav_path.stat().st_size > 0):
        y = decode_mono22k(zf.read(f"{clip_id}.mp3"))
        if y is None:
            return None
        y = morph(y)
        if y is None:
            return None
        sf.write(str(wav_path), y, TARGET_SR, subtype="PCM_16")
    return clip_id, phonemize(text, g2p)


def main() -> None:
    transcripts = load_transcripts()
    zip_path = pick_zip()
    g2p = _load_g2p()
    print(f"zip: {zip_path.name} | transcripts: {len(transcripts)} | "
          f"g2p: {'arbotts.frontend.g2p' if g2p else 'basic IPA fallback'}")

    with zipfile.ZipFile(zip_path) as zf:
        names = {n[:-4] for n in zf.namelist() if n.endswith(".mp3")}
        pool = sorted(names & transcripts.keys())
        random.Random(SEED).shuffle(pool)

        WAV_DIR.mkdir(parents=True, exist_ok=True)
        rows: list[tuple[str, str]] = []
        skipped = 0
        for clip_id in tqdm(pool, desc="prepare"):
            if len(rows) >= N_CLIPS:
                break
            result = process_clip(zf, clip_id, transcripts[clip_id], g2p)
            if result is None:
                skipped += 1
                continue
            rows.append(result)

    if len(rows) < N_CLIPS:
        raise RuntimeError(f"only {len(rows)}/{N_CLIPS} clips could be prepared")

    train_rows, val_rows = rows[:N_TRAIN], rows[N_TRAIN:]
    TRAIN_FILE.write_text(
        "".join(f"dataset/wavs/{cid}.wav|{ph}\n" for cid, ph in train_rows),
        encoding="utf-8")
    VAL_FILE.write_text(
        "".join(f"dataset/wavs/{cid}.wav|{ph}\n" for cid, ph in val_rows),
        encoding="utf-8")
    print(f"wrote {len(train_rows)} train + {len(val_rows)} val rows "
          f"({skipped} clips skipped)")


if __name__ == "__main__":
    main()
