"""Generate paired Arba (female presenter) audio dataset using Edge-TTS + Acoustic Morphing.

Internal distillation pipeline:
- Voice: id-ID-GadisNeural
- Target Sample Rate: 22,050 Hz (16-bit PCM mono)
- Acoustic Morphing: -0.8 semitones pitch shift, 3,500 Hz presence boost (+25%), peak normalized to 0.95
- Paired with Arbo (male) dataset into dataset/train_multispeaker.txt
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
import sys

REPO_ROOT = Path("D:/kerjaan/belajar/tts/arboTTS").resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import edge_tts
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from arbotts.morpher import morph_audio, DEFAULT_SPEAKERS

DATASET_DIR = REPO_ROOT / "dataset"
WAV_DIR = DATASET_DIR / "wavs"
RAW_TRAIN_FILE = DATASET_DIR / "train_pilot_raw.txt"
MULTISPK_TRAIN_FILE = DATASET_DIR / "train_multispeaker.txt"

N_ARBA_SAMPLES = 200
VOICE_NAME = "id-ID-GadisNeural"
TARGET_SR = 22050


async def synthesize_one(text: str, sem: asyncio.Semaphore) -> tuple[np.ndarray | None, int]:
    async with sem:
        try:
            communicate = edge_tts.Communicate(text, VOICE_NAME)
            mp3_bytes = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_bytes += chunk["data"]
            if not mp3_bytes:
                return None, 0
            audio, sr = sf.read(io.BytesIO(mp3_bytes), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return audio, sr
        except Exception as e:
            print(f"[WARN] Failed to synthesize: {e}", file=sys.stderr)
            return None, 0


async def generate_arba_corpus():
    WAV_DIR.mkdir(parents=True, exist_ok=True)

    with open(RAW_TRAIN_FILE, "r", encoding="utf-8") as f:
        raw_lines = [line.strip().split("|") for line in f if line.strip()]

    print(f"Loaded {len(raw_lines)} sentences from {RAW_TRAIN_FILE.name}")
    arba_subset = raw_lines[:N_ARBA_SAMPLES]
    print(f"Targeting {len(arba_subset)} paired sentences for Arba (Speaker 1)")

    recipe = DEFAULT_SPEAKERS["speakers"]["arba"]["dsp_recipe"]
    sem = asyncio.Semaphore(5)

    manifest_rows: list[str] = []

    # First, add all existing Arbo (Speaker 0) rows
    for row in raw_lines:
        wav_rel, text = row[0], row[1]
        manifest_rows.append(f"{wav_rel}|0|{text}")

    print("Synthesizing and morphing Arba audio in parallel...")
    sem = asyncio.Semaphore(8)

    async def process_one(idx: int, text: str):
        out_wav_name = f"arba_{idx:04d}.wav"
        out_wav_rel = f"dataset/wavs/{out_wav_name}"
        out_wav_full = WAV_DIR / out_wav_name

        if out_wav_full.exists() and out_wav_full.stat().st_size > 1000:
            return out_wav_rel, text, True

        raw_audio, sr = await synthesize_one(text, sem)
        if raw_audio is None or len(raw_audio) < 100:
            print(f"[WARN] Skipping empty audio for sentence {idx}", file=sys.stderr)
            return out_wav_rel, text, False

        # Resample if needed
        if sr != TARGET_SR:
            raw_audio = librosa.resample(raw_audio, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR

        # Apply acoustic morphing (-0.8 semitones, 3500 Hz presence boost)
        morphed, _ = morph_audio(raw_audio, sr, recipe)

        # Write 16-bit PCM WAV
        sf.write(str(out_wav_full), morphed, TARGET_SR, subtype="PCM_16")
        return out_wav_rel, text, True

    tasks = [process_one(idx, text) for idx, (_, text) in enumerate(arba_subset)]
    results = await asyncio.gather(*tasks)

    generated_count = 0
    for out_wav_rel, text, ok in results:
        if ok:
            manifest_rows.append(f"{out_wav_rel}|1|{text}")
            generated_count += 1

    # Write multi-speaker manifest
    with open(MULTISPK_TRAIN_FILE, "w", encoding="utf-8") as f:
        for r in manifest_rows:
            f.write(f"{r}\n")

    print(f"\nSuccessfully created {MULTISPK_TRAIN_FILE.name}:")
    print(f"  - Total samples: {len(manifest_rows)}")
    print(f"  - Arbo (Speaker 0): {len(raw_lines)}")
    print(f"  - Arba (Speaker 1): {generated_count}")


if __name__ == "__main__":
    asyncio.run(generate_arba_corpus())
