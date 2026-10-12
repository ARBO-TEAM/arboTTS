"""ArboTTS Voice Cloning Suite.

Clones user voice from audio reference (suara.m4a / user_ref.wav)
using KittenTTS 2 in-context reference conditioning.
Generates comprehensive audition samples for Indonesian speech.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

torch.set_num_threads(8)

from kittenml import KittenTTS

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REF_AUDIO = Path("D:/kerjaan/belajar/tts/arboTTS/recordings/user_ref.wav")
DEFAULT_OUT_DIR = Path("D:/kerjaan/belajar/tts/arboTTS/audition_clone")

DEFAULT_REF_TEXT = (
    "Halo, ini contoh suara untuk merekam suara saya sendiri. Saya membaca beberapa "
    "kalimat biasa dengan tempo yang santai. Kemarin pagi saya bersepeda ke pasar, "
    "membeli beras dan beberapa ekor bebek untuk dimasak bersama keluarga. Apakah "
    "suara ini terdengar jelas? Oh tidak, ternyata langit mendung dan sebentar lagi "
    "hujan turun. Semoga komputer nanti bisa berbicara seperti suara saya sendiri, "
    "tidak kaku dan tidak terburu-buru."
)

AUDITION_TESTS = {
    "test_01_sapaan": (
        "Halo, ini adalah contoh suara saya sendiri hasil uji coba kloning kecerdasan buatan. "
        "Bagaimana menurutmu, apakah intonasinya sudah terdengar mirip?"
    ),
    "test_02_berita": (
        "Pemerintah kota mengumumkan rencana revitalisasi jalur sepeda dan trotoar publik "
        "untuk meningkatkan kenyamanan pejalan kaki."
    ),
    "test_03_santai": (
        "Kemarin sore saya sempat mampir ke toko buku di dekat taman, suasananya tenang "
        "dan ada banyak buku baru yang menarik."
    ),
    "test_04_angka_rupiah": (
        "Total belanjaan untuk kebutuhan dapur minggu ini sekitar tiga ratus tujuh puluh lima ribu rupiah."
    ),
}


def run_voice_clone(
    ref_audio: Path = DEFAULT_REF_AUDIO,
    ref_text: str = DEFAULT_REF_TEXT,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=======================================================")
    print("🎙️  ArboTTS Voice Cloning Audition Suite")
    print(f"Acuan Suara: {ref_audio}")
    print(f"Folder Hasil: {out_dir}")
    print("=======================================================\n")

    t0 = time.time()
    print("Loading Voice Cloning Model (KittenTTS 2)...", flush=True)
    m = KittenTTS("KittenML/kitten-tts-2", weights="emb4")
    print(f"Model loaded in {time.time() - t0:.1f}s\n", flush=True)

    for test_name, text in AUDITION_TESTS.items():
        print(f"--> Mensintesis [{test_name}]...")
        print(f"    Teks: \"{text}\"")
        t_start = time.time()

        audio = m.generate(
            text,
            reference=str(ref_audio),
            reference_text=ref_text,
            normalize=False,
        )
        elapsed = time.time() - t_start
        dur = len(audio) / m.sample_rate

        # Normalize audio levels
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val
        elif max_val > 0.01:
            audio = audio * 0.95

        out_file = out_dir / f"{test_name}.wav"
        sf.write(str(out_file), audio, m.sample_rate, subtype="PCM_16")

        rms = float(np.sqrt(np.mean(audio**2)))
        print(
            f"    ✅ Disimpan: {out_file.name} | Durasi: {dur:.2f}s | "
            f"Waktu Proses: {elapsed:.1f}s | RMS: {rms:.4f}\n",
            flush=True,
        )

    print("🎉 Seluruh uji coba kloning suara selesai dan tersimpan dengan sukses!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ArboTTS Voice Cloning Auditions")
    parser.add_argument("--ref", type=str, default=str(DEFAULT_REF_AUDIO), help="Path to reference audio")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory")
    args = parser.parse_args()

    run_voice_clone(ref_audio=Path(args.ref), out_dir=Path(args.out))
