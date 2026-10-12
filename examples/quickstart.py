"""Quickstart example for ArboTTS (Bilingual: Indonesian & English)."""

import sys
from pathlib import Path

# Add repo root to path for standalone execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arbotts import ArboTTS


def main():
    print("=== ArboTTS Bilingual Quickstart Example ===")

    # 1. Initialize ArboTTS
    # Model is automatically resolved from local repo or downloaded from GitHub Releases
    tts = ArboTTS()

    # 2. Sentences to synthesize (Indonesian and English)
    items = [
        # Indonesian
        ("id_halo_pagi.wav", "Halo, selamat pagi, Indonesia. Awali hari Seninmu dengan penuh ceria!", "id"),
        ("id_berita.wav", "Harga beras medium di pasar tradisional mencapai empat belas ribu rupiah per kilogram.", "id"),
        # English
        ("en_good_morning.wav", "Good morning, and welcome to ArboTTS. Have a wonderful and productive day!", "en"),
        ("en_tech_overview.wav", "ArboTTS is an ultra lightweight text to speech model running entirely on CPU.", "en"),
    ]

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    for filename, text, lang in items:
        out_path = out_dir / filename
        duration = tts.speak(text, out_path, lang=lang, speed=0.85)
        print(f"[{lang.upper()}] Generated: {out_path} ({duration:.2f} s) -> '{text}'")


if __name__ == "__main__":
    main()
