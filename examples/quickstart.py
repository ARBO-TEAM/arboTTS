"""Quickstart example for ArboTTS."""

import sys
from pathlib import Path

# Add repo root to path for standalone execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arbotts import ArboTTS

def main():
    print("=== ArboTTS Quickstart Example ===")

    # 1. Initialize ArboTTS
    # Model is automatically resolved or downloaded from GitHub Releases
    tts = ArboTTS()

    # 2. Sentences to synthesize
    sentences = [
        ("halo_pagi.wav", "Halo, selamat pagi, Indonesia. Awali hari Senin-mu dengan penuh ceria!"),
        ("berita_pasar.wav", "Harga beras medium di pasar tradisional mencapai empat belas ribu rupiah per kilogram."),
        ("bebek_danau.wav", "Bebek-bebek kecil berenang tenang di tepi kolam belakang rumah."),
    ]

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    for filename, text in sentences:
        out_path = out_dir / filename
        duration = tts.speak(text, out_path, speed=0.85)
        print(f"Generated: {out_path} ({duration:.2f} s) -> '{text}'")

if __name__ == "__main__":
    main()
