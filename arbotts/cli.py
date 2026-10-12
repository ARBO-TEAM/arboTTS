"""CLI interface for ArboTTS."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from arbotts.synthesizer import ArboTTS


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="arbotts",
        description="ArboTTS: Ultra-lightweight Bilingual (Indonesian & English) Text-to-Speech (ONNX CPU Runtime)",
    )
    parser.add_argument("text", type=str, help="Text to speak")
    parser.add_argument("-o", "--output", type=str, default="output.wav", help="Output WAV path (default: output.wav)")
    parser.add_argument("-l", "--lang", type=str, default="id", choices=["id", "en"], help="Language code: 'id' or 'en' (default: 'id')")
    parser.add_argument("-s", "--speed", type=float, default=0.85, help="Speech speed (default: 0.85 for relaxed announcer tempo)")
    parser.add_argument("-m", "--model", type=str, default=None, help="Custom ONNX model path (optional)")
    parser.add_argument("-c", "--config", type=str, default=None, help="Custom config json path (optional)")

    args = parser.parse_args()

    tts = ArboTTS(model_path=args.model, config_path=args.config)
    dur = tts.speak(args.text, args.output, lang=args.lang, speed=args.speed)
    print(f"[ArboTTS] Synthesized successfully -> {args.output} ({dur:.2f} s)")


if __name__ == "__main__":
    main()
