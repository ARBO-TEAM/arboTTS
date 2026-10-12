"""CLI interface for ArboTTS."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from arbotts.synthesizer import ArboTTS
from arbotts.morpher import load_profiles


def main() -> None:
    # If first argument is 'serve', delegate to server
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        from arbotts.server import main as server_main

        sys.argv.pop(1)
        server_main()
        return

    parser = argparse.ArgumentParser(
        prog="arbotts",
        description="ArboTTS: Ultra-lightweight Bilingual (Indonesian & English) Text-to-Speech (ONNX CPU Runtime)",
    )
    parser.add_argument("text", type=str, nargs="?", default=None, help="Text to speak")
    parser.add_argument("-o", "--output", type=str, default="output.wav", help="Output WAV path (default: output.wav)")
    parser.add_argument("-l", "--lang", type=str, default="id", choices=["id", "en"], help="Language code: 'id' or 'en' (default: 'id')")
    available_speakers = list(load_profiles().get("speakers", {}).keys()) or [
        "arbo", "bimo", "dimas", "danu", "arba", "sita", "maya"
    ]
    parser.add_argument("-sp", "--speaker", type=str, default="arbo", choices=available_speakers, help=f"Speaker profile: {', '.join(available_speakers)} (default: 'arbo')")
    parser.add_argument("-s", "--speed", type=float, default=0.85, help="Speech speed (default: 0.85 for relaxed announcer tempo)")
    parser.add_argument("-t", "--tier", type=str, default="standard", choices=["standard", "small", "tiny"], help="Model tier: 'standard' (60MB), 'small' (~24MB), or 'tiny' (~12MB) (default: 'standard')")
    parser.add_argument("-m", "--model", type=str, default=None, help="Custom ONNX model path (optional)")
    parser.add_argument("-c", "--config", type=str, default=None, help="Custom config json path (optional)")

    args = parser.parse_args()

    if not args.text:
        parser.print_help()
        print("\nTip: Run 'arbotts serve' to launch the interactive local Web UI Studio.")
        return

    tts = ArboTTS(model_path=args.model, config_path=args.config, tier=args.tier)
    dur = tts.speak(args.text, args.output, lang=args.lang, speaker=args.speaker, speed=args.speed)
    print(f"[ArboTTS] Synthesized successfully ({args.speaker}) -> {args.output} ({dur:.2f} s)")


if __name__ == "__main__":
    main()
