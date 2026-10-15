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
    parser.add_argument("-t", "--tier", type=str, default="standard", choices=["standard", "small", "tiny", "nano", "paradee", "kokoro", "studio_clone", "mcu", "bilingual", "micro"], help="Model tier: 'mcu' / 'bilingual' (~1.55MB INT8 C99 for ESP32/embedded), 'nano' / 'paradee' (~8.6MB), 'standard' (60MB), 'small' (~24MB), 'tiny' (~12MB), 'kokoro' (~88MB ONNX 24kHz), or 'studio_clone' (~550MB in-context) (default: 'standard')")
    parser.add_argument("--reference", "-ref", type=str, default=None, help="Path to reference audio file for zero-shot voice cloning (e.g. sample.wav)")
    parser.add_argument("--reference-text", "-rt", type=str, default=None, help="Transcript of reference audio (optional, auto-transcribed if omitted)")
    parser.add_argument("-m", "--model", type=str, default=None, help="Custom ONNX model path (optional)")
    parser.add_argument("-c", "--config", type=str, default=None, help="Custom config json path (optional)")

    args = parser.parse_args()

    if not args.text:
        parser.print_help()
        print("\nTip: Run 'arbotts serve' to launch the interactive local Web UI Studio.")
        return

    if args.tier in ("mcu", "bilingual", "micro"):
        from arbotts.mcu_engine import BilingualMCUSynthesizer

        mcu_engine = BilingualMCUSynthesizer(bundle_path=args.model)
        latency_ms = mcu_engine.save(
            text=args.text,
            output_path=args.output,
            speed=args.speed if args.speed != 0.85 else None,
            lang=args.lang,
        )
        print(f"[ArboTTS Bilingual MCU ~1.55MB INT8] Synthesized successfully ({args.lang}) -> {args.output} (Latency: {latency_ms:.1f}ms, 22.05 kHz)")
        return

    if args.tier in ("nano", "paradee"):
        from arbotts.paradee_engine import ParadeeONNXSynthesizer

        paradee_engine = ParadeeONNXSynthesizer(model_path=args.model)
        latency_ms = paradee_engine.save(
            text=args.text,
            output_path=args.output,
            speed=args.speed,
        )
        print(f"[ArboTTS Paradee-8M Nano ~8.6MB] Synthesized successfully -> {args.output} (Latency: {latency_ms:.1f}ms, 24 kHz)")
        return

    if args.tier == "kokoro":
        from arbotts.kokoro_engine import KokoroONNXSynthesizer

        voice = args.reference or args.speaker
        kokoro_engine = KokoroONNXSynthesizer()
        dur = kokoro_engine.speak(
            text=args.text,
            output_path=args.output,
            voice=voice,
            speed=args.speed,
            lang=args.lang,
        )
        print(f"[ArboTTS Kokoro ONNX ~88MB] Synthesized successfully ({voice}) -> {args.output} ({dur:.2f} s, 24 kHz)")
        return

    if args.reference or args.tier == "studio_clone":
        if not args.reference:
            print("Error: Voice cloning requires --reference audio path.", file=sys.stderr)
            sys.exit(1)
        if args.tier == "studio_clone":
            from arbotts.clone import StudioClone

            clone_engine = StudioClone()
            audio, sr, latency_ms = clone_engine.clone(
                text=args.text,
                reference=args.reference,
                reference_text=args.reference_text,
                speed=args.speed,
            )
            tier_name = "Studio Clone ~550MB"
        else:
            from arbotts.clone import AutoClone

            clone_engine = AutoClone(tier=args.tier)
            audio, sr, latency_ms = clone_engine.clone(
                text=args.text,
                reference=args.reference,
                speed=args.speed,
            )
            tier_name = "Auto Clone ~96MB"

        import soundfile as sf
        sf.write(args.output, audio, sr)
        dur = len(audio) / sr
        print(f"[ArboTTS {tier_name}] Cloned successfully from {args.reference} -> {args.output} ({dur:.2f} s in {latency_ms/1000:.2f} s)")
        return

    tts = ArboTTS(model_path=args.model, config_path=args.config, tier=args.tier)
    dur = tts.speak(args.text, args.output, lang=args.lang, speaker=args.speaker, speed=args.speed)
    print(f"[ArboTTS] Synthesized successfully ({args.speaker}) -> {args.output} ({dur:.2f} s)")


if __name__ == "__main__":
    main()
