"""Record an automated demo video of ArboTTS Studio Web UI with synchronized audio using Playwright & FFmpeg."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parent.parent
RECORD_DIR = REPO_ROOT / "recordings"
RAW_DIR = RECORD_DIR / "raw"
OUTPUT_MP4 = RECORD_DIR / "arbotts_studio_demo.mp4"


def run_recording():
    # Clean previous raw dir
    if RAW_DIR.exists():
        shutil.rmtree(RAW_DIR)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    audio_events: list[dict] = []
    response_counter = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--autoplay-policy=no-user-gesture-required"]
        )

        context = browser.new_context(
            record_video_dir=str(RAW_DIR),
            record_video_size={"width": 1280, "height": 800},
            viewport={"width": 1280, "height": 800},
            device_scale_factor=1,
        )

        page = context.new_page()

        print("[1/5] Opening ArboTTS Studio at http://localhost:8000...")
        t_start = time.time()
        page.goto("http://localhost:8000", wait_until="networkidle")
        page.wait_for_timeout(1500)

        demo_steps = [
            {
                "speaker": "arbo",
                "text": "Halo, selamat pagi, Indonesia. Awali hari Seninmu dengan penuh ceria!",
                "speed": 0.85,
                "pause_after": 6000,
            },
            {
                "speaker": "dimas",
                "text": "Hai semuanya! Saya Dimas, asisten muda yang siap menemani hari kalian!",
                "speed": 0.90,
                "pause_after": 5500,
            },
            {
                "speaker": "bimo",
                "text": "Selamat datang di dokumenter resmi ArboTTS. Saya Bimo, narator baritone berwibawa.",
                "speed": 0.85,
                "pause_after": 6500,
            },
            {
                "speaker": "arba",
                "text": "Berikut adalah laporan berita terkini seputar perkembangan kecerdasan buatan di Indonesia.",
                "speed": 0.85,
                "pause_after": 6500,
            },
        ]

        for idx, step in enumerate(demo_steps, 1):
            spk = step["speaker"]
            print(f"[2/5] Running Demo Step {idx}: Speaker '{spk}'...")

            # Save audio clip directly from API
            wav_path = RAW_DIR / f"audio_{idx}.wav"
            res = page.request.post(
                "http://localhost:8000/v1/audio/speech",
                data={"input": step["text"], "voice": spk, "speed": step["speed"]},
            )
            wav_path.write_bytes(res.body())

            # Click speaker card
            spk_btn = page.locator(f"button.speaker-btn[onclick*=\"'{spk}'\"]")
            if spk_btn.count() > 0:
                spk_btn.first.click()
            page.wait_for_timeout(600)

            # Fill text
            text_input = page.locator("#textInput")
            text_input.fill("")
            page.wait_for_timeout(300)
            text_input.press_sequentially(step["text"], delay=20)
            page.wait_for_timeout(600)

            # Click generate
            gen_btn = page.locator("#genBtn")
            gen_btn.click()

            # Wait for output section and audio player
            page.wait_for_selector("#outputSection[style*='display: block']", timeout=15000)
            audio_start_time = time.time() - t_start

            audio_events.append({
                "id": idx,
                "file": wav_path,
                "play_timestamp": audio_start_time,
            })
            print(f"      Audio {idx} playing at video timestamp: {audio_start_time:.2f}s")

            # Let the audio play in UI
            page.wait_for_timeout(step["pause_after"])

        page.wait_for_timeout(2000)
        context.close()
        video_raw_path = page.video.path()
        browser.close()

    print(f"[3/5] Raw video recorded to: {video_raw_path}")
    print(f"[4/5] Audio clips captured: {len(audio_events)}")

    # Mux video with audio clips via FFmpeg
    mux_video(video_raw_path, audio_events, OUTPUT_MP4)


def mux_video(video_path: str, audio_events: list[dict], output_mp4: Path):
    print(f"[5/5] Muxing video and audio with FFmpeg -> {output_mp4}...")
    inputs = ["-i", video_path]
    filter_parts = []
    mix_inputs = []

    for idx, ev in enumerate(audio_events):
        wav = str(ev["file"])
        delay_ms = int(round(ev.get("play_timestamp", 0.0) * 1000))
        input_idx = idx + 1
        inputs.extend(["-i", wav])
        filter_parts.append(f"[{input_idx}:a]adelay={delay_ms}|{delay_ms}[a{input_idx}]")
        mix_inputs.append(f"[a{input_idx}]")

    if mix_inputs:
        filter_parts.append(f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:normalize=0[aout]")
        filter_complex = ";".join(filter_parts)

        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "22",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            str(output_mp4),
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "22",
            "-pix_fmt", "yuv420p",
            str(output_mp4),
        ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("FFmpeg error:", res.stderr)
        raise RuntimeError("FFmpeg muxing failed.")

    print(f"\n[DONE] Demo video generated successfully!")
    print(f"File: {output_mp4}")
    print(f"Size: {output_mp4.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    run_recording()
