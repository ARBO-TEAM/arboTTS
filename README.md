# ArboTTS (v2.5) 🎙️🇮🇩🇬🇧

**English** | [Bahasa Indonesia](README-ID.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Live Demo: GitHub Pages](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages%20(WASM)-38bdf8.svg)](https://arbo-team.github.io/arbotts-demo/)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![MCU Size: 1.55 MB](https://img.shields.io/badge/MCU%20Size-1.55%20MB%20INT8-brightgreen.svg)](https://github.com/ARBO-TEAM/arboTTS/releases)
[![Nano Size: 8.6 MB](https://img.shields.io/badge/Nano%20Size-8.62%20MB-orange.svg)](https://github.com/ARBO-TEAM/arboTTS/releases)
[![Inference: CPU](https://img.shields.io/badge/Inference-Pure%20CPU%20(RTF%200.025x)-purple.svg)](#-model-specifications--parameter-breakdown)

> 🌐 **Try it Live (100% Client-Side WebAssembly)**: Experience instant Indonesian TTS in your browser with zero server cost at **[arbo-team.github.io/arbotts-demo](https://arbo-team.github.io/arbotts-demo/)** (Source: [arbotts-demo](https://github.com/ARBO-TEAM/arbotts-demo)). No backend, no API keys, and complete privacy!

**ArboTTS** is an ultra-lightweight, production-grade bilingual (**Indonesian & English**) Text-to-Speech (TTS) engine powered by **VITS**, **Kokoro**, distilled **Paradee-8M**, and embedded **Piperlite C99 INT8** architectures, optimized for standalone CPU and microcontroller inference.

With version 2.5, ArboTTS brings the **Bilingual MCU INT8 tier (~1.55 MB, 22.05 kHz)** with zero PSRAM requirements on ESP32-S3, dual native Indonesian and English synthesis, and auto-cadence pacing (1.25 length_scale for English), alongside the **Nano Paradee-8M INT8 tier (8.62 MB, 24 kHz)** for browser WASM and edge devices, **Kokoro ONNX 24 kHz tier** with native Indonesian voice packs (`id_arbo`, `id_arba`), **Zero-Shot Voice Cloning on CPU** with flow-based ToneColorConverter, **High-Definition Multi-Persona Studio Synthesis**, an interactive browser-based **Web UI Studio**, and an **OpenAI-compatible REST API** with multi-tier engine support.

Engineered for resource-constrained environments, ArboTTS runs smoothly on standard laptops, mini PCs, edge devices, and server instances without requiring a GPU.

---

## 🎭 Studio Voice Personas

ArboTTS v2.0 includes 7 professionally calibrated voice personas, categorized into **Primary Male Personas** (recommended for native acoustic resonance and cadence) and **Additional Female Personas**:

### ⭐ Primary Male Personas (Recommended)
| Persona | Gender | Voice Role & Tone | Best Used For | CLI Option |
| :--- | :--- | :--- | :--- | :--- |
| **Arbo** | 👨 Male | **Formal Broadcast Announcer**<br>Authoritative, crisp plosive consonants, deep resonance | News anchors, official announcements, formal keynotes | `-sp arbo` *(default)* |
| **Bimo** | 👨 Male | **Deep Baritone & Authority**<br>Rich chest resonance, heavy weight, mature timbre | Documentaries, corporate voiceovers, trailers | `-sp bimo` |
| **Dimas** | 👨 Male | **Young & Energetic Casual**<br>Friendly conversational tempo, crisp consonant articulation | Virtual assistants, podcasts, modern explainers | `-sp dimas` |
| **Danu** | 👨 Male | **Calm Storyteller & Narrator**<br>Velvety warmth, gentle pacing, polite cadence | Audiobooks, meditation, bedtime stories, documentaries | `-sp danu` |

### 👩 Additional Female Personas
| Persona | Gender | Voice Role & Tone | Best Used For | CLI Option |
| :--- | :--- | :--- | :--- | :--- |
| **Arba** | 👩 Female | **Professional News Presenter**<br>Articulate, clear, polite, warm mid-range articulation | News broadcasts, corporate explainers, reporting | `-sp arba` |
| **Sita** | 👩 Female | **Smart Interactive Assistant**<br>Bright, cheerful, engaging conversational tempo | Virtual assistants, customer care, interactive bots | `-sp sita` |
| **Maya** | 👩 Female | **Calm Storyteller & Narration**<br>Gentle, relaxed, velvety warmth with soft cadence | Audiobooks, meditation, bedtime stories, documentaries | `-sp maya` |

---

## ⚡ Architectural Tiers (Model Compression)

ArboTTS provides tiered models to balance acoustic fidelity against memory footprint and compute latency:

| Tier | Model File | Size (ONNX/C99) | Parameters | Architecture Config | Mean CPU Latency / RTF | Target Platform |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ⚡ **MCU / Bilingual** *(v2.5)* | `mcu_bilingual_arbo_kokoro_745k_q8.bin` | **1.55 MB** | **1.56 M** | Distilled Kokoro + Arbo Student + Piperlite iSTFT (22.05 kHz, C99 INT8) | ~30–45 ms (0.015x RTF) | Microcontrollers (ESP32-S3 zero PSRAM), IoT, embedded edge |
| 🪶 **Nano / Paradee** *(v2.4)* | `arbo_paradee_8m.onnx` | **8.62 MB** | **8.08 M** | Distilled Albert + iSTFTNet (24 kHz, INT8 Quantized) | ~40–50 ms (0.04x RTF, 24x realtime) | Ultra-compact, IoT, edge devices, zero-dep standalone |
| 🌸 **Kokoro ONNX** *(v2.3)* | `model_quantized.onnx` | **~88 MB** | **82 M** | StyleTTS 2 + iSTFTNet (24 kHz, INT8 Quantized) | ~400 ms (0.05x RTF) | Studio 24 kHz, native Indonesian cadence & zero-shot cloning |
| 🎙️ **Studio Clone** | `Qwen3-SpeechLM` | **~550 MB** | **1.7 B (1.58-bit)** | Autoregressive Neural Codec Speech LM | Sub-second (RTX GPU) / ~1.0x RTF | True zero-shot voice cloning (timbre + prosody) |
| 🚀 **Standard** *(Default)* | `arbotts_v1.onnx` | **60.57 MB** | **23.66 M** | Hidden 192, Filter 768, 6 Layers, ResBlock1 | ~100–140 ms (0.025x RTF) | Studio production, desktop, server |
| ⚡ **Small** | `arbotts_small.onnx` | **31.29 MB** | **8.39 M** | Hidden 128, Filter 512, 4 Layers, ResBlock2 | ~45–65 ms (0.012x RTF) | Fast chatbots, edge apps, lightweight servers |
| 🪶 **Tiny** *(Roadmap)* | `arbotts_tiny.onnx` | *~12.8 MB* | *~3.65 M* | Hidden 96, Filter 256, 3 Layers, ResBlock2 | ~20–30 ms (0.006x RTF) | Raspberry Pi, IoT, wearables, mobile devices |

---

## 🎧 Audio Samples (Listen Directly)

You can listen to generated speech samples directly in your browser:

| Language | Context | Text Transcripts | Audio Player & Links |
| :--- | :--- | :--- | :--- |
| 🇮🇩 **ID** | **Morning Greeting (Broadcast Announcer)** | *"Halo, selamat pagi, Indonesia. Awali hari Seninmu dengan penuh ceria!"* | <audio controls src="https://raw.githubusercontent.com/ARBO-TEAM/arboTTS/main/examples/audio/id_halo_pagi.wav"></audio><br>[🔊 Play WAV](examples/audio/id_halo_pagi.wav) |
| 🇮🇩 **ID** | **Formal News Broadcast** | *"Harga beras medium di pasar tradisional mencapai empat belas ribu rupiah per kilogram hari ini."* | <audio controls src="https://raw.githubusercontent.com/ARBO-TEAM/arboTTS/main/examples/audio/id_berita_pasar.wav"></audio><br>[🔊 Play WAV](examples/audio/id_berita_pasar.wav) |
| 🇬🇧 **EN** | **English Greeting** | *"Good morning, and welcome to ArboTTS. Have a wonderful and productive day!"* | <audio controls src="https://raw.githubusercontent.com/ARBO-TEAM/arboTTS/main/examples/audio/en_good_morning.wav"></audio><br>[🔊 Play WAV](examples/audio/en_good_morning.wav) |
| 🇬🇧 **EN** | **Technical Explainer** | *"ArboTTS is an ultra lightweight text to speech model running entirely on CPU."* | <audio controls src="https://raw.githubusercontent.com/ARBO-TEAM/arboTTS/main/examples/audio/en_lightweight.wav"></audio><br>[🔊 Play WAV](examples/audio/en_lightweight.wav) |

---

## ✨ Key Features

- 🎭 **Studio Multi-Persona Support (v2.0)**: Switch seamlessly between 7 calibrated personas (**Arbo**, **Bimo**, **Dimas**, **Danu**, **Arba**, **Sita**, **Maya**) covering formal broadcast, deep baritone, casual, interactive assistant, and warm narration.
- 🧬 **Zero-Shot Voice Cloning on CPU (v2.1)**: Instant timbre transfer allowing cloning of user voices from a short reference audio clip (< 10 seconds) with ToneColorConverter, supporting both Standard (60 MB) and Small (31 MB) base synthesis tiers in real-time.
- 🌐 **Interactive Web UI Studio & REST API**: Full browser studio at `http://localhost:8000` with dedicated Voice Cloning tab and drop-in OpenAI-compatible `/v1/audio/speech` streaming endpoint.
- ⚡ **Blazing Fast on CPU (RTF 0.025x)**: Generates 5.0 seconds of speech in just **~120–140 ms** on consumer laptop CPUs (>35x faster than real-time).
- 🪶 **Ultra-Compact Model (~60 MB)**: Total parameter count of only **23.66 million** (standalone ONNX file size **60.57 MB**). Extremely low memory footprint (< 150 MB RAM).
- 🇮🇩🇬🇧 **Bilingual Support (ID + EN)**: Native Indonesian announcer cadence with clear English multilingual synthesis.
- 🎙️ **Authoritative Broadcast Timbre**: Warm, clear male announcer voice tuned for natural prosody, cadence, and syllable rhythm.
- 📦 **Zero-Config Auto Downloader**: Automatically downloads model weights from GitHub Releases on first invocation to a local cache directory (`~/.cache/arbotts/`).
- 🛡️ **PyTorch-Free Production Inference**: Runtime dependencies are strictly minimal (`onnxruntime`, `numpy`, `soundfile`).
- 🔒 **TOS & Licensing Safe**: Synthesized via transformed acoustic weight adaptation (-2.0 semitone morphing + vocoder distillation), free of third-party cloning or re-hosting liabilities.

---

## 📊 Model Specifications & Parameter Breakdown

| Architectural Component | Parameter Count | Percentage | Function |
| :--- | :--- | :--- | :--- |
| **Normalizing Flow (`flow`)** | 8,076,144 (8.08 M) | 34.1% | Transforms latent phoneme representations to mel-spectrogram space |
| **Text Encoder (`enc_p`)** | 6,341,760 (6.34 M) | 26.8% | Text processing and multilingual phoneme representation |
| **HiFi-GAN Vocoder (`dec`)** | 1,664,768 (1.66 M) | 7.0% | High-fidelity waveform synthesis (ResBlock2) |
| **Duration Predictor (`dp`)** | 1,317,168 (1.32 M) | 5.6% | Syllable rhythm and phoneme duration modeling |
| **Total Model Parameters** | **23,663,792 (23.66 M)** | **100%** | **End-to-End Standalone VITS Architecture** |

### Acoustic & Performance Metrics
- **Model Format**: Standalone ONNX (Opset 15)
- **File Size**: `63,516,051 bytes` (~60.57 MB)
- **Sampling Rate**: `22,050 Hz` (16-bit PCM Mono)
- **Peak RAM Usage**: `< 150 MB`
- **Mean CPU Latency**: `~100–140 ms` per sentence (RTF 0.025x)

---

## 🚀 Installation

### 1. From Source
```bash
git clone https://github.com/ARBO-TEAM/arboTTS.git
cd arboTTS
pip install -e .
```

### 2. System Requirements
ArboTTS uses `espeak-ng` for multilingual phonemization. Ensure `espeakng-loader` or `espeak-ng` is available:
```bash
pip install espeakng-loader phonemizer
```

---

## 💻 Quickstart

### Python API
Synthesize natural speech in Indonesian or English with just a few lines:

```python
from arbotts import ArboTTS

# Initialize synthesizer (Standard tier by default, or specify tier="small")
tts_standard = ArboTTS(tier="standard")  # 60.57 MB - Full Fidelity
tts_small = ArboTTS(tier="small")        # 31.29 MB - Ultra-Fast Distilled

# 🇮🇩 Indonesian synthesis with primary persona
tts_standard.speak(
    text="Halo, selamat pagi, Indonesia. Awali hari Seninmu dengan penuh ceria!",
    output_file="halo_id.wav",
    speaker="arbo",
    lang="id",
    speed=0.85,
)

# ⚡ Small tier fast synthesis
tts_small.speak(
    text="Sistem siap melayani permintaan pengguna dengan latensi ultra rendah.",
    output_file="fast_reply.wav",
    speaker="arbo",
    lang="id",
    speed=0.85,
)

# 🪶 Nano Tier (Paradee-8M): Ultra-compact 24 kHz (~8.6 MB INT8, zero dependencies)
tts_nano = ArboTTS(tier="nano")
tts_nano.speak(
    text="Halo, ini adalah model Paradee delapan megabyte suara Ardi bahasa Indonesia.",
    output_file="nano_output.wav",
)

# ⚡ MCU Bilingual Tier (~1.55 MB INT8): Embedded ESP32-S3 zero-PSRAM (ID + EN)
tts_mcu = ArboTTS(tier="mcu")
tts_mcu.speak(
    text="Halo, ini suara bilingual mikrokontroler ArboTTS.",
    output_file="mcu_id.wav",
    lang="id",
)
tts_mcu.speak(
    text="Hello, this is bilingual English speech synthesis running on microcontroller.",
    output_file="mcu_en.wav",
    lang="en",  # Auto-pacing length_scale = 1.25 for relaxed English cadence
)

# 🌸 Kokoro ONNX tier: 24 kHz Studio Naturalness (~88 MB INT8 Quantized)
tts_kokoro = ArboTTS(tier="kokoro")
tts_kokoro.speak(
    text="Halo, selamat pagi Indonesia. Suara ini sangat ekspresif dan jernih.",
    output_file="kokoro_output.wav",
    speaker="arbo",
)

# 🇬🇧 English synthesis
tts_standard.speak(
    text="Good morning, and welcome to ArboTTS. Have a wonderful and productive day!",
    output_file="welcome_en.wav",
    speaker="arbo",
    lang="en",
    speed=0.85,
)

# 🎙️ True Zero-Shot Voice Cloning (~550MB Studio Clone Tier)
# Automatically clones both vocal timbre and conversational cadence without persona selection:
tts = ArboTTS(tier="studio_clone")
audio, latency_ms = tts.clone(
    text="Halo bro, ini contoh hasil kloning suara menggunakan arboTTS studio clone.",
    reference="path/to/my_voice.wav",
)
```

### Command Line Interface (CLI)
Once installed, the `arbotts` command is accessible directly from your terminal:

```bash
# 🇮🇩 Indonesian synthesis with Standard tier (default)
arbotts "Halo, selamat datang di ArboTTS!" -o halo_id.wav -l id

# 🪶 Nano tier (Paradee-8M ~8.6 MB INT8) ultra-compact synthesis
arbotts "Halo, ini suara Paradee delapan megabita ArboTTS." --tier nano -o nano_out.wav

# ⚡ MCU tier (~1.55 MB INT8 C99) ultra-lightweight bilingual synthesis
arbotts "Halo selamat pagi dunia." --tier mcu -l id -o mcu_id.wav
arbotts "Hello world, bilingual English speech on microcontroller." --tier mcu -l en -o mcu_en.wav

# ⚡ Small tier fast synthesis
arbotts "Proses selesai dalam waktu sekejap." --tier small -o small_out.wav

# 🎙️ Zero-Shot Voice Cloning via CLI
arbotts "Halo teman-teman, ini rekaman suara saya sendiri." -ref path/to/suara.wav -o clone_out.wav

# 🇬🇧 English synthesis
arbotts "Good morning, welcome to ArboTTS!" -o welcome_en.wav -l en

# Adjust tempo and speaker persona (e.g. bimo for baritone voice)
arbotts "Harga beras medium di pasar tradisional stabil hari ini." -sp bimo -s 0.85 -o news.wav
```

### 🌐 Web UI Studio & OpenAI-Compatible REST API
Launch the interactive browser studio with a single command:

```bash
arbotts serve --port 8000
```

Open `http://localhost:8000` to synthesize audio, toggle model tiers (**Standard 60MB** vs **Small 31MB**), choose speaker personas (**Arbo**, **Bimo**, **Dimas**, **Danu**, **Arba**, **Sita**, **Maya**), adjust tempo, and audition speech directly in your browser.

#### OpenAI TTS Client Compatibility
ArboTTS includes a drop-in `/v1/audio/speech` endpoint compatible with the OpenAI Text-to-Speech API:

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Halo, selamat pagi, Indonesia!",
    "voice": "arbo",
    "tier": "small",
    "speed": 0.85
  }' \
  --output speech.wav
```

#### Zero-Shot Voice Cloning API
Clone a voice on-the-fly by uploading a reference sample:

```bash
curl -X POST http://localhost:8000/v1/audio/clone \
  -F "file=@recordings/user_ref.wav" \
  -F "text=Halo semuanya, ini suara saya hasil kloning dengan ArboTTS!" \
  -F "tier=standard" \
  --output cloned_speech.wav
```

---

## 💡 Speech Cadence & Articulation Guidelines

1. **Speech Rate (`speed`)**:
   - `speed = 0.85` *(Default)*: Ideal broadcast announcer tempo (firm plosive articulation, crisp consonants like [g], [b], [d]).
   - `speed = 1.00`: Casual fast conversational tempo (~185 WPM).
   - `speed = 0.75`: Relaxed narration / audiobook cadence.
2. **Natural Phrasing Punctuation**:
   - Include commas `,` at natural greeting boundaries and breath pauses (e.g., *"Halo, selamat pagi, Indonesia."* vs *"Halo selamat pagi Indonesia"*). Commas provide natural 250–350 ms acoustic pauses, yielding smooth intonation and distinct syllable transitions.

---

## 📁 Repository Structure

```text
arboTTS/
├── arbotts/                # Core Python package
│   ├── downloader.py       # GitHub Releases automatic weight downloader
│   ├── synthesizer.py      # High-definition studio & standalone ONNX engine
│   ├── morpher.py          # Acoustic DSP persona transformation & recipes
│   ├── server.py           # Web UI Studio & OpenAI-compatible REST API
│   ├── cli.py              # CLI command runner ('arbotts')
│   ├── vc/                 # ToneColorConverter voice conversion engine
│   └── frontend/           # Bilingual phoneme mapping & normalizer
├── configs/
│   └── speakers.json       # Speaker profiles & calibrated DSP recipes
├── examples/               # Example scripts & showcase
│   ├── audio/              # Pre-generated sample WAV files for demo
│   └── quickstart.py       # End-to-end bilingual synthesis demo
├── tools/                  # Utility & voice cloning tools
│   ├── infer_onnx.py       # Direct ONNX runner
│   ├── eval_voice_conversion.py # CPU voice conversion evaluator
│   ├── export_vc_onnx.py   # Voice converter ONNX exporter
│   └── quantize_vc.py      # INT8 dynamic quantization script
├── training/               # Fine-tuning & adaptation scripts
│   ├── train_finetune_vits.py # VITS transfer learning with anti-NaN clamps
│   └── benchmark_vits_ft.py   # Acoustic benchmarking suite
├── arbotts_v1.onnx.json    # Phoneme table and audio configuration
├── pyproject.toml          # Packaging metadata
└── README.md
```

---

## 🗺️ Roadmap

- [x] v1.0: Bilingual VITS Base Fine-Tuning + Standalone ONNX CPU Export (60 MB).
- [x] v1.0: Automatic Model Downloader & CLI Runner.
- [x] v1.0: Pre-generated Audio Showcase for instant evaluation.
- [x] v1.1: Integrated lexical e-pepet / e-taling dictionary (97.3% pronunciation accuracy).
- [x] v1.2: Standalone Local Web UI Studio & REST API (FastAPI) runner.
- [x] v2.0: High-Definition Multi-Speaker Studio Architecture (7 personas: Arbo, Bimo, Dimas, Danu, Arba, Sita, Maya + Web UI Studio).
- [x] v2.1: Zero-shot Voice Cloning (ToneColorConverter) with Multi-Tier Base Synthesis, Web UI Studio Audio Upload, and CPU FP16/INT8 Optimization.

---

## 📄 License & Attribution

This project is licensed under the [MIT License](LICENSE).  
Developed and maintained by [ARBO-TEAM](https://github.com/ARBO-TEAM). Contributions, issues, and pull requests are warmly welcomed!
