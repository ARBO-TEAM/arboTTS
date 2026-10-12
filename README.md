# ArboTTS (v2.0) 🎙️🇮🇩🇬🇧

**English** | [Bahasa Indonesia](README-ID.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Model Size: 60 MB](https://img.shields.io/badge/Model%20Size-60.57%20MB-orange.svg)](https://github.com/ARBO-TEAM/arboTTS/releases)
[![Inference: CPU](https://img.shields.io/badge/Inference-Pure%20CPU%20(RTF%200.025x)-purple.svg)](#-model-specifications--parameter-breakdown)

**ArboTTS** is an ultra-lightweight, production-grade bilingual (**Indonesian & English**) Text-to-Speech (TTS) engine powered by the **VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech)** architecture, optimized for standalone CPU inference using **ONNX Runtime**.

With version 2.0, ArboTTS brings **High-Definition Multi-Persona Studio Synthesis** (featuring 7 distinct voices: Arbo, Bimo, Dimas, Danu, Arba, Sita, and Maya), an interactive browser-based **Web UI Studio**, and an **OpenAI-compatible REST API**.

Engineered for resource-constrained environments, ArboTTS runs smoothly on standard laptops, mini PCs, edge devices, and server instances without requiring a GPU.

---

## 🎭 Studio Voice Personas

ArboTTS v2.0 includes 7 professionally calibrated voice personas across male and female vocal ranges:

| Persona | Gender | Voice Role & Tone | Best Used For | CLI Option |
| :--- | :--- | :--- | :--- | :--- |
| **Arbo** | 👨 Male | **Formal Broadcast Announcer**<br>Authoritative, crisp plosive consonants, deep resonance | News anchors, official announcements, formal keynotes | `-sp arbo` *(default)* |
| **Bimo** | 👨 Male | **Deep Baritone & Authority**<br>Rich chest resonance, heavy weight, mature timbre | Documentaries, corporate voiceovers, trailers | `-sp bimo` |
| **Dimas** | 👨 Male | **Young & Energetic Casual**<br>Friendly conversational tempo, crisp consonant articulation | Virtual assistants, podcasts, modern explainers | `-sp dimas` |
| **Danu** | 👨 Male | **Calm Storyteller & Narrator**<br>Velvety warmth, gentle pacing, polite cadence | Audiobooks, meditation, bedtime stories, documentaries | `-sp danu` |
| **Arba** | 👩 Female | **Professional News Presenter**<br>Articulate, clear, polite, warm mid-range articulation | News broadcasts, corporate explainers, reporting | `-sp arba` |
| **Sita** | 👩 Female | **Smart Interactive Assistant**<br>Bright, cheerful, engaging conversational tempo | Virtual assistants, customer care, interactive bots | `-sp sita` |
| **Maya** | 👩 Female | **Calm Storyteller & Narration**<br>Gentle, relaxed, velvety warmth with soft cadence | Audiobooks, meditation, bedtime stories, documentaries | `-sp maya` |

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
- 🌐 **Interactive Web UI Studio & REST API**: Full browser studio at `http://localhost:8000` and drop-in OpenAI-compatible `/v1/audio/speech` streaming endpoint.
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

# Initialize synthesizer (auto-downloads model on first run)
tts = ArboTTS()

# 🇮🇩 Indonesian synthesis
tts.speak(
    text="Halo, selamat pagi, Indonesia. Awali hari Seninmu dengan penuh ceria!",
    output_file="halo_id.wav",
    lang="id",
    speed=0.85,
)

# 🇬🇧 English synthesis
tts.speak(
    text="Good morning, and welcome to ArboTTS. Have a wonderful and productive day!",
    output_file="welcome_en.wav",
    lang="en",
    speed=0.85,
)
```

### Command Line Interface (CLI)
Once installed, the `arbotts` command is accessible directly from your terminal:

```bash
# 🇮🇩 Indonesian synthesis (default)
arbotts "Halo, selamat datang di ArboTTS!" -o halo_id.wav -l id

# 🇬🇧 English synthesis
arbotts "Good morning, welcome to ArboTTS!" -o welcome_en.wav -l en

# Adjust tempo and speaker (e.g. arba for female presenter)
arbotts "Harga beras medium di pasar tradisional stabil hari ini." -sp arba -s 0.85 -o news.wav
```

### 🌐 Web UI Studio & OpenAI-Compatible REST API
Launch the interactive browser studio with a single command:

```bash
arbotts serve --port 8000
```

Open `http://localhost:8000` to synthesize audio, choose speaker personas (**Arbo**, **Bimo**, **Dimas**, **Danu**, **Arba**, **Sita**, **Maya**), adjust tempo, and audition speech directly in your browser.

#### OpenAI TTS Client Compatibility
ArboTTS includes a drop-in `/v1/audio/speech` endpoint compatible with the OpenAI Text-to-Speech API:

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Halo, selamat pagi, Indonesia!",
    "voice": "arba",
    "speed": 0.85
  }' \
  --output speech.wav
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
│   └── frontend/           # Bilingual phoneme mapping & normalizer
├── configs/
│   └── speakers.json       # Speaker profiles & calibrated DSP recipes
├── examples/               # Example scripts & showcase
│   ├── audio/              # Pre-generated sample WAV files for demo
│   └── quickstart.py       # End-to-end bilingual synthesis demo
├── tools/                  # Utility scripts
│   ├── infer_onnx.py       # Direct ONNX runner
│   └── export_arbotts_onnx.py # PyTorch to ONNX exporter (opset 15)
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

---

## 📄 License & Attribution

This project is licensed under the [MIT License](LICENSE).  
Developed and maintained by [ARBO-TEAM](https://github.com/ARBO-TEAM). Contributions, issues, and pull requests are warmly welcomed!
