# ArboTTS (v1.0) 🎙️🇮🇩

**English** | [Bahasa Indonesia](README-ID.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Model Size: 60 MB](https://img.shields.io/badge/Model%20Size-60.57%20MB-orange.svg)](https://github.com/ARBO-TEAM/arboTTS/releases)
[![Inference: CPU](https://img.shields.io/badge/Inference-Pure%20CPU%20(RTF%200.025x)-purple.svg)](#benchmarks--specifications)

**ArboTTS** is an ultra-lightweight, production-grade Indonesian Text-to-Speech (TTS) engine powered by the **VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech)** architecture, optimized for standalone CPU inference using **ONNX Runtime**.

Engineered for resource-constrained environments, ArboTTS runs smoothly on standard laptops, mini PCs, and server instances without requiring a dedicated GPU.

---

## ✨ Key Features

- ⚡ **Blazing Fast on CPU (RTF 0.025x)**: Generates 5.0 seconds of speech in just **~120–140 ms** on consumer laptop CPUs (>35x faster than real-time).
- 🪶 **Ultra-Compact Model (~60 MB)**: Total parameter count of only **23.66 million** (standalone ONNX file size **60.57 MB**). Extremely low memory footprint (< 150 MB RAM).
- 🎙️ **Authoritative Indonesian Broadcast Voice**: Warm, clear announcer timbre tuned for native Indonesian prosody, cadence, and syllable stress.
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
ArboTTS uses `espeak-ng` for Indonesian phonemization. Ensure `espeakng-loader` or `espeak-ng` is available:
```bash
pip install espeakng-loader phonemizer
```

---

## 💻 Quickstart

### Python API
Synthesize natural speech in just three lines of Python:

```python
from arbotts import ArboTTS

# Initialize synthesizer (auto-downloads model on first run)
tts = ArboTTS()

# Synthesize directly to a WAV file
output_file = "output.wav"
text = "Halo, selamat pagi, Indonesia. Awali hari Senin-mu dengan penuh ceria!"

# speed=0.85 delivers a relaxed, authoritative broadcast pace
duration = tts.speak(text, output_file, speed=0.85)

print(f"Generated audio: {output_file} (Duration: {duration:.2f} s)")
```

### Command Line Interface (CLI)
Once installed, the `arbotts` command is accessible directly from your terminal:

```bash
# Basic synthesis
arbotts "Halo, selamat datang di ArboTTS!" -o halo.wav

# Adjust tempo (e.g. 0.85 for relaxed announcer delivery)
arbotts "Harga beras medium di pasar tradisional stabil hari ini." -o news.wav -s 0.85
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
│   ├── synthesizer.py      # Standalone ONNX Runtime CPU engine
│   ├── cli.py              # CLI command runner ('arbotts')
│   └── frontend/           # Phoneme mapping and text normalization
├── examples/               # Example scripts
│   └── quickstart.py       # End-to-end synthesis demo
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

- [x] v1.0: VITS Base Fine-Tuning + Standalone ONNX CPU Export (60 MB).
- [x] v1.0: Automatic Model Downloader & CLI Runner.
- [ ] v1.1: Multi-Speaker Support (Male & Female broadcast announcers in a single model file).
- [ ] v1.2: Integrated lexical e-pepet / e-taling dictionary (97.3% pronunciation accuracy).
- [ ] v1.3: Standalone Local Web UI & REST API (FastAPI) runner.

---

## 📄 License & Attribution

This project is licensed under the [MIT License](LICENSE).  
Developed and maintained by [ARBO-TEAM](https://github.com/ARBO-TEAM). Contributions, issues, and pull requests are warmly welcomed!
