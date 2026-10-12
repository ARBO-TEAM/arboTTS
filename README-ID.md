# ArboTTS (v2.1) 🎙️🇮🇩🇬🇧

[English](README.md) | **Bahasa Indonesia**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Live Demo: GitHub Pages](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages%20(WASM)-38bdf8.svg)](https://arbo-team.github.io/arbotts-demo/)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Model Size: 60 MB](https://img.shields.io/badge/Model%20Size-60.57%20MB-orange.svg)](https://github.com/ARBO-TEAM/arboTTS/releases)
[![Inference: CPU](https://img.shields.io/badge/Inference-Pure%20CPU%20(RTF%200.025x)-purple.svg)](#metrik-akustik--performa)

> 🌐 **Coba Langsung (100% Client-Side WebAssembly)**: Dengarkan sintesis suara Bahasa Indonesia instan langsung di browser Anda tanpa server di **[arbo-team.github.io/arbotts-demo](https://arbo-team.github.io/arbotts-demo/)** (Source: [arbotts-demo](https://github.com/ARBO-TEAM/arbotts-demo)). Tanpa backend, tanpa API key, dan privasi penuh!

**ArboTTS** adalah mesin Text-to-Speech (TTS) bilingual (**Bahasa Indonesia & Bahasa Inggris**) ultra-ringan berbasis arsitektur **VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech)** yang dioptimasi khusus untuk eksekusi CPU standalone berformat **ONNX Runtime**.

Pada versi 2.1, ArboTTS menghadirkan fitur **Zero-Shot Voice Cloning di CPU** dengan ToneColorConverter, **Sintesis Studio Multi-Persona Definisi Tinggi** (7 karakter suara: Arbo, Bimo, Dimas, Danu, Arba, Sita, dan Maya), antarmuka interaktif **Web UI Studio** di browser, serta endpoint **REST API kompatibel OpenAI** dengan dukungan multi-tier engine.

Didesain untuk kebutuhan produksi yang efisien, hemat daya, dan dapat berjalan mulus di laptop standar, mini PC, maupun server tanpa GPU.

---

## 🎭 Karakter Profil Suara Studio

ArboTTS v2.0 menyediakan 7 persona suara yang dikalibrasi ke dalam dua kelompok: **Persona Utama Pria** (sangat direkomendasikan untuk resonansi akustik dan ritme paling alami) serta **Persona Tambahan Wanita**:

### ⭐ Persona Utama Pria (Direkomendasikan)
| Persona | Gender | Karakter & Intonasi | Peruntukan Terbaik | Opsi CLI |
| :--- | :--- | :--- | :--- | :--- |
| **Arbo** | 👨 Pria | **Penyiar Berita Formal**<br>Berwibawa, konsonan letup tegas, resonansi dalam | Berita formal, pengumuman resmi, pidato | `-sp arbo` *(default)* |
| **Bimo** | 👨 Pria | **Baritone Berat & Berwibawa**<br>Resonansi dada tebal, berbobot, wibawa matang | Dokumenter, profil perusahaan, voiceover trailer | `-sp bimo` |
| **Dimas** | 👨 Pria | **Pria Muda Kasual & Energik**<br>Tempo dinamis, ramah, artikulasi konsonan letup jernih | Asisten pintar, podcast, video explainer modern | `-sp dimas` |
| **Danu** | 👨 Pria | **Narator Tenang & Santun**<br>Tekstur hangat, tempo lembut, intonasi menenangkan | Audiobook, meditasi, dongeng, pemandu museum | `-sp danu` |

### 👩 Persona Tambahan Wanita
| Persona | Gender | Karakter & Intonasi | Peruntukan Terbaik | Opsi CLI |
| :--- | :--- | :--- | :--- | :--- |
| **Arba** | 👩 Wanita | **Presenter Berita Profesional**<br>Artikulasi jernih, santun, lugas dengan kehangatan mid-range | Siaran berita, video korporat, laporan jurnalis | `-sp arba` |
| **Sita** | 👩 Wanita | **Asisten Interaktif Cerdas**<br>Ceria, bersahabat, tempo percakapan interaktif | Asisten virtual, CS bot, panduan aplikasi | `-sp sita` |
| **Maya** | 👩 Wanita | **Narator Tenang & Storyteller**<br>Lembut, santai, tekstur hangat dengan ritme halus | Audiobook, mediasi, dongeng, dokumenter | `-sp maya` |

---

## ⚡ Tingkatan Arsitektur Model (Model Tiers)

ArboTTS menyediakan tingkatan model untuk menyeimbangkan ketepatan akustik dengan ukuran memori dan latensi komputasi:

| Tier | Berkas Model | Ukuran (ONNX) | Parameter | Konfigurasi Arsitektur | Rata-rata Latensi CPU / RTF | Target Penggunaan |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 🚀 **Standard** *(Default)* | `arbotts_v1.onnx` | **60.57 MB** | **23.66 M** | Hidden 192, Filter 768, 6 Layers, ResBlock1 | ~100–140 ms (0.025x RTF) | Produksi studio, desktop, server |
| ⚡ **Small** | `arbotts_small.onnx` | **31.29 MB** | **8.39 M** | Hidden 128, Filter 512, 4 Layers, ResBlock2 | ~45–65 ms (0.012x RTF) | Chatbot interaktif, aplikasi edge, server hemat RAM |
| 🪶 **Tiny** *(Roadmap)* | `arbotts_tiny.onnx` | *~12.8 MB* | *~3.65 M* | Hidden 96, Filter 256, 3 Layers, ResBlock2 | ~20–30 ms (0.006x RTF) | Raspberry Pi, IoT, perangkat wearable & mobile |

---

## 🎧 Sampel Suara (Dengarkan Langsung)

Anda dapat langsung mendengarkan contoh audio hasil sintesis di bawah ini:

| Bahasa | Konteks | Teks Kalimat | Pemutar Audio & Tautan |
| :--- | :--- | :--- | :--- |
| 🇮🇩 **ID** | **Sapaan Pagi (Penyiar Berita)** | *"Halo, selamat pagi, Indonesia. Awali hari Seninmu dengan penuh ceria!"* | <audio controls src="https://raw.githubusercontent.com/ARBO-TEAM/arboTTS/main/examples/audio/id_halo_pagi.wav"></audio><br>[🔊 Putar WAV](examples/audio/id_halo_pagi.wav) |
| 🇮🇩 **ID** | **Berita Formal** | *"Harga beras medium di pasar tradisional mencapai empat belas ribu rupiah per kilogram hari ini."* | <audio controls src="https://raw.githubusercontent.com/ARBO-TEAM/arboTTS/main/examples/audio/id_berita_pasar.wav"></audio><br>[🔊 Putar WAV](examples/audio/id_berita_pasar.wav) |
| 🇬🇧 **EN** | **Sapaan Bahasa Inggris** | *"Good morning, and welcome to ArboTTS. Have a wonderful and productive day!"* | <audio controls src="https://raw.githubusercontent.com/ARBO-TEAM/arboTTS/main/examples/audio/en_good_morning.wav"></audio><br>[🔊 Putar WAV](examples/audio/en_good_morning.wav) |
| 🇬🇧 **EN** | **Penjelasan Teknis** | *"ArboTTS is an ultra lightweight text to speech model running entirely on CPU."* | <audio controls src="https://raw.githubusercontent.com/ARBO-TEAM/arboTTS/main/examples/audio/en_lightweight.wav"></audio><br>[🔊 Putar WAV](examples/audio/en_lightweight.wav) |

---

## ✨ Fitur Utama

- 🎭 **Dukungan Multi-Persona Studio (v2.0)**: Beralih instan antar 7 karakter suara terkalibrasi (**Arbo**, **Bimo**, **Dimas**, **Danu**, **Arba**, **Sita**, **Maya**) untuk berita formal, baritone berbobot, kasual, asisten interaktif, dan narasi santai.
- 🧬 **Zero-Shot Voice Cloning di CPU (v2.1)**: Mentransfer warna suara pengguna secara instan dari sampel audio referensi singkat (< 10 detik) dengan ToneColorConverter, mendukung base engine Standard (60 MB) maupun Small (31 MB) secara real-time.
- 🌐 **Web UI Studio & REST API Interaktif**: Studio berbasis web lokal di `http://localhost:8000` dengan tab khusus Voice Cloning serta endpoint streaming `/v1/audio/speech` kompatibel standar OpenAI.
- ⚡ **Ultra-Cepat di CPU (RTF 0.025x)**: Menghasilkan 5 detik audio dalam waktu **~120-140 milidetik** di CPU laptop standar (>35x lebih cepat dari waktu nyata).
- 🪶 **Model Ultra-Kompak (~60 MB)**: Total hanya **23.66 juta parameter** (ukuran file ONNX **60.57 MB**). Hemat penyimpanan dan memori RAM (< 150 MB).
- 🇮🇩🇬🇧 **Dukungan Bilingual (ID + EN)**: Artikulasi fasih bahasa Indonesia berkarakter penyiar serta sintesis bahasa Inggris yang jernih.
- 🎙️ **Karakter Suara Penyiar Berwibawa**: Mengadopsi timbre penyiar berita pria (*broadcast announcer*) dengan intonasi dan ritme suku kata yang tegas dan alami.
- 📦 **Zero-Config Auto Downloader**: Mengunduh bobot model ONNX secara otomatis dari GitHub Releases pada pemanggilan pertama ke folder cache lokal (`~/.cache/arbotts/`).
- 🛡️ **Mandiri Tanpa PyTorch saat Inferensi**: Mode produksi hanya memerlukan pustaka ringan (`onnxruntime`, `numpy`, `soundfile`).
- 🔒 **Aman Lisensi & TOS**: Bobot suara dihasilkan melalui adaptasi morfologis akustik (*acoustic morphing* -2.0 semitone + *transformative vocoder distillation*), bebas dari risiko kloning suara mentah pihak ketiga.

---

## 📊 Spesifikasi Model & Parameter

| Komponen Arsitektur | Jumlah Parameter | Persentase | Fungsi |
| :--- | :--- | :--- | :--- |
| **Normalizing Flow (`flow`)** | 8,076,144 (8.08 M) | 34.1% | Transformasi distribusi representasi laten fonem ke spektrogram mel |
| **Text Encoder (`enc_p`)** | 6,341,760 (6.34 M) | 26.8% | Pemrosesan teks dan representasi fonem multibahasa |
| **HiFi-GAN Vocoder (`dec`)** | 1,664,768 (1.66 M) | 7.0% | Sintesis gelombang suara (*waveform*) beresolusi tinggi (ResBlock2) |
| **Duration Predictor (`dp`)** | 1,317,168 (1.32 M) | 5.6% | Prediksi durasi dan ritme pengucapan suku kata |
| **Total Parameter** | **23,663,792 (23.66 M)** | **100%** | **Arsitektur End-to-End VITS Standalone** |

### Metrik Akustik & Performa
- **Format Model**: Standalone ONNX (Opset 15)
- **Ukuran File**: `63,516,051 bytes` (~60.57 MB)
- **Frekuensi Sampling**: `22,050 Hz` (16-bit PCM Mono)
- **Konsumsi RAM**: `< 150 MB`
- **Latensi Rata-Rata CPU**: `~100–140 ms` per kalimat (RTF 0.025x)

---

## 🚀 Cara Instalasi

### 1. Dari Source
```bash
git clone https://github.com/ARBO-TEAM/arboTTS.git
cd arboTTS
pip install -e .
```

### 2. Kebutuhan Sistem
ArboTTS menggunakan pustaka fonemizer `espeak-ng`. Pastikan `espeakng-loader` atau `espeak-ng` terpasang di sistem:
```bash
pip install espeakng-loader phonemizer
```

---

## 💻 Cara Penggunaan

### Menggunakan Python API
Sintesis suara alami dalam Bahasa Indonesia atau Bahasa Inggris hanya dengan beberapa baris kode:

```python
from arbotts import ArboTTS

# Inisialisasi synthesizer (Tingkat Standard default, atau opsi tier="small")
tts_standard = ArboTTS(tier="standard")  # 60.57 MB - Full Fidelity Studio
tts_small = ArboTTS(tier="small")        # 31.29 MB - Distilled Sangat Cepat

# 🇮🇩 Sintesis Bahasa Indonesia dengan persona utama
tts_standard.speak(
    text="Halo, selamat pagi, Indonesia. Awali hari Seninmu dengan penuh ceria!",
    output_file="halo_id.wav",
    speaker="arbo",
    lang="id",
    speed=0.85,
)

# ⚡ Sintesis cepat dengan Small tier
tts_small.speak(
    text="Sistem siap melayani permintaan pengguna dengan latensi ultra rendah.",
    output_file="jawaban_cepat.wav",
    speaker="arbo",
    lang="id",
    speed=0.85,
)

# 🇬🇧 Sintesis Bahasa Inggris
tts_standard.speak(
    text="Good morning, and welcome to ArboTTS. Have a wonderful and productive day!",
    output_file="welcome_en.wav",
    speaker="arbo",
    lang="en",
    speed=0.85,
)
```

### Menggunakan Command Line (CLI)
Setelah terinstal, perintah `arbotts` dapat langsung dipanggil dari terminal / CMD:

```bash
# 🇮🇩 Sintesis Bahasa Indonesia dengan Standard tier (default)
arbotts "Halo, selamat datang di ArboTTS!" -o halo_id.wav -l id

# ⚡ Sintesis cepat dengan Small tier
arbotts "Proses selesai dalam waktu sekejap." --tier small -o small_out.wav

# 🇬🇧 Sintesis Bahasa Inggris
arbotts "Good morning, welcome to ArboTTS!" -o welcome_en.wav -l en

# Menyetel tempo bicara dan speaker persona (misal bimo untuk suara baritone)
arbotts "Harga beras medium di pasar tradisional stabil hari ini." -sp bimo -s 0.85 -o berita.wav
```

### 🌐 Web UI Studio & REST API Kompatibel OpenAI
Jalankan studio antarmuka web lokal hanya dengan satu perintah:

```bash
arbotts serve --port 8000
```

Buka `http://localhost:8000` di browser untuk mengetik kalimat, beralih tier model (**Standard 60MB** vs **Small 31MB**), memilih profil suara (**Arbo**, **Bimo**, **Dimas**, **Danu**, **Arba**, **Sita**, **Maya**), menyetel tempo, dan mendengarkan hasil audio langsung.

#### Kompatibilitas Endpoint OpenAI TTS
ArboTTS menyediakan endpoint standar `/v1/audio/speech` yang kompatibel dengan protokol OpenAI Text-to-Speech:

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Halo, selamat pagi, Indonesia!",
    "voice": "arbo",
    "tier": "small",
    "speed": 0.85
  }' \
  --output pidato.wav
```

#### Endpoint Kloning Suara (Zero-Shot Voice Cloning)
Kloning karakter suara pengguna secara instan dengan mengunggah sampel referensi suara:

```bash
curl -X POST http://localhost:8000/v1/audio/clone \
  -F "file=@recordings/user_ref.wav" \
  -F "text=Halo semuanya, ini suara saya hasil kloning dengan ArboTTS!" \
  -F "tier=standard" \
  --output suara_kloning.wav
```

---

## 💡 Tips Artikulasi & Tempo Bicara

1. **Pengaturan Tempo Bicara (`speed`)**:
   - `speed = 0.85` *(Default)*: Tempo ideal pembawa berita (artikulasi tegas, konsonan letup seperti [g], [b], [d] terdengar sangat jelas).
   - `speed = 1.00`: Tempo percakapan kasual yang cepat (~185 kata per menit).
   - `speed = 0.75`: Tempo narasi tenang / audiobook.
2. **Penggunaan Tanda Baca Alami**:
   - Tambahkan koma `,` pada frasa sapaan atau klausa jeda napas (contoh: *"Halo, selamat pagi, Indonesia."* vs *"Halo selamat pagi Indonesia"*). Koma memberikan jeda akustik alami 250–350 ms yang membuat intonasi terdengar luwes.

---

## 📁 Struktur Repositori

```text
arboTTS/
├── arbotts/                # Pustaka utama Python
│   ├── downloader.py       # Auto-downloader bobot model dari GitHub Releases
│   ├── synthesizer.py      # Engine studio definisi tinggi & ONNX Runtime CPU
│   ├── morpher.py          # Resep transformasi akustik DSP per persona
│   ├── server.py           # Web UI Studio & REST API kompatibel OpenAI
│   ├── cli.py              # CLI runner (perintah 'arbotts')
│   ├── vc/                 # Engine konversi suara ToneColorConverter
│   └── frontend/           # Pemrosesan teks dan pemetaan fonem bilingual
├── configs/
│   └── speakers.json       # Profil speaker & resep DSP terkalibrasi
├── examples/               # Skrip contoh penggunaan
│   ├── audio/              # Sampel WAV hasil generate untuk showcase
│   └── quickstart.py       # Contoh sintesis end-to-end bilingual
├── tools/                  # Script pendukung & kloning suara
│   ├── infer_onnx.py       # Standalone ONNX inference runner
│   ├── eval_voice_conversion.py # Evaluasi konversi suara CPU
│   ├── export_vc_onnx.py   # Eksportir voice converter ke ONNX
│   └── quantize_vc.py      # Skrip kuantisasi dinamis INT8
├── training/               # Skrip pelatihan & transfer learning
│   ├── train_finetune_vits.py # Script fine-tuning VITS dengan anti-NaN clamp
│   └── benchmark_vits_ft.py   # Suite evaluasi metrik akustik broadcast
├── arbotts_v1.onnx.json    # Konfigurasi fonem dan parameter inferensi
├── pyproject.toml          # Konfigurasi build package
├── Makefile                # Shortcut perintah umum pengembang
└── README.md
```

---

## 🗺️ Roadmap Pengembangan

- [x] v1.0: Model VITS Base Fine-Tuning + Standalone ONNX CPU Export (60 MB).
- [x] v1.0: Auto-Downloader dari GitHub Releases & CLI interface.
- [x] v1.0: Showcase Pemutar Audio langsung di README.
- [x] v1.1: Integrasi kamus leksikal e-pepet / e-taling otomatis (akurasi 97.3%).
- [x] v1.2: Antarmuka Web UI Studio lokal & REST API (FastAPI) siap pakai.
- [x] v2.0: Arsitektur Multi-Speaker Studio Resolusi Tinggi (7 Karakter: Arbo, Bimo, Dimas, Danu, Arba, Sita, Maya + Web UI Studio).
- [x] v2.1: Kloning Suara Zero-shot (ToneColorConverter) dengan Base Engine Multi-Tier, Unggah Audio di Web UI Studio, dan Optimasi FP16/INT8 CPU.

---

## 📄 Lisensi & Kontribusi

Proyek ini dirilis di bawah lisensi [MIT License](LICENSE).  
Dikelola dan dikembangkan oleh [ARBO-TEAM](https://github.com/ARBO-TEAM). Kontribusi, perbaikan issue, dan pull request sangat disambut!
