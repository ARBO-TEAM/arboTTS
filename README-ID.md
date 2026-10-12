# ArboTTS (v1.0) 🎙️🇮🇩🇬🇧

[English](README.md) | **Bahasa Indonesia**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Model Size: 60 MB](https://img.shields.io/badge/Model%20Size-60.57%20MB-orange.svg)](https://github.com/ARBO-TEAM/arboTTS/releases)
[![Inference: CPU](https://img.shields.io/badge/Inference-Pure%20CPU%20(RTF%200.025x)-purple.svg)](#metrik-akustik--performa)

**ArboTTS** adalah mesin Text-to-Speech (TTS) bilingual (**Bahasa Indonesia & Bahasa Inggris**) ultra-ringan berbasis arsitektur **VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech)** yang dioptimasi khusus untuk eksekusi CPU standalone berformat **ONNX Runtime**.

Didesain untuk kebutuhan produksi yang efisien, hemat daya, dan dapat berjalan mulus di laptop standar, mini PC, maupun server tanpa GPU.

---

## 🎧 Sampel Suara (Dengarkan Langsung)

Anda dapat langsung mendengarkan contoh audio hasil sintesis di bawah ini:

| Bahasa | Konteks | Teks Kalimat | Pemutar Audio & Tautan |
| :--- | :--- | :--- | :--- |
| 🇮🇩 **ID** | **Sapaan Pagi (Penyiar Berita)** | *"Halo, selamat pagi, Indonesia. Awali hari Senin-mu dengan penuh ceria!"* | <audio controls src="https://raw.githubusercontent.com/ARBO-TEAM/arboTTS/main/examples/audio/id_halo_pagi.wav"></audio><br>[🔊 Putar WAV](examples/audio/id_halo_pagi.wav) |
| 🇮🇩 **ID** | **Berita Formal** | *"Harga beras medium di pasar tradisional mencapai empat belas ribu rupiah per kilogram hari ini."* | <audio controls src="https://raw.githubusercontent.com/ARBO-TEAM/arboTTS/main/examples/audio/id_berita_pasar.wav"></audio><br>[🔊 Putar WAV](examples/audio/id_berita_pasar.wav) |
| 🇬🇧 **EN** | **Sapaan Bahasa Inggris** | *"Good morning, and welcome to ArboTTS. Have a wonderful and productive day!"* | <audio controls src="https://raw.githubusercontent.com/ARBO-TEAM/arboTTS/main/examples/audio/en_good_morning.wav"></audio><br>[🔊 Putar WAV](examples/audio/en_good_morning.wav) |
| 🇬🇧 **EN** | **Penjelasan Teknis** | *"ArboTTS is an ultra lightweight text to speech model running entirely on CPU."* | <audio controls src="https://raw.githubusercontent.com/ARBO-TEAM/arboTTS/main/examples/audio/en_lightweight.wav"></audio><br>[🔊 Putar WAV](examples/audio/en_lightweight.wav) |

---

## ✨ Fitur Utama

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

# Inisialisasi synthesizer (otomatis download model jika belum ada)
tts = ArboTTS()

# 🇮🇩 Sintesis Bahasa Indonesia
tts.speak(
    text="Halo, selamat pagi, Indonesia. Awali hari Senin-mu dengan penuh ceria!",
    output_file="halo_id.wav",
    lang="id",
    speed=0.85,
)

# 🇬🇧 Sintesis Bahasa Inggris
tts.speak(
    text="Good morning, and welcome to ArboTTS. Have a wonderful and productive day!",
    output_file="welcome_en.wav",
    lang="en",
    speed=0.85,
)
```

### Menggunakan Command Line (CLI)
Setelah terinstal, perintah `arbotts` dapat langsung dipanggil dari terminal / CMD:

```bash
# 🇮🇩 Sintesis Bahasa Indonesia (default)
arbotts "Halo, selamat datang di ArboTTS!" -o halo_id.wav -l id

# 🇬🇧 Sintesis Bahasa Inggris
arbotts "Good morning, welcome to ArboTTS!" -o welcome_en.wav -l en

# Menyetel tempo bicara (misal 0.85 untuk gaya penyiar rileks)
arbotts "Harga beras medium di pasar tradisional stabil hari ini." -o berita.wav -s 0.85
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
│   ├── synthesizer.py      # Engine inferensi ONNX Runtime CPU
│   ├── cli.py              # CLI runner (perintah 'arbotts')
│   └── frontend/           # Pemrosesan teks dan pemetaan fonem bilingual
├── examples/               # Skrip contoh penggunaan
│   ├── audio/              # Sampel WAV hasil generate untuk showcase
│   └── quickstart.py       # Contoh sintesis end-to-end bilingual
├── tools/                  # Script pendukung
│   ├── infer_onnx.py       # Standalone ONNX inference runner
│   └── export_arbotts_onnx.py # Script ekspor PyTorch ke ONNX Opset 15
├── training/               # Skrip pelatihan & transfer learning
│   ├── train_finetune_vits.py # Script fine-tuning VITS dengan anti-NaN clamp
│   └── benchmark_vits_ft.py   # Suite evaluasi metrik akustik broadcast
├── arbotts_v1.onnx.json    # Konfigurasi fonem dan parameter inferensi
├── pyproject.toml          # Konfigurasi build package
└── README.md
```

---

## 🗺️ Roadmap Pengembangan

- [x] v1.0: Model VITS Base Fine-Tuning + Standalone ONNX CPU Export (60 MB).
- [x] v1.0: Auto-Downloader dari GitHub Releases & CLI interface.
- [x] v1.0: Showcase Pemutar Audio langsung di README.
- [ ] v1.1: Integrasi kamus leksikal e-pepet / e-taling otomatis (akurasi 97.3%).
- [ ] v1.2: Antarmuka Web UI lokal & REST API (FastAPI) siap pakai.
- [ ] v2.0: Dukungan Multi-Speaker (Pilihan suara pria & wanita dalam 1 file model terpadu).

---

## 📄 Lisensi & Kontribusi

Proyek ini dirilis di bawah lisensi [MIT License](LICENSE).  
Dikelola dan dikembangkan oleh [ARBO-TEAM](https://github.com/ARBO-TEAM). Kontribusi, perbaikan issue, dan pull request sangat disambut!
