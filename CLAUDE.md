# ArboTTS — Developer & Agent Guidelines (CLAUDE.md)

ArboTTS adalah mesin Text-to-Speech (TTS) bilingual (Indonesia & Inggris) ultra-ringan berbasis arsitektur **VITS** standalone ONNX Runtime CPU serta pipeline **Studio Multi-Persona HD** (7 Karakter: Arbo, Bimo, Dimas, Danu, Arba, Sita, Maya).

---

## 🏗️ Arsitektur & Prinsip Utama

1. **Dual-Mode Hybrid Synthesis**:
   - **Arbo (Pria - Broadcast Announcer)**: Inferensi lokal murni di CPU via ONNX Runtime (`arbotts_v1.onnx`, 22,050 Hz, RTF ~0.025x, latensi ~13–60 ms).
   - **Studio Personas (Pria: Bimo, Dimas, Danu; Wanita: Arba, Sita, Maya)**: Stream sintesis definisi tinggi neural (`id-ID-ArdiNeural` / `id-ID-GadisNeural` untuk ID, `en-US-GuyNeural` / `en-US-JennyNeural` untuk EN) dikombinasikan dengan transformasi DSP akustik presisi dari `configs/speakers.json` (dengan fallback offline otomatis ke VITS CPU).
2. **Karakter & Resep Akustik DSP** (`configs/speakers.json` & `arbotts/morpher.py`):
   - `arbo`: Bobot VITS asli, tempo broadcast tegas (`speed=0.85`).
   - `bimo`: Pitch shift `-1.5` st, lowpass warmth 5,500 Hz, resonansi dada tebal baritone.
   - `dimas`: Pitch shift `+1.0` st, tempo dinamis kasual, artikulasi konsonan letup velar [g] jernih (n_fft=2048).
   - `danu`: Pitch shift `-0.7` st, lowpass warmth 6,000 Hz, narasi santun dan lembut.
   - `arba`: Pitch shift `-0.8` st, highpass presence boost 3,500 Hz (+25%), peak normalize 0.95.
   - `sita`: Pitch shift `+1.2` st, highpass presence boost 3,500 Hz (+25%), tempo interaktif ceria.
   - `maya`: Pitch shift `-1.4` st, lowpass warmth 5,500 Hz, narasi santai dan hangat.
3. **Penyusunan Teks & Fonetik (G2P)**:
   - Kamus leksikal pepet/taling berada di `arbotts/frontend/lexicon.py`.
   - Kata berklitik disatukan tanpa tanda hubung (contoh standar: `Seninmu`, bukan `Senin-mu`).
   - Tanda baca koma `,` berfungsi sebagai jeda akustik alami 250–350 ms.

---

## 🛠️ Perintah Umum Pengembang (Developer Commands)

Gunakan `make` atau perintah Python langsung:

```bash
# Instalasi
make install            # Install core package (mode editable)
make install-web        # Install dependensi Web UI Studio & FastAPI
make install-all        # Install semua dependensi (web + training)

# Menjalankan Server & Web UI
make serve              # Jalankan Web UI Studio di http://localhost:8000
python -m arbotts.server --host 0.0.0.0 --port 8000

# Pengujian (Test Suite)
make test               # Jalankan seluruh test suite dengan pytest
pytest tests -v

# Uji Coba Sintesis Audio
make demo               # Generate audio demo Arbo (demo_arbo.wav)
make demo-all           # Generate 4 audio demo untuk semua persona ke demo_outputs/

# Docker
make docker-build       # Build image arbotts:v2.0
make docker-run         # Jalankan container di port 8000
```

---

## 🧪 Konvensi Kode & Testing

1. **Python Standards**:
   - Python `>=3.10`.
   - Gunakan type annotations lengkap (`from __future__ import annotations`).
   - Dependencies inferensi inti harus tetap minimal: `onnxruntime`, `numpy`, `soundfile`, `scipy`.
2. **Testing Gate**:
   - Sebelum commit atau push, pastikan `pytest tests -v` selalu berstatus hijau (100% pass).
   - Test mencakup fonemizer ID/EN, kamus pepet/taling, sintesis endpoint API, dan integritas server.
3. **Environment**:
   - Saat menjalankan modul secara langsung di terminal Windows / PowerShell, pastikan `PYTHONPATH=.` atau `$env:PYTHONPATH = "D:\kerjaan\belajar\tts\arboTTS"`.

---

## 🔒 Privasi, Keamanan, & Lisensi

1. **Model Weights & Third-Party TTS**:
   - Bobot model hasil sintesis neural stream **wajib** melalui tahap transformasi DSP (*acoustic morphing*) untuk mencegah liabilitas kloning suara mentah pihak ketiga.
2. **Privasi Rekaman Suara**:
   - Rekaman suara mentah pengguna bersifat privat dan **tidak boleh** diunggah ke repositori publik.
3. **Credentials & Secrets**:
   - Jangan pernah melakukan hardcode atau commit terhadap API key, private tokens, atau credentials personal ke dalam git.

---

## 📦 Rilis & Git Workflow

- **Branch utama**: `main`.
- **Format Commit**: Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`).
- Sertakan co-author tag pada commit:
  ```text
  Co-Authored-By: Claude Code <noreply@anthropic.com>
  ```
- **Rilis GitHub**: Gunakan `gh release create <tag>` dengan menyertakan aset model `arbotts_v1.onnx`, `arbotts_v1.onnx.json`, dan `speakers.json`.
