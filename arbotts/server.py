"""FastAPI REST API server and local Web UI for ArboTTS."""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
import soundfile as sf
import uvicorn

from arbotts.synthesizer import ArboTTS

REPO_ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="ArboTTS API",
    description="Ultra-lightweight bilingual (Indonesian & English) Text-to-Speech engine with Native Multi-Speaker VITS",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_ENGINES: dict[str, ArboTTS] = {}


def get_engine(tier: str = "standard") -> ArboTTS:
    global _ENGINES
    normalized_tier = tier.lower().strip()
    if normalized_tier not in _ENGINES:
        _ENGINES[normalized_tier] = ArboTTS(tier=normalized_tier)
    return _ENGINES[normalized_tier]


class OpenAISpeechRequest(BaseModel):
    model: str = "arbotts"
    input: str = Field(..., description="The text to generate audio for.")
    voice: str = Field("arbo", description="Speaker profile: arbo, bimo, dimas, danu, arba, sita, maya.")
    response_format: str = Field("wav", description="Audio format (wav).")
    speed: float = Field(0.85, ge=0.25, le=4.0, description="Speed factor (default: 0.85).")
    lang: str = Field("id", description="Language code: 'id' or 'en'.")
    tier: str = Field("standard", description="Model tier: 'standard' (60MB) or 'small' (31MB).")


WEB_UI_HTML = """<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ArboTTS Studio 🎙️</title>
  <style>
    :root {
      --bg: #0d1117;
      --card-bg: #161b22;
      --border: #30363d;
      --accent: #58a6ff;
      --accent-hover: #1f6feb;
      --text: #f0f6fc;
      --text-muted: #8b949e;
      --success: #238636;
      --chip-bg: #21262d;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
    body { background-color: var(--bg); color: var(--text); padding: 24px; min-height: 100vh; display: flex; justify-content: center; align-items: flex-start; }
    .container { width: 100%; max-width: 840px; background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 28px; box-shadow: 0 8px 24px rgba(0,0,0,0.5); }
    header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 24px; }
    .title-group h1 { font-size: 24px; font-weight: 700; display: flex; align-items: center; gap: 8px; }
    .badge { font-size: 11px; background: rgba(88, 166, 255, 0.15); color: var(--accent); padding: 3px 8px; border-radius: 20px; font-weight: 600; border: 1px solid rgba(88, 166, 255, 0.3); }
    .subtitle { font-size: 13px; color: var(--text-muted); margin-top: 4px; }

    .presets { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
    .preset-btn { background: var(--chip-bg); border: 1px solid var(--border); color: var(--text); font-size: 12px; padding: 6px 12px; border-radius: 6px; cursor: pointer; transition: 0.2s; }
    .preset-btn:hover { background: #30363d; border-color: var(--text-muted); }

    textarea { width: 100%; height: 110px; background: #0d1117; border: 1px solid var(--border); border-radius: 8px; color: var(--text); padding: 12px; font-size: 14px; line-height: 1.5; resize: vertical; margin-bottom: 20px; }
    textarea:focus { outline: none; border-color: var(--accent); }

    .control-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }
    @media (max-width: 600px) { .control-grid { grid-template-columns: 1fr; } }

    .control-card { background: #0d1117; border: 1px solid var(--border); border-radius: 8px; padding: 14px; }
    .control-label { font-size: 12px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px; display: flex; justify-content: space-between; }

    .speakers-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }
    @media (max-width: 600px) { .speakers-row { grid-template-columns: repeat(2, 1fr); } }
    .speaker-btn { background: var(--chip-bg); border: 1px solid var(--border); color: var(--text); padding: 10px 6px; border-radius: 6px; font-size: 12px; text-align: center; cursor: pointer; transition: 0.2s; }
    .speaker-btn:hover { border-color: var(--text-muted); }
    .speaker-btn.active { background: rgba(88, 166, 255, 0.2); border-color: var(--accent); color: var(--accent); font-weight: 600; }
    .speaker-btn .icon { font-size: 16px; display: block; margin-bottom: 2px; }

    .slider-row { display: flex; align-items: center; gap: 12px; }
    input[type="range"] { flex: 1; accent-color: var(--accent); }
    .slider-val { font-size: 13px; font-weight: 600; width: 44px; text-align: right; }

    .lang-row { display: flex; gap: 8px; }
    .lang-btn { flex: 1; background: var(--chip-bg); border: 1px solid var(--border); color: var(--text); padding: 8px; border-radius: 6px; font-size: 12px; cursor: pointer; text-align: center; }
    .lang-btn.active { background: rgba(88, 166, 255, 0.2); border-color: var(--accent); color: var(--accent); font-weight: 600; }

    .action-btn { width: 100%; background: var(--success); border: none; color: #fff; font-size: 15px; font-weight: 600; padding: 14px; border-radius: 8px; cursor: pointer; transition: 0.2s; display: flex; justify-content: center; align-items: center; gap: 8px; }
    .action-btn:hover { background: #2ea043; }
    .action-btn:disabled { opacity: 0.6; cursor: not-allowed; }

    .output-section { margin-top: 24px; padding-top: 20px; border-top: 1px solid var(--border); }
    .audio-player { width: 100%; margin-top: 12px; border-radius: 8px; outline: none; }
    .metrics-row { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 12px; }
    .metric-item { background: #0d1117; border: 1px solid var(--border); border-radius: 6px; padding: 6px 12px; font-size: 12px; }
    .metric-item span { color: var(--text-muted); margin-right: 4px; }
    .metric-item strong { color: var(--accent); }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="title-group">
        <h1>ArboTTS Studio 🎙️ <span class="badge">v2.0 (Multi-Speaker VITS)</span></h1>
        <div class="subtitle">Bilingual Text-to-Speech Indonesia & Inggris • Multi-Speaker VITS ONNX 67 MB</div>
      </div>
    </header>

    <div style="font-size: 11px; color: var(--text-muted); margin-bottom: 6px; font-weight: 600;">CONTOH KALIMAT (PRESETS)</div>
    <div class="presets" style="margin-bottom: 6px;">
      <span style="font-size: 11px; color: var(--accent); align-self: center; font-weight: 600;">🇮🇩 ID:</span>
      <button class="preset-btn" onclick="setPreset('Halo, selamat pagi, Indonesia. Awali hari Seninmu dengan penuh ceria!', 'id')">🌅 Sapaan Pagi</button>
      <button class="preset-btn" onclick="setPreset('Harga beras medium di pasar tradisional mencapai empat belas ribu rupiah per kilogram hari ini.', 'id')">🌾 Berita Pasar</button>
      <button class="preset-btn" onclick="setPreset('Ibu membeli sate ayam dan tempe bacem di tepi jalan sore hari.', 'id')">🦆 Uji Pepet & Taling</button>
    </div>
    <div class="presets" style="margin-bottom: 14px;">
      <span style="font-size: 11px; color: var(--accent); align-self: center; font-weight: 600;">🇬🇧 EN:</span>
      <button class="preset-btn" onclick="setPreset('Good morning, and welcome to ArboTTS. Have a wonderful and productive day!', 'en')">🇬🇧 English Welcome</button>
      <button class="preset-btn" onclick="setPreset('Breaking news: global tech markets showed positive momentum today following the latest report.', 'en')">🎙️ Broadcast News</button>
      <button class="preset-btn" onclick="setPreset('Certainly! I have scheduled your team meeting for tomorrow morning at nine.', 'en')">🤖 Smart Assistant</button>
      <button class="preset-btn" onclick="setPreset('The old oak tree stood quietly at the edge of the forest, watching generations pass by.', 'en')">📖 Calm Narration</button>
    </div>

    <div style="background:#0d1117;border:1px solid #238636;border-radius:8px;padding:12px 14px;margin-bottom:16px;">
      <div style="font-size:12px;font-weight:700;color:#3fb950;display:flex;align-items:center;gap:6px;margin-bottom:8px;">
        <span>🎙️ HASIL UJI COBA KLONING SUARA SAYA (USER CLONE AUDITIONS)</span>
      </div>
      <div style="display:grid;grid-template-columns:repeat(2, 1fr);gap:8px;">
        <button class="preset-btn" style="text-align:left;padding:8px 10px;" onclick="playCloneAudio('/audition/test_01_sapaan.wav', 'Sapaan Pembuka AI (12.2s)')">
          ▶️ <strong>Uji 1: Sapaan Pembuka</strong><br><span style="font-size:10px;color:var(--text-muted)">"Halo, ini contoh suara saya sendiri..."</span>
        </button>
        <button class="preset-btn" style="text-align:left;padding:8px 10px;" onclick="playCloneAudio('/audition/test_02_berita.wav', 'Berita Publik (10.6s)')">
          ▶️ <strong>Uji 2: Narasi Berita</strong><br><span style="font-size:10px;color:var(--text-muted)">"Pemerintah kota mengumumkan rencana..."</span>
        </button>
        <button class="preset-btn" style="text-align:left;padding:8px 10px;" onclick="playCloneAudio('/audition/test_03_santai.wav', 'Percakapan Santai (9.3s)')">
          ▶️ <strong>Uji 3: Percakapan Santai</strong><br><span style="font-size:10px;color:var(--text-muted)">"Kemarin sore saya mampir ke toko buku..."</span>
        </button>
        <button class="preset-btn" style="text-align:left;padding:8px 10px;" onclick="playCloneAudio('/audition/test_04_angka_rupiah.wav', 'Angka & Rupiah (8.2s)')">
          ▶️ <strong>Uji 4: Angka & Rupiah</strong><br><span style="font-size:10px;color:var(--text-muted)">"Total belanjaan... 375 ribu rupiah"</span>
        </button>
      </div>
    </div>

    <div class="control-label" style="margin-bottom:6px">PILIH MODEL TIER (UKURAN & KECEPATAN)</div>
    <div class="tier-row" style="display:flex;gap:10px;margin-bottom:16px">
      <div class="tier-btn active" id="tierStandard" onclick="setTier('standard')" style="flex:1;background:rgba(88,166,255,0.15);border:1px solid var(--accent);color:var(--text);padding:10px;border-radius:8px;cursor:pointer;transition:0.2s;text-align:center;">
        <div style="font-weight:700;font-size:13px;display:flex;align-items:center;justify-content:center;gap:6px;">🚀 Standard Tier <span style="font-size:10px;padding:2px 6px;border-radius:10px;background:rgba(88,166,255,0.2);color:var(--accent);">60 MB</span></div>
        <div style="font-size:11px;color:var(--text-muted);margin-top:2px;">Studio High-Fidelity VITS</div>
      </div>
      <div class="tier-btn" id="tierSmall" onclick="setTier('small')" style="flex:1;background:var(--chip-bg);border:1px solid var(--border);color:var(--text);padding:10px;border-radius:8px;cursor:pointer;transition:0.2s;text-align:center;">
        <div style="font-weight:700;font-size:13px;display:flex;align-items:center;justify-content:center;gap:6px;">⚡ Small Tier <span style="font-size:10px;padding:2px 6px;border-radius:10px;background:rgba(46,160,67,0.2);color:#3fb950;">31 MB</span></div>
        <div style="font-size:11px;color:var(--text-muted);margin-top:2px;">Distilled VITS (Ultra-Low RTF)</div>
      </div>
    </div>

    <textarea id="textInput" placeholder="Ketik kalimat yang ingin diucapkan di sini...">Halo, selamat pagi, Indonesia. Awali hari Seninmu dengan penuh ceria!</textarea>

    <div class="control-grid">
      <div class="control-card">
        <div class="control-label">Pilih Karakter Suara (Speaker)</div>
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
          <div style="font-size:10px;color:var(--accent);font-weight:700">⭐ PERSONA UTAMA (PRIA - REKOMENDASI):</div>
        </div>
        <div class="speakers-row" style="margin-bottom:10px">
          <div class="speaker-btn active" onclick="setSpeaker('arbo', this)">
            <span class="icon">🎙️</span> Arbo<br><span style="font-size:10px;color:var(--text-muted)">Penyiar (ONNX)</span>
          </div>
          <div class="speaker-btn" onclick="setSpeaker('bimo', this)">
            <span class="icon">📢</span> Bimo<br><span style="font-size:10px;color:var(--text-muted)">Baritone</span>
          </div>
          <div class="speaker-btn" onclick="setSpeaker('dimas', this)">
            <span class="icon">⚡</span> Dimas<br><span style="font-size:10px;color:var(--text-muted)">Kasual</span>
          </div>
          <div class="speaker-btn" onclick="setSpeaker('danu', this)">
            <span class="icon">☕</span> Danu<br><span style="font-size:10px;color:var(--text-muted)">Santun</span>
          </div>
        </div>
        <div style="font-size:10px;color:var(--text-muted);margin-bottom:6px;font-weight:600">👩 PERSONA TAMBAHAN (WANITA):</div>
        <div class="speakers-row" style="grid-template-columns: repeat(3, 1fr)">
          <div class="speaker-btn" onclick="setSpeaker('arba', this)">
            <span class="icon">👩</span> Arba<br><span style="font-size:10px;color:var(--text-muted)">Presenter</span>
          </div>
          <div class="speaker-btn" onclick="setSpeaker('sita', this)">
            <span class="icon">🤖</span> Sita<br><span style="font-size:10px;color:var(--text-muted)">Asisten</span>
          </div>
          <div class="speaker-btn" onclick="setSpeaker('maya', this)">
            <span class="icon">📖</span> Maya<br><span style="font-size:10px;color:var(--text-muted)">Narasi</span>
          </div>
        </div>
      </div>

      <div class="control-card">
        <div class="control-label">
          <span>Tempo Bicara (Speed)</span>
          <span class="slider-val" id="speedVal">0.85x</span>
        </div>
        <div class="slider-row" style="margin-top: 6px;">
          <input type="range" id="speedSlider" min="0.5" max="1.5" step="0.05" value="0.85" oninput="updateSpeed(this.value)">
        </div>

        <div class="control-label" style="margin-top: 14px;">Bahasa (Language)</div>
        <div class="lang-row">
          <div class="lang-btn active" id="langId" onclick="setLang('id')">🇮🇩 Indonesia</div>
          <div class="lang-btn" id="langEn" onclick="setLang('en')">🇬🇧 English</div>
        </div>
      </div>
    </div>

    <button id="genBtn" class="action-btn" onclick="generateSpeech()">
      <span>⚡ Sintesis Suara (Generate Speech)</span>
    </button>

    <div id="outputSection" class="output-section" style="display:none;">
      <div class="control-label">Hasil Audio</div>
      <audio id="audioPlayer" class="audio-player" controls autoplay></audio>
      <div class="metrics-row">
        <div class="metric-item"><span>Tier:</span><strong id="metricTier">-</strong></div>
        <div class="metric-item"><span>Latensi CPU:</span><strong id="metricLatency">-</strong></div>
        <div class="metric-item"><span>Durasi Audio:</span><strong id="metricDuration">-</strong></div>
        <div class="metric-item"><span>Real-Time Factor:</span><strong id="metricRTF">-</strong></div>
        <div class="metric-item"><span>Speaker:</span><strong id="metricSpeaker">-</strong></div>
      </div>
    </div>
  </div>

  <script>
    let currentTier = 'standard';
    let currentSpeaker = 'arbo';
    let currentLang = 'id';
    let currentSpeed = 0.85;

    function setTier(tier) {
      currentTier = tier;
      const isStd = tier === 'standard';
      const stdEl = document.getElementById('tierStandard');
      const smlEl = document.getElementById('tierSmall');
      stdEl.classList.toggle('active', isStd);
      smlEl.classList.toggle('active', !isStd);
      if (isStd) {
        stdEl.style.borderColor = 'var(--accent)';
        stdEl.style.background = 'rgba(88, 166, 255, 0.15)';
        smlEl.style.borderColor = 'var(--border)';
        smlEl.style.background = 'var(--chip-bg)';
      } else {
        smlEl.style.borderColor = '#3fb950';
        smlEl.style.background = 'rgba(46, 160, 67, 0.15)';
        stdEl.style.borderColor = 'var(--border)';
        stdEl.style.background = 'var(--chip-bg)';
      }
    }

    function playCloneAudio(url, title) {
      const player = document.getElementById('audioPlayer');
      player.src = url;
      document.getElementById('outputSection').style.display = 'block';
      document.getElementById('metricSpeaker').textContent = 'USER CLONE';
      document.getElementById('metricTier').textContent = 'CLONE';
      document.getElementById('metricLatency').textContent = 'Prasintesis';
      document.getElementById('metricDuration').textContent = title;
      document.getElementById('metricRTF').textContent = '1.0x';
      player.play().catch(e => console.log('Autoplay prevented:', e));
    }

    function setPreset(text, lang = 'id') {
      document.getElementById('textInput').value = text;
      setLang(lang);
    }

    function setSpeaker(name, el) {
      currentSpeaker = name;
      document.querySelectorAll('.speaker-btn').forEach(b => b.classList.remove('active'));
      el.classList.add('active');
    }

    function updateSpeed(val) {
      currentSpeed = parseFloat(val);
      document.getElementById('speedVal').textContent = currentSpeed.toFixed(2) + 'x';
    }

    function setLang(lang) {
      currentLang = lang;
      document.getElementById('langId').classList.toggle('active', lang === 'id');
      document.getElementById('langEn').classList.toggle('active', lang === 'en');
    }

    // Ctrl+Enter / Cmd+Enter to generate
    document.getElementById('textInput').addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        generateSpeech();
      }
    });

    async function generateSpeech() {
      const text = document.getElementById('textInput').value.trim();
      if (!text) return;

      const btn = document.getElementById('genBtn');
      btn.disabled = true;
      btn.innerHTML = '<span>⏳ Memproses di CPU...</span>';

      const t0 = performance.now();
      try {
        const res = await fetch('/v1/audio/speech', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            input: text,
            voice: currentSpeaker,
            lang: currentLang,
            speed: currentSpeed,
            tier: currentTier,
            response_format: 'wav'
          })
        });

        if (!res.ok) throw new Error('Sintesis gagal');

        const tierHeader = res.headers.get('X-Tier');
        const latencyHeader = res.headers.get('X-Latency-Ms');
        const durHeader = res.headers.get('X-Duration-Sec');
        const rtfHeader = res.headers.get('X-RTF');

        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const player = document.getElementById('audioPlayer');
        player.src = url;

        document.getElementById('outputSection').style.display = 'block';
        document.getElementById('metricTier').textContent = tierHeader ? tierHeader : currentTier.toUpperCase();
        document.getElementById('metricLatency').textContent = latencyHeader ? `${parseFloat(latencyHeader).toFixed(0)} ms` : `${(performance.now() - t0).toFixed(0)} ms`;
        document.getElementById('metricDuration').textContent = durHeader ? `${parseFloat(durHeader).toFixed(2)} s` : '-';
        document.getElementById('metricRTF').textContent = rtfHeader ? `${parseFloat(rtfHeader).toFixed(3)}x` : '-';
        document.getElementById('metricSpeaker').textContent = currentSpeaker.toUpperCase();
      } catch (err) {
        alert('Gagal menghasilkan suara: ' + err.message);
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<span>⚡ Sintesis Suara (Generate Speech)</span>';
      }
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Serve ArboTTS Studio Web UI."""
    return HTMLResponse(content=WEB_UI_HTML)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "2.0.0"}


@app.get("/audition/{filename}")
def get_audition_audio(filename: str) -> Response:
    audition_dir = REPO_ROOT / "audition_clone"
    file_path = audition_dir / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Berkas uji coba suara tidak ditemukan")
    data = file_path.read_bytes()
    return Response(content=data, media_type="audio/wav")


@app.get("/v1/speakers")
def list_speakers() -> dict[str, Any]:
    engine = get_engine("standard")
    return engine.profiles.get("speakers", {})


@app.get("/v1/tiers")
def list_tiers() -> dict[str, Any]:
    tiers_path = REPO_ROOT / "configs" / "tiers.json"
    if tiers_path.exists():
        with open(tiers_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"tiers": {}}


@app.post("/v1/audio/speech")
def create_speech(req: OpenAISpeechRequest) -> Response:
    """OpenAI-compatible Text-to-Speech endpoint."""
    engine = get_engine(req.tier)
    t0 = time.perf_counter()

    audio, synth_latency_ms = engine.synthesize(
        text=req.input,
        lang=req.lang,
        speaker=req.voice,
        speed=req.speed,
    )
    total_time_ms = (time.perf_counter() - t0) * 1000
    duration_sec = len(audio) / engine.sampling_rate
    rtf = (total_time_ms / 1000.0) / max(duration_sec, 1e-4)

    buf = io.BytesIO()
    sf.write(buf, audio, engine.sampling_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)

    headers = {
        "Content-Type": "audio/wav",
        "Content-Disposition": f'inline; filename="{req.voice}_speech.wav"',
        "X-Tier": req.tier.upper(),
        "X-Latency-Ms": f"{total_time_ms:.1f}",
        "X-Duration-Sec": f"{duration_sec:.2f}",
        "X-RTF": f"{rtf:.4f}",
    }
    return Response(content=buf.getvalue(), media_type="audio/wav", headers=headers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start ArboTTS Web UI & REST API server.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--model", type=str, default=None, help="Custom ONNX model path")
    parser.add_argument("--config", type=str, default=None, help="Custom config json path")

    args = parser.parse_args()

    # Preload default standard engine and small engine if available
    _ENGINES["standard"] = ArboTTS(model_path=args.model, config_path=args.config, tier="standard")
    try:
        _ENGINES["small"] = ArboTTS(tier="small")
    except Exception as e:
        print(f"[Warning] Small tier engine preload skipped: {e}")

    print(f"\n=======================================================")
    print(f"🎙️  ArboTTS Studio (v2.0) running at: http://{args.host}:{args.port}")
    print(f"📡 OpenAI-compatible endpoint: POST http://{args.host}:{args.port}/v1/audio/speech")
    print(f"   Tiers available: Standard (60 MB), Small (31 MB)")
    print(f"=======================================================\n")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
