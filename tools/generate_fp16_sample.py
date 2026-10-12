import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arbotts.vc import ToneColorConverter

print("=== Generating FP16 / BFloat16 Sample ===")

vc = ToneColorConverter(device="cpu")
ref_wav = REPO_ROOT / "recordings" / "user_ref.wav"
src_wav = REPO_ROOT / "audition_clone" / "test_01_sapaan.wav"

target_se = vc.extract_se(ref_wav)
source_se = vc.extract_se(src_wav)

y, _ = librosa.load(str(src_wav), sr=vc.sampling_rate)
y_t = torch.FloatTensor(y).unsqueeze(0)
spec = vc._get_spec(y_t)
spec_len = torch.LongTensor([spec.size(-1)])

# Run in CPU BFloat16 (Half Precision)
t0 = time.perf_counter()
with torch.no_grad():
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out_bf16 = vc.model.voice_conversion(
            spec, spec_len, sid_src=source_se, sid_tgt=target_se, tau=0.3
        )[0][0, 0].cpu().float().numpy()
t_bf16 = time.perf_counter() - t0
dur = len(out_bf16) / vc.sampling_rate

# Normalize
out_bf16_norm = out_bf16 / max(np.abs(out_bf16).max(), 1e-4) * 0.95
out_file = REPO_ROOT / "audition_clone" / "eval_vc_fp16.wav"
sf.write(str(out_file), out_bf16_norm, vc.sampling_rate)

# Acoustic analysis
f0, _, _ = librosa.pyin(out_bf16_norm, fmin=60, fmax=350, sr=22050)
f0_clean = f0[~np.isnan(f0)]
f0_med = float(np.median(f0_clean)) if len(f0_clean) > 0 else 0.0
sc = float(np.mean(librosa.feature.spectral_centroid(y=out_bf16_norm, sr=22050)))

print(f"FP16/BF16 Latency: {t_bf16*1000:.1f} ms | Duration: {dur:.2f}s | RTF: {t_bf16/dur:.3f}")
print(f"F0 Median: {f0_med:.1f} Hz | Spectral Centroid: {sc:.1f} Hz")
print(f"Saved: {out_file.name}")
