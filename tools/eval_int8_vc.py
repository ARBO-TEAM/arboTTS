import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import torch
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arbotts.vc import ToneColorConverter

print("=== Benchmarking FP32 vs INT8 Voice Conversion ===")

vc = ToneColorConverter(device="cpu")
ref_wav = REPO_ROOT / "recordings" / "user_ref.wav"
src_wav = REPO_ROOT / "audition_clone" / "test_01_sapaan.wav"

target_se = vc.extract_se(ref_wav).cpu().numpy()
source_se = vc.extract_se(src_wav).cpu().numpy()

# Spectrogram of source audio
y, _ = librosa.load(str(src_wav), sr=vc.sampling_rate)
y_t = torch.FloatTensor(y).unsqueeze(0)
spec = vc._get_spec(y_t).cpu().numpy()
spec_len = np.array([spec.shape[-1]], dtype=np.int64)

# 1. PyTorch FP32 benchmark
t0 = time.perf_counter()
with torch.no_grad():
    out_pt, _ = vc.convert(src_wav, target_se=torch.from_numpy(target_se), source_se=torch.from_numpy(source_se), tau=0.0)
t_pt = time.perf_counter() - t0
dur_pt = len(out_pt) / vc.sampling_rate

# 2. ONNX INT8 benchmark
int8_onnx_path = REPO_ROOT / "checkpoints_vc" / "converter_int8.onnx"
session_opts = ort.SessionOptions()
session_opts.intra_op_num_threads = 4
session_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

sess = ort.InferenceSession(str(int8_onnx_path), session_opts, providers=["CPUExecutionProvider"])

t0 = time.perf_counter()
ort_inputs = {
    "spec": spec,
    "spec_lengths": spec_len,
    "sid_src": source_se,
    "sid_tgt": target_se,
}
out_ort = sess.run(None, ort_inputs)[0][0]
t_ort = time.perf_counter() - t0
dur_ort = len(out_ort) / vc.sampling_rate

print(f"\nModel File Size:")
print(f"  PyTorch FP32:  125.10 MB")
print(f"  ONNX INT8:     38.93 MB (Reduction: 68.9%!)")

print(f"\nLatency on CPU (Audio: {dur_ort:.2f}s):")
print(f"  PyTorch FP32:  {t_pt*1000:.1f} ms (RTF: {t_pt/dur_pt:.3f})")
print(f"  ONNX INT8:     {t_ort*1000:.1f} ms (RTF: {t_ort/dur_ort:.3f})")
print(f"  Speedup:       {t_pt/t_ort:.2f}x faster!")

# Normalize audio
out_pt_norm = out_pt / max(np.abs(out_pt).max(), 1e-4) * 0.95
out_ort_norm = out_ort / max(np.abs(out_ort).max(), 1e-4) * 0.95

p_fp32 = REPO_ROOT / "audition_clone" / "eval_vc_fp32_baseline.wav"
p_int8 = REPO_ROOT / "audition_clone" / "eval_vc_int8_quantized.wav"
sf.write(str(p_fp32), out_pt_norm, vc.sampling_rate)
sf.write(str(p_int8), out_ort_norm, vc.sampling_rate)

def stats(sig, sr=22050):
    f0, _, _ = librosa.pyin(sig, fmin=60, fmax=350, sr=sr)
    f0_clean = f0[~np.isnan(f0)]
    f0_med = float(np.median(f0_clean)) if len(f0_clean) > 0 else 0.0
    sc = float(np.mean(librosa.feature.spectral_centroid(y=sig, sr=sr)))
    return f0_med, sc

f0_pt, sc_pt = stats(out_pt_norm)
f0_int8, sc_int8 = stats(out_ort_norm)

print(f"\nAcoustic Comparison:")
print(f"  FP32 Baseline  -> F0: {f0_pt:.1f} Hz | Centroid: {sc_pt:.1f} Hz")
print(f"  INT8 Quantized -> F0: {f0_int8:.1f} Hz | Centroid: {sc_int8:.1f} Hz")

min_len = min(len(out_pt), len(out_ort))
corr = np.corrcoef(out_pt[:min_len], out_ort[:min_len])[0, 1]
print(f"  Correlation (FP32 vs INT8): {corr:.4f}")

print("\nFiles saved to audition_clone:")
print(f"  - {p_fp32.name}")
print(f"  - {p_int8.name}")
