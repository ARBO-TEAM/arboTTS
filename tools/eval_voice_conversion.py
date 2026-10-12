import sys
import time
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arbotts import ArboTTS
from arbotts.vc import ToneColorConverter

def analyze_audio(y: np.ndarray, sr: int = 22050):
    f0, _, _ = librosa.pyin(y, fmin=60, fmax=350, sr=sr)
    f0_clean = f0[~np.isnan(f0)]
    f0_med = float(np.median(f0_clean)) if len(f0_clean) > 0 else 0.0
    f0_std = float(np.std(f0_clean)) if len(f0_clean) > 0 else 0.0
    sc = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    rms = float(np.sqrt(np.mean(y**2)))
    dur = float(len(y) / sr)
    return {"dur": dur, "f0_med": f0_med, "f0_std": f0_std, "sc": sc, "rms": rms}

def main():
    print("=== ArboTTS + ToneColorConverter End-to-End Evaluation on CPU ===")
    
    # 1. Load Engines
    print("Loading ArboTTS Base ONNX engine...")
    tts = ArboTTS(tier="standard")
    
    print("Loading ToneColorConverter on CPU...")
    vc = ToneColorConverter(device="cpu")
    
    # 2. Extract Target Speaker Embedding from User Ref
    ref_path = REPO_ROOT / "recordings" / "user_ref.wav"
    print(f"Extracting user reference embedding from {ref_path.name}...")
    target_se = vc.extract_se(ref_path)
    
    # Target profile
    y_ref, sr_ref = librosa.load(str(ref_path), sr=22050)
    ref_stats = analyze_audio(y_ref, sr_ref)
    print(f"\n[TARGET USER REFERENCE]")
    print(f"  F0 Median: {ref_stats['f0_med']:.1f} Hz | F0 Std: {ref_stats['f0_std']:.1f} Hz")
    print(f"  Spectral Centroid: {ref_stats['sc']:.1f} Hz | RMS: {ref_stats['rms']:.4f}\n")
    
    test_sentences = [
        (
            "vc_eval_01_sapaan.wav",
            "Halo, selamat pagi semuanya! Ini adalah uji coba suara saya sendiri menggunakan teknologi voice conversion ArboTTS."
        ),
        (
            "vc_eval_02_teknologi.wav",
            "Sistem ini menggabungkan kecepatan VITS ONNX dengan pengubah warna suara berbasis flow yang sangat ringan di prosesor komputer."
        ),
        (
            "vc_eval_03_santai.wav",
            "Gimana menurut kamu? Sekarang suara saya bisa membaca teks apapun secara lancar dan jelas tanpa perlu menyalakan kartu grafis."
        )
    ]
    
    out_dir = REPO_ROOT / "audition_clone"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'Sentence':<24} | {'TTS (ms)':<9} | {'VC (ms)':<8} | {'Total (ms)':<10} | {'Dur (s)':<7} | {'RTF':<6} | {'F0 (Hz)':<7} | {'Centroid':<8}")
    print("-" * 92)
    
    for filename, text in test_sentences:
        # TTS synthesis
        t_start = time.perf_counter()
        base_audio, tts_latency_ms = tts.synthesize(text=text, lang="id", speaker="arbo", speed=0.9)
        t_tts_end = time.perf_counter()
        
        # VC conversion
        converted_audio, vc_sec = vc.convert(base_audio, target_se=target_se, tau=0.3)
        t_total_sec = time.perf_counter() - t_start
        
        # Normalize
        max_val = np.abs(converted_audio).max()
        if max_val > 0.01:
            converted_audio = converted_audio / max_val * 0.95
            
        stats = analyze_audio(converted_audio, sr=22050)
        rtf = t_total_sec / stats['dur']
        
        out_file = out_dir / filename
        sf.write(str(out_file), converted_audio, 22050)
        
        print(
            f"{filename:<24} | "
            f"{tts_latency_ms:>8.1f} | "
            f"{vc_sec*1000:>7.1f} | "
            f"{t_total_sec*1000:>9.1f} | "
            f"{stats['dur']:>6.2f} | "
            f"{rtf:>5.3f} | "
            f"{stats['f0_med']:>6.1f} | "
            f"{stats['sc']:>7.1f}"
        )

    print("-" * 92)
    print("Done! All samples generated and saved to audition_clone/")

if __name__ == "__main__":
    main()
