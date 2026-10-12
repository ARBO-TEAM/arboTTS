import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arbotts import ArboTTS
from arbotts.vc import ToneColorConverter

def main():
    print("=== Generating User Voice Variations with ToneColorConverter ===")
    tts = ArboTTS(tier="standard")
    vc = ToneColorConverter(device="cpu")
    
    # Use exact reference and base embeddings
    user_ref = REPO_ROOT / "recordings" / "user_ref.wav"
    target_se = vc.extract_se(user_ref)
    
    # Base Arbo embedding
    arbo_se_path = REPO_ROOT / "checkpoints_vc" / "arbo_base_se.pth"
    if arbo_se_path.exists():
        source_se = torch.load(arbo_se_path, map_location="cpu")
    else:
        source_se = None

    sentences = [
        (
            "variasi_01_santai_ngopi.wav",
            "Bro, nanti sore jadi ngopi nggak? Kebetulan kerjaan kantor udah beres semua nih, kita cari tempat yang adem aja.",
            "Obrolan Santai / Kasual"
        ),
        (
            "variasi_02_formal_kantor.wav",
            "Selamat siang rekan-rekan. Melalui pesan ini saya ingin menyampaikan bahwa pembaruan sistem tahap kedua telah berhasil diuji dan siap diimplementasikan minggu depan.",
            "Formal / Presentasi Kantor"
        ),
        (
            "variasi_03_asisten_jadwal.wav",
            "Pengingat harian: Hari ini kamu ada tiga agenda penting, termasuk pertemuan dengan tim pengembang pada pukul dua siang.",
            "Asisten Cerdas / Pengingat"
        ),
        (
            "variasi_04_podcast_edukasi.wav",
            "Banyak orang mengira kecerdasan buatan itu rumit, padahal kuncinya ada pada bagaimana kita memanfaatkan teknologi untuk mempermudah rutinitas sehari-hari.",
            "Narasi Podcast / Edukasi"
        ),
        (
            "variasi_05_angka_rupiah.wav",
            "Total transaksi yang tercatat adalah sebesar satu juta lima ratus ribu rupiah, atas nama Budi Santoso di Jakarta Selatan.",
            "Uji Angka, Rupiah & Nama"
        ),
    ]

    out_dir = REPO_ROOT / "audition_clone"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'Filename':<30} | {'Kategori':<26} | {'Durasi':<8} | {'Latency':<9}")
    print("-" * 78)

    for fname, text, category in sentences:
        t0 = time.perf_counter()
        # 1. Base TTS synthesis (Arbo)
        base_audio, tts_ms = tts.synthesize(text, lang="id", speaker="arbo", speed=0.9)
        
        # 2. Voice Conversion
        converted_audio, vc_sec = vc.convert(base_audio, target_se=target_se, source_se=source_se, tau=0.3)
        total_ms = (time.perf_counter() - t0) * 1000
        dur_s = len(converted_audio) / 22050
        
        # Normalize peak
        max_val = np.abs(converted_audio).max()
        if max_val > 0.01:
            converted_audio = converted_audio / max_val * 0.95
            
        out_path = out_dir / fname
        sf.write(str(out_path), converted_audio, 22050)
        
        print(f"{fname:<30} | {category:<26} | {dur_s:>6.2f}s | {total_ms:>7.1f} ms")

    print("-" * 78)
    print("All variations generated successfully!")

if __name__ == "__main__":
    main()
