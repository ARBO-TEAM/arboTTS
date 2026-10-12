"""ArboTTS Small Tier (24MB) Student Model Distillation Training.

Distills knowledge from the 60MB Standard teacher (epoch199.ckpt)
into a compact 8.39M-parameter (hidden_channels=128, n_layers=4)
student model on NVIDIA CUDA GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from pathlib import Path

# Fix Windows PosixPath unpickling
pathlib.PosixPath = pathlib.WindowsPath

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPER_DIR = Path("D:/kerjaan/belajar/tts/piper/src/python").resolve()
SANOTTS_DIR = Path("D:/kerjaan/belajar/tts/sanoTTS/pypkg").resolve()

for p in [REPO_ROOT, PIPER_DIR, SANOTTS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import math
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch import autocast, optim
from torch.utils.data import DataLoader

from piper_train.vits import monotonic_align
from piper_train.vits.commons import rand_slice_segments, slice_segments
from piper_train.vits.losses import discriminator_loss, feature_loss, generator_loss
from piper_train.vits.mel_processing import (
    mel_spectrogram_torch,
    spec_to_mel_torch,
)
from piper_train.vits.models import MultiPeriodDiscriminator, SynthesizerTrn
from sanotts import frontend
from training.train_finetune_vits import ArboFTCollate, ArboFTDataset, safe_kl_loss


def init_student_from_teacher(student: SynthesizerTrn, teacher_sd: dict[str, torch.Tensor]) -> None:
    """Warm-start student model with sliced and matched weights from teacher.
    Duration predictor (dp) is initialized cleanly from scratch to preserve flow transformations.
    """
    student_sd = student.state_dict()
    transferred = 0
    sliced = 0

    for k, v in teacher_sd.items():
        if k not in student_sd or k.startswith("dp."):
            continue
        tgt_shape = student_sd[k].shape
        src_shape = v.shape

        if tgt_shape == src_shape:
            student_sd[k] = v.clone()
            transferred += 1
        elif len(tgt_shape) == len(src_shape):
            # Slice matching dimensions
            slices = tuple(slice(0, min(td, sd)) for td, sd in zip(tgt_shape, src_shape))
            student_sd[k][slices] = v[slices].clone()
            sliced += 1

    student.load_state_dict(student_sd)
    print(f"[Distillation Init] Transferred {transferred} exact layers and {sliced} sliced layers from Teacher to Student.")


def evaluate_audition(
    model: SynthesizerTrn,
    table: frontend.PhonemeTable,
    step: int,
    out_dir: Path,
    device: torch.device,
) -> None:
    """Synthesize test audition for Small tier Arbo and Arba."""
    test_text = "Halo, selamat datang di ArboTTS Studio Small Tier. Suara ini sangat cepat dan jernih."
    phoneme_ids = frontend.text_to_phoneme_ids(test_text, table)
    x = torch.tensor(phoneme_ids, dtype=torch.long, device=device).unsqueeze(0)
    x_lengths = torch.tensor([len(phoneme_ids)], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        for spk_name, spk_id in [("arbo", 0), ("arba", 1)]:
            sid_t = torch.tensor([spk_id], dtype=torch.long, device=device)
            audio_tensor = model.infer(
                x,
                x_lengths,
                sid=sid_t,
                noise_scale=0.667,
                length_scale=1.0,
                noise_scale_w=0.8,
            )[0][0, 0]
            audio = audio_tensor.cpu().numpy()

            max_val = np.abs(audio).max()
            if max_val > 1.0:
                audio = audio / max_val
            elif max_val > 0.01:
                audio = audio * 0.95

            out_file = out_dir / f"audition_small_{step:04d}_{spk_name}.wav"
            sf.write(str(out_file), audio, 22050, subtype="PCM_16")

            duration = len(audio) / 22050
            rms = float(np.sqrt(np.mean(audio**2)))
            print(
                f"--> [Small Tier Audition Step {step} | {spk_name.upper()}] Saved {out_file.name} | "
                f"Dur: {duration:.2f}s | RMS: {rms:.4f}",
                flush=True,
            )

    model.train()


def train_distill(
    base_ckpt: str = "D:/kerjaan/belajar/tts/sanoTTS/experiments/epoch199.ckpt",
    cfg_json: str = "D:/kerjaan/belajar/tts/arboTTS/arbotts_small.onnx.json",
    meta_txt: str = "dataset/train_multispeaker.txt",
    ckpt_dir_name: str = "checkpoints_vits_small",
    max_steps: int = 1000,
    batch_size: int = 16,
    lr: float = 1e-4,
    eval_interval: int = 200,
    save_interval: int = 500,
) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\n=======================================================")
    print(f"🚀 ArboTTS Small Tier Knowledge Distillation on {gpu_name}")
    print(f"=======================================================")

    # 1. Load Pretrained Teacher Checkpoint
    ckpt_path = Path(base_ckpt)
    print(f"Loading Teacher Checkpoint: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]

    cfg_path = Path(cfg_json)
    table = frontend.load_phoneme_table(cfg_path)
    print("Loaded Phoneme Table.")

    # 2. Build Teacher Model (Frozen)
    net_g_teacher = SynthesizerTrn(
        n_vocab=hp["num_symbols"],
        spec_channels=hp["filter_length"] // 2 + 1,
        segment_size=hp["segment_size"] // hp["hop_length"],
        inter_channels=hp["inter_channels"],
        hidden_channels=hp["hidden_channels"],
        filter_channels=hp["filter_channels"],
        n_heads=hp["n_heads"],
        n_layers=hp["n_layers"],
        kernel_size=hp["kernel_size"],
        p_dropout=hp["p_dropout"],
        resblock=str(hp["resblock"]),
        resblock_kernel_sizes=hp["resblock_kernel_sizes"],
        resblock_dilation_sizes=hp["resblock_dilation_sizes"],
        upsample_rates=hp["upsample_rates"],
        upsample_initial_channel=hp["upsample_initial_channel"],
        upsample_kernel_sizes=hp["upsample_kernel_sizes"],
        n_speakers=2,
        gin_channels=256,
        use_sdp=hp["use_sdp"],
    ).to(device)

    sd = ckpt["state_dict"]
    g_teacher_sd = {k.replace("model_g.", ""): v for k, v in sd.items() if k.startswith("model_g.")}
    d_sd = {k.replace("model_d.", ""): v for k, v in sd.items() if k.startswith("model_d.")}

    net_g_teacher.load_state_dict(g_teacher_sd, strict=False)
    net_g_teacher.eval()
    for p in net_g_teacher.parameters():
        p.requires_grad = False
    print("Teacher model instantiated and frozen.")

    # 3. Build Student Model (Small Tier: hidden=128, filter=512, layers=4)
    with open(cfg_path, "r", encoding="utf-8") as f:
        small_cfg = json.load(f)["model"]

    net_g = SynthesizerTrn(
        n_vocab=hp["num_symbols"],
        spec_channels=hp["filter_length"] // 2 + 1,
        segment_size=hp["segment_size"] // hp["hop_length"],
        inter_channels=small_cfg["hidden_channels"],
        hidden_channels=small_cfg["hidden_channels"],
        filter_channels=small_cfg["filter_channels"],
        n_heads=small_cfg["n_heads"],
        n_layers=small_cfg["n_layers"],
        kernel_size=hp["kernel_size"],
        p_dropout=hp["p_dropout"],
        resblock=str(small_cfg["resblock"]),
        resblock_kernel_sizes=hp["resblock_kernel_sizes"],
        resblock_dilation_sizes=hp["resblock_dilation_sizes"],
        upsample_rates=hp["upsample_rates"],
        upsample_initial_channel=small_cfg["upsample_initial_channel"],
        upsample_kernel_sizes=hp["upsample_kernel_sizes"],
        n_speakers=2,
        gin_channels=256,
        use_sdp=hp["use_sdp"],
    ).to(device)

    # Transfer pre-trained vocoder & sliced encoder layers
    init_student_from_teacher(net_g, g_teacher_sd)

    # 4. Build Discriminator
    net_d = MultiPeriodDiscriminator(use_spectral_norm=hp["use_spectral_norm"]).to(device)
    net_d.load_state_dict(d_sd, strict=True)
    print("MultiPeriodDiscriminator initialized from Teacher.")

    # 5. Check for Resuming Checkpoints
    ckpt_dir = REPO_ROOT / ckpt_dir_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    existing_g = sorted(
        [p for p in ckpt_dir.glob("G_small_*.pth") if p.stem.split("_")[-1].isdigit()],
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    global_step = 0
    if existing_g:
        last_step = int(existing_g[-1].stem.split("_")[-1])
        if last_step < max_steps:
            last_d = ckpt_dir / f"D_small_{last_step}.pth"
            if last_d.exists():
                print(f"Resuming Student Distillation from step {last_step}...")
                net_g.load_state_dict(torch.load(existing_g[-1], map_location=device))
                net_d.load_state_dict(torch.load(last_d, map_location=device))
                global_step = last_step

    # 6. Optimizers
    optim_g = optim.AdamW(net_g.parameters(), lr=lr, betas=hp["betas"], eps=hp["eps"])
    optim_d = optim.AdamW(net_d.parameters(), lr=lr, betas=hp["betas"], eps=hp["eps"])

    # 7. Dataset and DataLoader
    dataset = ArboFTDataset(
        metadata_path=meta_txt,
        table=table,
        filter_length=hp["filter_length"],
        hop_length=hp["hop_length"],
        win_length=hp["win_length"],
        sampling_rate=hp["sample_rate"],
    )
    collate_fn = ArboFTCollate()

    spk_counts = {}
    for item in dataset.items:
        spk = item[2]
        spk_counts[spk] = spk_counts.get(spk, 0) + 1
    weights = [1.0 / spk_counts[item[2]] for item in dataset.items]
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    print(f"Loaded {len(dataset)} training clips. Batch size: {batch_size}. Batches per epoch: {len(loader)}")

    net_g.train()
    net_d.train()

    c_mel = float(hp["c_mel"])
    c_kl = float(hp["c_kl"])
    segment_size = hp["segment_size"]
    hop_length = hp["hop_length"]

    start_time = time.time()
    epoch = 0

    if global_step == 0:
        evaluate_audition(net_g, table, 0, ckpt_dir, device)

    print(f"\nStarting Distillation Training: Step {global_step} -> {max_steps}...", flush=True)

    while global_step < max_steps:
        epoch += 1
        for batch in loader:
            if not batch:
                continue

            x, x_lengths, spec, spec_lengths, y, y_lengths, sid = [b.to(device) for b in batch]

            # ------------------------------------
            # 1. Forward Teacher Prior (No Grad)
            # ------------------------------------
            with torch.no_grad():
                with autocast("cuda", dtype=torch.bfloat16):
                    _, m_p_t, logs_p_t, _ = net_g_teacher.enc_p(x, x_lengths)

            # ------------------------------------
            # 2. Train Discriminator
            # ------------------------------------
            optim_d.zero_grad()
            with autocast("cuda", dtype=torch.bfloat16):
                (
                    y_hat_s,
                    l_length,
                    attn,
                    ids_slice,
                    x_mask,
                    y_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = net_g(x, x_lengths, spec, spec_lengths, sid=sid)

                y_slice = slice_segments(y.unsqueeze(1), ids_slice * hop_length, segment_size)
                y_d_rs, y_d_gs, fmap_rs, fmap_gs = net_d(y_slice, y_hat_s.detach())
                loss_disc, _, _ = discriminator_loss(y_d_rs, y_d_gs)

            if torch.isnan(loss_disc) or torch.isinf(loss_disc):
                print(f"[Warning] Step {global_step} skipped due to NaN/Inf in loss_disc!", flush=True)
                optim_d.zero_grad()
            else:
                loss_disc.backward()
                norm_d = torch.nn.utils.clip_grad_norm_(net_d.parameters(), 1.0)
                if torch.isnan(norm_d) or torch.isinf(norm_d):
                    optim_d.zero_grad()
                else:
                    optim_d.step()

            # ------------------------------------
            # 3. Train Student Generator (Distilled)
            # ------------------------------------
            optim_g.zero_grad()
            with autocast("cuda", dtype=torch.bfloat16):
                # Distill text prior representations
                _, m_p_s, logs_p_s, _ = net_g.enc_p(x, x_lengths)
                h_ch = small_cfg["hidden_channels"]
                loss_distill_prior = (
                    F.l1_loss(m_p_s, m_p_t[:, :h_ch, :]) +
                    F.l1_loss(logs_p_s, logs_p_t[:, :h_ch, :])
                ) * 5.0

                y_d_rs, y_d_gs, fmap_rs, fmap_gs = net_d(y_slice, y_hat_s)
                loss_gen, _ = generator_loss(y_d_gs)
                loss_fm = feature_loss(fmap_rs, fmap_gs)
                loss_dur = torch.sum(l_length.float())

            with autocast("cuda", enabled=False):
                y_mel = slice_segments(
                    spec_to_mel_torch(
                        spec.float(),
                        hp["filter_length"],
                        hp["mel_channels"],
                        hp["sample_rate"],
                        hp["mel_fmin"],
                        hp["mel_fmax"],
                    ),
                    ids_slice,
                    segment_size // hop_length,
                )
                y_hat_s_mel = mel_spectrogram_torch(
                    y_hat_s.squeeze(1).float(),
                    hp["filter_length"],
                    hp["mel_channels"],
                    hp["sample_rate"],
                    hop_length,
                    hp["win_length"],
                    hp["mel_fmin"],
                    hp["mel_fmax"],
                )
                # Ground truth Mel loss
                loss_mel = F.l1_loss(y_mel, y_hat_s_mel) * c_mel

                # VITS safe KL divergence
                loss_kl_val = safe_kl_loss(z_p, logs_q, m_p, logs_p, y_mask) * c_kl

                loss_total = loss_gen + loss_fm + loss_mel + loss_distill_prior + loss_kl_val + loss_dur

            if torch.isnan(loss_total) or torch.isinf(loss_total):
                print(f"[Warning] Step {global_step} skipped due to NaN/Inf in loss_total!", flush=True)
                optim_g.zero_grad()
                continue

            loss_total.backward()
            norm_g = torch.nn.utils.clip_grad_norm_(net_g.parameters(), 1.0)
            if torch.isnan(norm_g) or torch.isinf(norm_g):
                optim_g.zero_grad()
            else:
                optim_g.step()

            global_step += 1

            # Progress log every 25 steps
            if global_step % 25 == 0 or global_step == 1:
                elapsed = time.time() - start_time
                steps_per_sec = global_step / max(elapsed, 1e-4)
                print(
                    f"Step {global_step:4d}/{max_steps} | "
                    f"Loss Total: {loss_total.item():.3f} | "
                    f"Mel: {loss_mel.item():.3f} | "
                    f"Distill: {loss_distill_prior.item():.3f} | "
                    f"Gen: {loss_gen.item():.3f} | "
                    f"KL: {loss_kl_val.item():.3f} | "
                    f"Speed: {steps_per_sec:.2f} step/s",
                    flush=True,
                )

            # Audition synthesis
            if global_step % eval_interval == 0:
                evaluate_audition(net_g, table, global_step, ckpt_dir, device)

            # Checkpoint save
            if global_step % save_interval == 0 or global_step == max_steps:
                ckpt_g = ckpt_dir / f"G_small_{global_step}.pth"
                ckpt_d = ckpt_dir / f"D_small_{global_step}.pth"
                torch.save(net_g.state_dict(), ckpt_g)
                torch.save(net_d.state_dict(), ckpt_d)
                print(f"--> [Checkpoint] Saved: {ckpt_g.name} & {ckpt_d.name}", flush=True)

                all_g = sorted(
                    [p for p in ckpt_dir.glob("G_small_*.pth") if p.stem.split("_")[-1].isdigit()],
                    key=lambda p: int(p.stem.split("_")[-1]),
                )
                keep_steps = {500, 1000, 1500, 2000, max_steps}
                if len(all_g) > 2:
                    for old_p in all_g[:-2]:
                        s = int(old_p.stem.split("_")[-1])
                        if s not in keep_steps:
                            old_d = ckpt_dir / f"D_small_{s}.pth"
                            old_p.unlink(missing_ok=True)
                            old_d.unlink(missing_ok=True)

                torch.cuda.empty_cache()

            if global_step >= max_steps:
                break

    total_min = (time.time() - start_time) / 60
    print(f"\n🎉 ArboTTS Small Tier Distillation completed in {total_min:.2f} minutes!", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ArboTTS Small Tier Knowledge Distillation Training")
    parser.add_argument("--steps", type=int, default=1000, help="Max distillation steps")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--meta", type=str, default="dataset/train_multispeaker.txt", help="Metadata path")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints_vits_small", help="Checkpoint directory")
    parser.add_argument("--eval-interval", type=int, default=200, help="Evaluation audition interval")
    parser.add_argument("--save-interval", type=int, default=500, help="Checkpoint save interval")
    args = parser.parse_args()

    train_distill(
        meta_txt=args.meta,
        ckpt_dir_name=args.ckpt_dir,
        max_steps=args.steps,
        batch_size=args.batch,
        lr=args.lr,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
    )
