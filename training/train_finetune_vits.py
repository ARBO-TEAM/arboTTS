"""Fine-tuning / Transfer Learning of Pretrained Multi-Speaker VITS on Arbo & Arba Voice Corpus.

Target: Broadcast announcer quality using pretrained Indonesian VITS base.
Hardware: NVIDIA RTX 5060 Ti 16GB GPU with FP16/BF16 AMP.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time
from pathlib import Path

# Windows PosixPath unpickle fix
pathlib.PosixPath = pathlib.WindowsPath

REPO_ROOT = Path("D:/kerjaan/belajar/tts/arboTTS").resolve()
PIPER_DIR = Path("D:/kerjaan/belajar/tts/piper/src/python").resolve()
SANOTTS_DIR = Path("D:/kerjaan/belajar/tts/sanoTTS/pypkg").resolve()

for p in [REPO_ROOT, PIPER_DIR, SANOTTS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import soundfile as sf
import torch
from torch import nn, optim
from torch.amp import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from piper_train.vits.commons import rand_slice_segments, slice_segments
from piper_train.vits.losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
)
from piper_train.vits.mel_processing import (
    mel_spectrogram_torch,
    spec_to_mel_torch,
    spectrogram_torch,
)
from piper_train.vits.models import MultiPeriodDiscriminator, SynthesizerTrn
from sanotts import frontend


def safe_kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """Numerically stable KL divergence loss preventing exp overflow."""
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = torch.clamp(logs_p.float(), min=-10.0, max=10.0)
    z_mask = z_mask.float()
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    return kl / torch.clamp(torch.sum(z_mask), min=1.0)


class ArboFTDataset(Dataset):
    """Loads audio waveforms, phonemes, and speaker IDs for fine-tuning."""

    def __init__(
        self,
        metadata_path: str | Path,
        table: frontend.PhonemeTable,
        filter_length: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        sampling_rate: int = 22050,
    ) -> None:
        self.table = table
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate

        self.items: list[tuple[str, str, int]] = []
        meta_p = REPO_ROOT / metadata_path if not Path(metadata_path).is_absolute() else Path(metadata_path)
        with open(meta_p, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 3:
                    wav_rel, spk_str, text = parts[0], parts[1], parts[2]
                    wav_full = REPO_ROOT / wav_rel
                    if wav_full.is_file():
                        self.items.append((str(wav_full), text, int(spk_str)))
                elif len(parts) == 2:
                    wav_rel, text = parts[0], parts[1]
                    wav_full = REPO_ROOT / wav_rel
                    if wav_full.is_file():
                        self.items.append((str(wav_full), text, 0))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        wav_path, text, spk_id = self.items[index]

        # 1. Text to phoneme IDs using pretrained table
        phoneme_ids = frontend.text_to_phoneme_ids(text, self.table)
        text_tensor = torch.tensor(phoneme_ids, dtype=torch.long)

        # 2. Audio loading
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)

        # 3. Linear Spectrogram
        spec = spectrogram_torch(
            audio_tensor,
            n_fft=self.filter_length,
            sampling_rate=self.sampling_rate,
            hop_size=self.hop_length,
            win_size=self.win_length,
            center=False,
        ).squeeze(0)

        return text_tensor, spec, audio_tensor.squeeze(0), spk_id


class ArboFTCollate:
    """Collate batch with dynamic padding including speaker IDs."""

    def __call__(self, batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]) -> tuple[torch.Tensor, ...]:
        batch = [b for b in batch if b[0].size(0) > 0 and b[1].size(1) > 0]
        if not batch:
            return ()

        # Sort descending by spec length
        batch.sort(key=lambda x: x[1].size(1), reverse=True)

        max_text_len = max([x[0].size(0) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(0) for x in batch])

        b_size = len(batch)
        text_padded = torch.zeros(b_size, max_text_len, dtype=torch.long)
        spec_padded = torch.zeros(b_size, batch[0][1].size(0), max_spec_len, dtype=torch.float32)
        wav_padded = torch.zeros(b_size, max_wav_len, dtype=torch.float32)

        text_lengths = torch.zeros(b_size, dtype=torch.long)
        spec_lengths = torch.zeros(b_size, dtype=torch.long)
        wav_lengths = torch.zeros(b_size, dtype=torch.long)
        speaker_ids = torch.zeros(b_size, dtype=torch.long)

        for i, (text, spec, wav, spk_id) in enumerate(batch):
            text_len = text.size(0)
            spec_len = spec.size(1)
            wav_len = wav.size(0)

            text_padded[i, :text_len] = text
            text_lengths[i] = text_len

            spec_padded[i, :, :spec_len] = spec
            spec_lengths[i] = spec_len

            wav_padded[i, :wav_len] = wav
            wav_lengths[i] = wav_len

            speaker_ids[i] = spk_id

        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, speaker_ids


def evaluate_audition(
    model: SynthesizerTrn,
    table: frontend.PhonemeTable,
    step: int,
    out_dir: Path,
    device: torch.device,
) -> None:
    """Synthesize test audition for both Arbo and Arba and report acoustic metrics."""
    test_text = "Halo, selamat datang di ArboTTS Studio. Suara ini dihasilkan oleh kecerdasan buatan."
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

            out_file = out_dir / f"audition_step_{step:04d}_{spk_name}.wav"
            sf.write(str(out_file), audio, 22050, subtype="PCM_16")

            duration = len(audio) / 22050
            rms = float(np.sqrt(np.mean(audio**2)))
            print(
                f"--> [Audition Step {step} | {spk_name.upper()}] Saved {out_file.name} | "
                f"Dur: {duration:.2f}s | RMS: {rms:.4f}",
                flush=True,
            )

    model.train()


def train_finetune(
    base_ckpt: str = "D:/kerjaan/belajar/tts/sanoTTS/experiments/epoch199.ckpt",
    cfg_json: str = "D:/kerjaan/belajar/tts/sanoTTS/experiments/_pilot/teacher.onnx.json",
    meta_txt: str = "dataset/train_multispeaker.txt",
    ckpt_dir_name: str = "checkpoints_vits_multispk",
    n_speakers: int = 2,
    gin_channels: int = 256,
    max_steps: int = 1000,
    batch_size: int = 16,
    lr: float = 2e-5,
    eval_interval: int = 250,
    save_interval: int = 500,
) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting ArboTTS Pretrained Multi-Speaker VITS Transfer Learning on {torch.cuda.get_device_name(0)} ===")

    # 1. Load Pretrained Checkpoint Metadata
    ckpt_path = Path(base_ckpt)
    print(f"Loading base checkpoint: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]

    cfg_path = Path(cfg_json)
    table = frontend.load_phoneme_table(cfg_path)
    print("Loaded Piper/VITS phoneme table.")

    # 2. Build Multi-Speaker Models
    net_g = SynthesizerTrn(
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
        n_speakers=n_speakers,
        gin_channels=gin_channels,
        use_sdp=hp["use_sdp"],
    ).to(device)

    net_d = MultiPeriodDiscriminator(use_spectral_norm=hp["use_spectral_norm"]).to(device)

    # 3. Load Pretrained Weights
    sd = ckpt["state_dict"]
    g_sd = {k.replace("model_g.", ""): v for k, v in sd.items() if k.startswith("model_g.")}
    d_sd = {k.replace("model_d.", ""): v for k, v in sd.items() if k.startswith("model_d.")}

    # Strict=False accommodates newly added multi-speaker conditioning layers
    net_g.load_state_dict(g_sd, strict=False)
    net_d.load_state_dict(d_sd, strict=True)
    print("Successfully transferred base weights to Multi-Speaker VITS architecture!")

    # Freeze Text Encoder to maintain pronunciation stability
    for param in net_g.enc_p.parameters():
        param.requires_grad = False
    print("Frozen Text Encoder — fine-tuning Speaker Embedding, Vocoder (Generator), Flow, and Posterior Encoder.")

    # 4. Check for Resuming Checkpoints
    ckpt_dir = REPO_ROOT / ckpt_dir_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    existing_g = sorted(
        [p for p in ckpt_dir.glob("G_ft_*.pth") if p.stem.split("_")[-1].isdigit()],
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    global_step = 0
    if existing_g:
        last_step = int(existing_g[-1].stem.split("_")[-1])
        if last_step < max_steps:
            last_d = ckpt_dir / f"D_ft_{last_step}.pth"
            if last_d.exists():
                print(f"Resuming multi-speaker fine-tuning from step {last_step}...")
                net_g.load_state_dict(torch.load(existing_g[-1], map_location=device))
                net_d.load_state_dict(torch.load(last_d, map_location=device))
                global_step = last_step

    # 5. Optimizers with fine-tuning learning rate
    trainable_g = [p for p in net_g.parameters() if p.requires_grad]
    optim_g = optim.AdamW(trainable_g, lr=lr, betas=hp["betas"], eps=hp["eps"])
    optim_d = optim.AdamW(net_d.parameters(), lr=lr, betas=hp["betas"], eps=hp["eps"])

    # 6. DataLoader with balanced speaker sampling
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

    print(f"Starting multi-speaker fine-tuning loop: step {global_step} -> {max_steps}...", flush=True)

    while global_step < max_steps:
        epoch += 1
        for batch in loader:
            if not batch:
                continue

            x, x_lengths, spec, spec_lengths, y, y_lengths, sid = [b.to(device) for b in batch]

            # ---------------------
            # Train Discriminator
            # ---------------------
            optim_d.zero_grad()
            with autocast("cuda", dtype=torch.bfloat16):
                (
                    y_hat,
                    l_length,
                    attn,
                    ids_slice,
                    x_mask,
                    y_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = net_g(x, x_lengths, spec, spec_lengths, sid=sid)

                y_slice = slice_segments(y.unsqueeze(1), ids_slice * hop_length, segment_size)
                y_d_rs, y_d_gs, fmap_rs, fmap_gs = net_d(y_slice, y_hat.detach())
                loss_disc, _, _ = discriminator_loss(y_d_rs, y_d_gs)

            if torch.isnan(loss_disc) or torch.isinf(loss_disc):
                print(f"[Warning] Step {global_step} skipped due to NaN/Inf in loss_disc!", flush=True)
                optim_d.zero_grad()
            else:
                loss_disc.backward()
                norm_d = torch.nn.utils.clip_grad_norm_(net_d.parameters(), 1.0)
                if torch.isnan(norm_d) or torch.isinf(norm_d):
                    print(f"[Warning] NaN grad norm in D at step {global_step}, skipping step", flush=True)
                    optim_d.zero_grad()
                else:
                    optim_d.step()

            # ---------------------
            # Train Generator
            # ---------------------
            optim_g.zero_grad()
            with autocast("cuda", dtype=torch.bfloat16):
                y_d_rs, y_d_gs, fmap_rs, fmap_gs = net_d(y_slice, y_hat)
                loss_gen, _ = generator_loss(y_d_gs)
                loss_fm = feature_loss(fmap_rs, fmap_gs)
                loss_dur = torch.sum(l_length.float())

            # Mel & KL losses in float32
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
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1).float(),
                    hp["filter_length"],
                    hp["mel_channels"],
                    hp["sample_rate"],
                    hop_length,
                    hp["win_length"],
                    hp["mel_fmin"],
                    hp["mel_fmax"],
                )
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * c_mel
                loss_kl_val = safe_kl_loss(z_p, logs_q, m_p, logs_p, y_mask) * c_kl
                loss_total = loss_gen + loss_fm + loss_mel + loss_kl_val + loss_dur

            if torch.isnan(loss_total) or torch.isinf(loss_total):
                print(f"[Warning] Step {global_step} skipped due to NaN/Inf in loss_total! KL: {loss_kl_val.item()}", flush=True)
                optim_g.zero_grad()
                continue

            loss_total.backward()
            norm_g = torch.nn.utils.clip_grad_norm_(net_g.parameters(), 1.0)
            if torch.isnan(norm_g) or torch.isinf(norm_g):
                print(f"[Warning] NaN grad norm in G at step {global_step}, skipping step", flush=True)
                optim_g.zero_grad()
            else:
                optim_g.step()

            global_step += 1

            if global_step % 50 == 0:
                elapsed = time.time() - start_time
                steps_per_sec = global_step / max(elapsed, 1e-5)
                print(
                    f"[Step {global_step:4d}/{max_steps}] | "
                    f"Gen: {loss_gen.item():.4f} | "
                    f"Mel: {loss_mel.item():.4f} | "
                    f"FM: {loss_fm.item():.4f} | "
                    f"Dur: {loss_dur.item():.4f} | "
                    f"Disc: {loss_disc.item():.4f} | "
                    f"Speed: {steps_per_sec:.2f} st/s",
                    flush=True,
                )

            # Audition synthesis
            if global_step % eval_interval == 0:
                evaluate_audition(net_g, table, global_step, ckpt_dir, device)

            # Checkpoint save & prune
            if global_step % save_interval == 0 or global_step == max_steps:
                ckpt_g = ckpt_dir / f"G_ft_{global_step}.pth"
                ckpt_d = ckpt_dir / f"D_ft_{global_step}.pth"
                torch.save(net_g.state_dict(), ckpt_g)
                torch.save(net_d.state_dict(), ckpt_d)
                print(f"--> Saved checkpoint: {ckpt_g.name} and {ckpt_d.name}", flush=True)

                all_g = sorted(
                    [p for p in ckpt_dir.glob("G_ft_*.pth") if p.stem.split("_")[-1].isdigit()],
                    key=lambda p: int(p.stem.split("_")[-1]),
                )
                keep_steps = {500, 1000, 1500, 2000, max_steps}
                if len(all_g) > 2:
                    for old_p in all_g[:-2]:
                        s = int(old_p.stem.split("_")[-1])
                        if s not in keep_steps:
                            old_d = ckpt_dir / f"D_ft_{s}.pth"
                            old_p.unlink(missing_ok=True)
                            old_d.unlink(missing_ok=True)

                torch.cuda.empty_cache()

            if global_step >= max_steps:
                break

    total_min = (time.time() - start_time) / 60
    print(f"=== Multi-speaker fine-tuning completed in {total_min:.2f} minutes! ===", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ArboTTS Multi-Speaker VITS Fine-Tuning")
    parser.add_argument("--steps", type=int, default=1000, help="Max fine-tuning steps")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--n-speakers", type=int, default=2, help="Number of speakers")
    parser.add_argument("--gin-channels", type=int, default=256, help="Speaker embedding channels")
    parser.add_argument("--meta", type=str, default="dataset/train_multispeaker.txt", help="Metadata path")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints_vits_multispk", help="Checkpoint directory")
    parser.add_argument("--eval-interval", type=int, default=250, help="Evaluation audition interval")
    parser.add_argument("--save-interval", type=int, default=500, help="Checkpoint save interval")
    args = parser.parse_args()

    train_finetune(
        meta_txt=args.meta,
        ckpt_dir_name=args.ckpt_dir,
        n_speakers=args.n_speakers,
        gin_channels=args.gin_channels,
        max_steps=args.steps,
        batch_size=args.batch,
        lr=args.lr,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
    )
