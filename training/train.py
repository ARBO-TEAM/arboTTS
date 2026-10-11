"""Main CUDA Training Script for ArboTTS Pilot on NVIDIA RTX 5060 Ti."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch import optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.data_utils import TextAudioLoader, TextAudioCollate
from training.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from training.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from training.models import MultiPeriodDiscriminator, SynthesizerTrn, slice_segments


def train_pilot(
    config_path: str = "config/config.json",
    max_steps: int = 1500,
    model_name: str = "arbo_pilot",
) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting ArboTTS Pilot Training on {torch.cuda.get_device_name(0)} ===")

    with open(REPO_ROOT / config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    # 1. Dataset & DataLoader
    train_dataset = TextAudioLoader(data_cfg["training_files"], root_dir=str(REPO_ROOT))
    collate_fn = TextAudioCollate()
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=0,  # 0 for reliable Windows multi-threading
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    print(f"Loaded {len(train_dataset)} training clips. Batch size: {train_cfg['batch_size']}")

    # 2. Models
    net_g = SynthesizerTrn(
        n_vocab=128,
        spec_channels=513,
        segment_size=train_cfg["segment_size"],
        **model_cfg,
    ).to(device)

    net_d = MultiPeriodDiscriminator().to(device)

    # 3. Optimizers
    optim_g = optim.AdamW(
        net_g.parameters(),
        lr=train_cfg["learning_rate"],
        betas=train_cfg["betas"],
        eps=train_cfg["eps"],
    )
    optim_d = optim.AdamW(
        net_d.parameters(),
        lr=train_cfg["learning_rate"],
        betas=train_cfg["betas"],
        eps=train_cfg["eps"],
    )

    scaler = GradScaler("cuda", enabled=train_cfg.get("fp16_run", True))

    checkpoint_dir = REPO_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    start_time = time.time()
    epoch = 0

    net_g.train()
    net_d.train()

    print(f"Training loop initialized. Target: {max_steps} steps. Checkpoints dir: {checkpoint_dir}")

    while global_step < max_steps:
        epoch += 1
        for batch_idx, batch in enumerate(train_loader):
            if not batch:
                continue

            x, x_lengths, spec, spec_lengths, y, y_lengths = [b.to(device) for b in batch]

            # ---------------------
            # Train Discriminator
            # ---------------------
            optim_d.zero_grad()
            with autocast("cuda", enabled=train_cfg.get("fp16_run", True)):
                y_hat, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(
                    x, x_lengths, spec, spec_lengths
                )

                # Slice real audio to match generated audio slice
                y_mel = slice_segments(
                    spec_to_mel_torch(
                        spec,
                        data_cfg["filter_length"],
                        data_cfg["n_mel_channels"],
                        data_cfg["sampling_rate"],
                        data_cfg["mel_fmin"],
                        data_cfg["mel_fmax"],
                    ),
                    ids_slice,
                    train_cfg["segment_size"] // data_cfg["hop_length"],
                )
                y_slice = slice_segments(
                    y.unsqueeze(1),
                    ids_slice * data_cfg["hop_length"],
                    train_cfg["segment_size"],
                )

                y_d_rs, y_d_gs, fmap_rs, fmap_gs = net_d(y_slice, y_hat.detach())
                loss_disc, _, _ = discriminator_loss(y_d_rs, y_d_gs)

            scaler.scale(loss_disc).backward()
            scaler.step(optim_d)

            # ---------------------
            # Train Generator
            # ---------------------
            optim_g.zero_grad()
            with autocast("cuda", enabled=train_cfg.get("fp16_run", True)):
                y_d_rs, y_d_gs, fmap_rs, fmap_gs = net_d(y_slice, y_hat)
                loss_gen, _ = generator_loss(y_d_gs)
                loss_fm = feature_loss(fmap_rs, fmap_gs)

                # Mel loss
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1),
                    data_cfg["filter_length"],
                    data_cfg["n_mel_channels"],
                    data_cfg["sampling_rate"],
                    data_cfg["hop_length"],
                    data_cfg["win_length"],
                    data_cfg["mel_fmin"],
                    data_cfg["mel_fmax"],
                )
                loss_mel = torch.nn.functional.l1_loss(y_mel, y_hat_mel) * train_cfg["c_mel"]

                # KL divergence loss
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, y_mask) * train_cfg["c_kl"]

                loss_tot = loss_gen + loss_fm + loss_mel + loss_kl

            scaler.scale(loss_tot).backward()
            scaler.step(optim_g)
            scaler.update()

            global_step += 1

            # Logging
            if global_step % train_cfg["log_interval"] == 0:
                elapsed = time.time() - start_time
                steps_per_sec = global_step / elapsed
                print(
                    f"[Step {global_step:4d}/{max_steps}] | "
                    f"Loss Gen: {loss_gen.item():.4f} | "
                    f"Mel: {loss_mel.item():.4f} | "
                    f"FM: {loss_fm.item():.4f} | "
                    f"Disc: {loss_disc.item():.4f} | "
                    f"Speed: {steps_per_sec:.2f} steps/s"
                )

            # Save Checkpoint
            if global_step % train_cfg["eval_interval"] == 0 or global_step == max_steps:
                ckpt_path_g = checkpoint_dir / f"G_{global_step}.pth"
                ckpt_path_d = checkpoint_dir / f"D_{global_step}.pth"
                torch.save(net_g.state_dict(), ckpt_path_g)
                torch.save(net_d.state_dict(), ckpt_path_d)
                print(f"--> Saved checkpoint at step {global_step}: {ckpt_path_g.name}")

            if global_step >= max_steps:
                break

    total_time = (time.time() - start_time) / 60
    print(f"=== Pilot Training Finished! Total time: {total_time:.2f} minutes ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config/config.json")
    parser.add_argument("-s", "--steps", type=int, default=1500)
    parser.add_argument("-m", "--model", type=str, default="arbo_pilot")
    args = parser.parse_args()

    train_pilot(config_path=args.config, max_steps=args.steps, model_name=args.model)
