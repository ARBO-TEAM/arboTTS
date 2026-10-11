"""Loss functions for VITS training."""

from __future__ import annotations

import torch


def feature_loss(fmap_r: list[list[torch.Tensor]], fmap_g: list[list[torch.Tensor]]) -> torch.Tensor:
    """Feature matching loss between discriminator representations of real and generated audio."""
    loss = 0.0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2.0


def discriminator_loss(
    disc_real_outputs: list[torch.Tensor],
    disc_generated_outputs: list[torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """Least-squares adversarial loss for Multi-Period / Multi-Scale Discriminator."""
    loss = 0.0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1.0 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += r_loss + g_loss
        r_losses.append(r_loss)
        g_losses.append(g_loss)
    return loss, r_losses, g_losses


def generator_loss(disc_outputs: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Least-squares adversarial loss for VITS Generator."""
    loss = 0.0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1.0 - dg) ** 2)
        loss += l
        gen_losses.append(l)
    return loss, gen_losses


def kl_loss(
    z_p: torch.Tensor,
    logs_q: torch.Tensor,
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    z_mask: torch.Tensor,
) -> torch.Tensor:
    """KL divergence loss between posterior encoder distribution and text prior distribution."""
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask) / (torch.sum(z_mask) + 1e-8)
    return kl
