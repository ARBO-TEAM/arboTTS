"""PyTorch VITS Model Architecture for ArboTTS."""

from __future__ import annotations

import math
import numba
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def sequence_mask(length: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


@numba.jit(nopython=True, fastmath=True)
def _maximum_path_numba(neg_cent: np.ndarray, mask: np.ndarray) -> np.ndarray:
    b, t_t, t_s = neg_cent.shape
    paths = np.zeros((b, t_t, t_s), dtype=np.float32)

    for i in range(b):
        v = np.full((t_t,), -1e9, dtype=np.float32)
        v[0] = neg_cent[i, 0, 0]
        from_prev = np.zeros((t_t, t_s), dtype=np.bool_)

        for t in range(1, t_s):
            v_move = np.full((t_t,), -1e9, dtype=np.float32)
            v_move[1:] = v[:-1]

            for s in range(t_t):
                if v_move[s] > v[s]:
                    from_prev[s, t] = True
                    v[s] = v_move[s] + neg_cent[i, s, t]
                else:
                    v[s] = v[s] + neg_cent[i, s, t]

                if mask[i, s, t] < 0.5:
                    v[s] = -1e9

        curr_s = t_t - 1
        for t in range(t_s - 1, -1, -1):
            paths[i, curr_s, t] = 1.0
            if curr_s > 0 and from_prev[curr_s, t]:
                curr_s -= 1

    return paths


def maximum_path(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fast JIT Monotonic Alignment Search (MAS) for VITS (~15ms per batch)."""
    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cpu = neg_cent.detach().cpu().numpy().astype(np.float32)
    mask_cpu = mask.detach().cpu().numpy().astype(np.float32)
    paths_np = _maximum_path_numba(neg_cpu, mask_cpu)
    return torch.from_numpy(paths_np).to(device=device, dtype=dtype)


def slice_segments(x: torch.Tensor, ids_str: torch.Tensor, segment_size: int = 4) -> torch.Tensor:
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def rand_slice_segments(
    x: torch.Tensor, x_lengths: torch.Tensor | None = None, segment_size: int = 4
) -> tuple[torch.Tensor, torch.Tensor]:
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = torch.clamp(x_lengths - segment_size + 1, min=0)
    ids_str = (torch.rand([b], device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


class TextEncoder(nn.Module):
    """Encodes phoneme token IDs into latent text representations."""

    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        conv_layers = []
        for _ in range(4):
            conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_channels, filter_channels, kernel_size, padding=kernel_size // 2),
                    nn.GELU(),
                    nn.Dropout(p_dropout),
                    nn.Conv1d(filter_channels, hidden_channels, 1),
                )
            )
        self.encoder = nn.ModuleList(conv_layers)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.emb(x) * math.sqrt(self.hidden_channels)
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        for layer in self.encoder:
            x = (x + layer(x * x_mask)) * x_mask

        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class WN(nn.Module):
    """WaveNet-style residual blocks for Posterior Encoder."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int, dilation_rate: int, n_layers: int, p_dropout: float = 0.0) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size, dilation=dilation, padding=padding)
            self.in_layers.append(in_layer)

            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels
            res_skip_layer = nn.Conv1d(hidden_channels, res_skip_channels, 1)
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(x)
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            x_in = self.drop(x_in)
            acts = torch.split(x_in, self.hidden_channels, dim=1)
            x_acts = torch.tanh(acts[0]) * torch.sigmoid(acts[1])
            res_skip_acts = self.res_skip_layers[i](x_acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, :self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts
        return output * x_mask


class PosteriorEncoder(nn.Module):
    """Encodes linear spectrograms into latent space z."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        n_layers: int = 16,
    ) -> None:
        super().__init__()
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(in_channels, hidden_channels, kernel_size, dilation_rate, n_layers)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, stats.size(1) // 2, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: tuple[int, int, int] = (1, 3, 5)) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=self._get_padding(kernel_size, dilation[0])),
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=self._get_padding(kernel_size, dilation[1])),
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding=self._get_padding(kernel_size, dilation[2])),
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=self._get_padding(kernel_size, 1)),
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=self._get_padding(kernel_size, 1)),
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=self._get_padding(kernel_size, 1)),
        ])

    def _get_padding(self, kernel_size: int, dilation: int = 1) -> int:
        return int((kernel_size * dilation - dilation) / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x


class Generator(nn.Module):
    """HiFi-GAN vocoder: converts latent z into 22,050 Hz waveform."""

    def __init__(
        self,
        initial_channel: int = 192,
        resblock_kernel_sizes: list[int] = [3, 7, 11],
        resblock_dilation_sizes: list[list[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates: list[int] = [8, 8, 2, 2],
        upsample_initial_channel: int = 512,
        upsample_kernel_sizes: list[int] = [16, 16, 4, 4],
    ) -> None:
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(
                    upsample_initial_channel // (2 ** i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    k,
                    u,
                    padding=(k - u) // 2,
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock1(ch, k, tuple(d)))

        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x


class DiscriminatorP(nn.Module):
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3) -> None:
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(2, 0)),
            nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(2, 0)),
            nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(2, 0)),
            nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(2, 0)),
            nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0)),
        ])
        self.conv_post = nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: list[int] = [2, 3, 5, 7, 11]) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(p) for p in periods])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor], list[list[torch.Tensor]], list[list[torch.Tensor]]]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class SynthesizerTrn(nn.Module):
    """Full VITS Synthesizer Model."""

    def __init__(
        self,
        n_vocab: int = 128,
        spec_channels: int = 513,
        segment_size: int = 8192,
        inter_channels: int = 192,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        resblock: str = "1",
        resblock_kernel_sizes: list[int] = [3, 7, 11],
        resblock_dilation_sizes: list[list[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates: list[int] = [8, 8, 2, 2],
        upsample_initial_channel: int = 512,
        upsample_kernel_sizes: list[int] = [16, 16, 4, 4],
        n_layers_q: int = 3,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.segment_size = segment_size

        self.enc_p = TextEncoder(
            n_vocab,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        self.dec = Generator(
            inter_channels,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=16,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths)

        # Monotonic Alignment between text prior (m_p, logs_p) and posterior z
        with torch.no_grad():
            # Distance: -0.5 * sum((z - m_p)^2)
            neg_cent = -0.5 * torch.sum(
                (z.unsqueeze(2) - m_p.unsqueeze(3)) ** 2, dim=1
            )  # [b, t_text, t_spec]
            mask = x_mask.unsqueeze(3) * y_mask.unsqueeze(2)
            attn = maximum_path(neg_cent, mask.squeeze(1))  # [b, t_text, t_spec]

        # Align text prior statistics to audio spectrogram frame length
        m_p_aligned = torch.matmul(attn.transpose(1, 2), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p_aligned = torch.matmul(attn.transpose(1, 2), logs_p.transpose(1, 2)).transpose(1, 2)

        z_slice, ids_slice = rand_slice_segments(z, y_lengths, self.segment_size // 256)
        o = self.dec(z_slice)

        return o, ids_slice, x_mask, y_mask, (z, z, m_p_aligned, logs_p_aligned, m_q, logs_q)

    def infer(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        durations: torch.Tensor | None = None,
        length_scale: float = 1.0,
        phoneme_rate: float = 5.8,
    ) -> torch.Tensor:
        """Inference from phoneme IDs directly to audio waveform with duration scaling."""
        _, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        if durations is not None:
            # Discrete token expansion matching training MAS alignment
            dur_adj = torch.clamp(torch.round(durations.float() / length_scale).long(), min=1)
            z = torch.repeat_interleave(m_p, dur_adj, dim=2)
        else:
            scale = phoneme_rate * length_scale
            z = F.interpolate(m_p, scale_factor=scale, mode="linear", align_corners=False)
        o = self.dec(z)
        return o
