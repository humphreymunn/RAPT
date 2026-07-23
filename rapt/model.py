"""RAPT network: residual state encoder → GRU temporal bridge → latent
bottleneck → probabilistic decoder.

Single source of truth for the architecture used in training, evaluation,
saliency, and ONNX export. Matches the paper's final configuration:
``embed_dim=256``, 4 residual blocks each side, 25% latent compression,
diagonal-Gaussian head ``[mu | log_var]`` trained with per-dimension NLL.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import RaptConfig

LOG_VAR_CLAMP = 6.0


class ResidualBlock(nn.Module):
    """Linear(d, 2d') → LayerNorm → ReLU → Dropout → Linear(2d', d') →
    LayerNorm → Dropout, with a ReLU residual connection when shapes match."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float, use_residual: bool):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 2 * out_dim),
            nn.LayerNorm(2 * out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
        )
        self.residual = use_residual and in_dim == out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.residual:
            out = torch.relu(x + out)
        return out


class RaptModel(nn.Module):
    """Probabilistic recurrent trajectory model.

    Input is ``o_t`` (reconstruction mode) or ``[o_t, a_t]`` (forward-dynamics
    mode); output is a diagonal Gaussian over the target observation
    (``o_t`` or ``o_{t+1}`` respectively), flattened as ``[mu | log_var]``.

    ``forward`` accepts ``[B, D]`` (stateless single step), ``[B, T, D]``
    (sequence), and an optional GRU ``hidden`` state ``[1, B, embed_dim]``
    for streaming inference.
    """

    def __init__(self, cfg: RaptConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.embed_dim

        self.encoder_in = nn.Sequential(nn.Linear(cfg.input_dim, d), nn.ReLU())
        self.encoder_blocks = nn.Sequential(
            *[ResidualBlock(d, d, cfg.dropout, cfg.use_residual) for _ in range(cfg.num_blocks)]
        )
        self.gru = nn.GRU(d, d, num_layers=1, batch_first=True) if cfg.use_temporal else None
        if cfg.reconstruction_type == "bottleneck":
            b = cfg.bottleneck_dim
            self.compress = nn.Sequential(nn.Linear(d, b), nn.LayerNorm(b), nn.ReLU())
            self.decompress = nn.Sequential(nn.Linear(b, d), nn.ReLU())
        else:  # "masked": denoise via input dropout instead of a bottleneck
            self.compress = self.decompress = None
        self.decoder_blocks = nn.Sequential(
            *[ResidualBlock(d, d, cfg.dropout, cfg.use_residual) for _ in range(cfg.num_blocks)]
        )
        out_mult = 2 if cfg.use_probabilistic else 1
        self.head = nn.Linear(d, cfg.obs_dim * out_mult)

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor | None = None,
        mask: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Returns ``(mu, log_var, hidden)``.

        ``mask=True`` applies input dropout at rate ``compression_ratio``
        (training-time only, "masked" reconstruction type).
        """
        squeeze_time = x.dim() == 2
        if squeeze_time:
            x = x.unsqueeze(1)  # [B, 1, D]

        if mask and self.cfg.reconstruction_type == "masked":
            keep = (torch.rand_like(x) > self.cfg.compression_ratio).float()
            x = x * keep

        h = self.encoder_blocks(self.encoder_in(x))
        if self.gru is not None:
            h, hidden = self.gru(h, hidden)
        if self.compress is not None:
            h = self.decompress(self.compress(h))
        out = self.head(self.decoder_blocks(h))

        if self.cfg.use_probabilistic:
            mu, log_var = out.chunk(2, dim=-1)
            log_var = torch.clamp(log_var, -LOG_VAR_CLAMP, LOG_VAR_CLAMP)
        else:
            mu, log_var = out, torch.zeros_like(out)

        if squeeze_time:
            mu, log_var = mu.squeeze(1), log_var.squeeze(1)
        return mu, log_var, hidden

    def init_hidden(self, batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.zeros(1, batch_size, self.cfg.embed_dim, device=device)


def nll_per_dim(mu: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-dimension anomaly score ``(target - mu)^2 / var`` (a.k.a.
    ``mse_over_var``) — the quantity thresholded by the detector gates."""
    return torch.exp(-log_var) * (target - mu) ** 2


def nll_loss(mu: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Diagonal-Gaussian negative log-likelihood (constant dropped):
    ``0.5 * (precision * mse + log_var)`` averaged over all elements."""
    return (0.5 * (nll_per_dim(mu, log_var, target) + log_var)).mean()
