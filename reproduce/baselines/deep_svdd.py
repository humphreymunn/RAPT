"""Deep SVDD baseline (Ruff et al., 2018).

Paper configuration: 2-hidden-layer MLP (hidden 128, latent 32, bias-free
output), center initialized from the initial embeddings, one-class objective;
100 epochs, batch 512, lr 1e-3. Score = squared distance to the center.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class SVDDNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepSVDDDetector:
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 32,
                 device: str = "cpu"):
        self.model = SVDDNetwork(input_dim, hidden_dim, latent_dim).to(device)
        self.device = device
        self.center: torch.Tensor | None = None
        self.mu = self.std = None

    def _norm(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu) / self.std

    def fit(self, sequences: list[np.ndarray], epochs: int = 100, batch_size: int = 512,
            lr: float = 1e-3, log_fn=print) -> None:
        flat = np.concatenate(sequences, 0)
        self.mu, self.std = flat.mean(0), flat.std(0)
        self.std[self.std < 1e-6] = 1e-6
        x = torch.from_numpy(self._norm(flat).astype(np.float32))
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x), batch_size=batch_size, shuffle=True
        )
        with torch.no_grad():  # center = mean initial embedding, off-zero clamped
            self.model.eval()
            c = torch.cat([self.model(xb.to(self.device)) for (xb,) in loader]).mean(0)
            c[(c.abs() < 0.1)] = 0.1
            self.center = c
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-6)
        self.model.train()
        for ep in range(epochs):
            losses = []
            for (xb,) in loader:
                z = self.model(xb.to(self.device))
                loss = ((z - self.center) ** 2).sum(-1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if (ep + 1) % 10 == 0:
                log_fn(f"  Deep SVDD epoch {ep + 1}/{epochs} dist {np.mean(losses):.4f}")

    @torch.no_grad()
    def score(self, seq: np.ndarray) -> np.ndarray:
        self.model.eval()
        z = self.model(torch.from_numpy(self._norm(seq).astype(np.float32)).to(self.device))
        return ((z - self.center) ** 2).sum(-1).cpu().numpy()
