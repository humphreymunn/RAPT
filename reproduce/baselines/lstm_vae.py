"""LSTM-VAE baseline (Park et al., 2018) with SVR-based expected-NLL scoring.

Paper configuration: hidden 256, latent 24, per-step diagonal-Gaussian
reconstruction NLL; an RBF SVR fit on nominal data predicts the *expected*
NLL for a state, and the anomaly score is the NLL excess over that
expectation (downsampled to <=20k samples for the SVR fit).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVR


class LSTMVAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, latent_dim: int = 24):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_z_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_z_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc_dec_in = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc_x_mu = nn.Linear(hidden_dim, input_dim)
        self.fc_x_var = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor):
        h, _ = self.encoder(x)
        z_mu, z_logvar = self.fc_z_mu(h), self.fc_z_logvar(h)
        z = z_mu + torch.randn_like(z_mu) * torch.exp(0.5 * z_logvar)
        d, _ = self.decoder(self.fc_dec_in(z))
        x_mu = torch.sigmoid(self.fc_x_mu(d))
        x_var = torch.nn.functional.softplus(self.fc_x_var(d)) + 1e-4
        return x_mu, x_var, z_mu, z_logvar

    def nll(self, x: torch.Tensor, x_mu: torch.Tensor, x_var: torch.Tensor) -> torch.Tensor:
        return 0.5 * (torch.log(x_var) + (x - x_mu) ** 2 / x_var).sum(dim=-1)


class LSTMVAEDetector:
    """Trains on [0,1]-normalized nominal windows; scores per-step NLL excess."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, latent_dim: int = 24,
                 seq_len: int = 50, device: str = "cpu"):
        self.model = LSTMVAE(input_dim, hidden_dim, latent_dim).to(device)
        self.seq_len = seq_len
        self.device = device
        self.svr: SVR | None = None
        self.lo = self.range = None

    def _norm(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.lo) / self.range, 0.0, 1.0)

    def fit(self, sequences: list[np.ndarray], epochs: int = 100, batch_size: int = 256,
            lr: float = 1e-3, beta: float = 0.1, log_fn=print) -> None:
        flat = np.concatenate(sequences, 0)
        self.lo = flat.min(0)
        self.range = flat.max(0) - self.lo
        self.range[self.range < 1e-6] = 1e-6
        windows = []
        for s in sequences:
            s = self._norm(s)
            n = len(s) // self.seq_len
            if n:
                windows.append(s[: n * self.seq_len].reshape(n, self.seq_len, -1))
        x = torch.from_numpy(np.concatenate(windows).astype(np.float32))
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x), batch_size=batch_size, shuffle=True
        )
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        for ep in range(epochs):
            losses = []
            for (xb,) in loader:
                xb = xb.to(self.device)
                x_mu, x_var, z_mu, z_logvar = self.model(xb)
                recon = self.model.nll(xb, x_mu, x_var).mean()
                kl = (-0.5 * (1 + z_logvar - z_mu**2 - z_logvar.exp()).sum(-1)).mean()
                loss = recon + beta * kl
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if (ep + 1) % 10 == 0:
                log_fn(f"  LSTM-VAE epoch {ep + 1}/{epochs} loss {np.mean(losses):.4f}")

        # SVR: expected NLL as a function of the observation (Park et al.)
        obs, nll = [], []
        for s in sequences:
            nll.append(self._step_nll(s))
            obs.append(self._norm(s))
        obs, nll = np.concatenate(obs), np.concatenate(nll)
        if len(obs) > 20000:
            idx = np.random.default_rng(0).choice(len(obs), 20000, replace=False)
            obs, nll = obs[idx], nll[idx]
        self.svr = SVR(kernel="rbf")
        self.svr.fit(obs, nll)

    @torch.no_grad()
    def _step_nll(self, seq: np.ndarray) -> np.ndarray:
        self.model.eval()
        x = torch.from_numpy(self._norm(seq).astype(np.float32)).unsqueeze(0).to(self.device)
        x_mu, x_var, _, _ = self.model(x)
        return self.model.nll(x, x_mu, x_var)[0].cpu().numpy()

    def score(self, seq: np.ndarray) -> np.ndarray:
        """Per-step anomaly score: NLL minus SVR-predicted expected NLL."""
        nll = self._step_nll(seq)
        if self.svr is not None:
            nll = nll - self.svr.predict(self._norm(seq))
        return nll
