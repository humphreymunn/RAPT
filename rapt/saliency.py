"""Temporal root-cause attribution via Integrated Gradients through time.

When the detector fires at time ``t``, we attribute the final-timestep NLL to
the observation history window ``X_{t-H:t}`` by backpropagating through the
recurrent model (BPTT), producing a spatio-temporal map ``Phi [H, D]`` that
shows *which* dimensions and *when* they drove the anomaly. The baseline is
the mean nominal observation (zero in normalized space); the path integral is
approximated with ``m`` Riemann steps (paper: m=50, H=200 ≈ 4 s @ 50 Hz).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import NormStats
from .model import RaptModel, nll_per_dim

ONSET_THRESHOLD = 1e-5
LOG_EPS = 1e-12


@dataclass
class SaliencyResult:
    attribution: np.ndarray  # [H, D_input] signed IG attribution
    top_idx: np.ndarray  # [K] most salient input dims (by max |Phi| over time)
    top_names: list[str]
    onset_steps: dict[int, int]  # dim -> first step where |Phi| > threshold
    input_names: list[str]

    @property
    def abs_log(self) -> np.ndarray:
        return np.log10(np.maximum(np.abs(self.attribution), LOG_EPS))


def integrated_gradients_bptt(
    model: RaptModel,
    stats: NormStats,
    obs_window: np.ndarray,
    actions_window: np.ndarray | None = None,
    ig_steps: int = 50,
    top_k: int = 10,
    dim_names: list[str] | None = None,
    device: str = "cpu",
) -> SaliencyResult:
    """IG attribution of the final-step NLL over a ``[H, D]`` history window.

    For forward-dynamics models pass the aligned ``[H, A]`` action window;
    attribution then covers the concatenated ``[obs ‖ action]`` input columns.
    """
    model.eval().to(device)
    cfg = model.cfg
    o = torch.from_numpy(stats.normalize(obs_window).astype(np.float32)).to(device)

    if cfg.train_dynamics:
        if actions_window is None:
            raise ValueError("Forward-dynamics model requires the action window.")
        a = torch.from_numpy(actions_window.astype(np.float32)).to(device)
        x = torch.cat([o[:-1], a[:-1]], dim=1)  # [H-1, D+A]
        target = o[-1]
    else:
        x, target = o, o[-1]

    baseline = torch.zeros_like(x)  # mean nominal observation in normalized space
    alphas = torch.linspace(0.0, 1.0, ig_steps, device=device).view(-1, 1, 1)
    interp = (baseline + alphas * (x - baseline)).requires_grad_(True)  # [m, H, D_in]

    was_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False  # cuDNN RNN blocks input-grad in eval mode
    try:
        mu, log_var, _ = model(interp)
        nll = (0.5 * (nll_per_dim(mu[:, -1], log_var[:, -1], target) + log_var[:, -1])).sum()
        grads = torch.autograd.grad(nll, interp)[0]  # [m, H, D_in]
    finally:
        torch.backends.cudnn.enabled = was_cudnn

    phi = ((x - baseline) * grads.mean(dim=0)).detach().cpu().numpy()

    names = list(dim_names) if dim_names else [f"dim_{i}" for i in range(cfg.obs_dim)]
    if cfg.train_dynamics:
        names = names + [f"action_{i}" for i in range(cfg.action_dim)]
    per_dim = np.abs(phi).max(axis=0)
    top_idx = np.argsort(per_dim)[::-1][: min(top_k, len(per_dim))].copy()
    onset = {}
    for i in top_idx:
        active = np.nonzero(np.abs(phi[:, i]) > ONSET_THRESHOLD)[0]
        if len(active):
            onset[int(i)] = int(active[0])
    return SaliencyResult(
        attribution=phi,
        top_idx=top_idx,
        top_names=[names[i] for i in top_idx],
        onset_steps=onset,
        input_names=names,
    )


def plot_saliency_heatmap(
    result: SaliencyResult,
    path: str,
    dt: float = 0.02,
    title: str = "Temporal saliency (log10 |IG|)",
) -> None:
    """Log-scale top-K saliency heatmap with onset markers, saved to ``path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phi = result.abs_log[:, result.top_idx].T  # [K, H]
    h = phi.shape[1]
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(result.top_idx) + 2))
    im = ax.imshow(
        phi,
        aspect="auto",
        cmap="inferno",
        extent=(-h * dt, 0.0, len(result.top_idx) - 0.5, -0.5),
        vmin=np.log10(LOG_EPS),
    )
    for row, dim in enumerate(result.top_idx):
        step = result.onset_steps.get(int(dim))
        if step is not None:
            ax.axvline((step - h) * dt, color="cyan", lw=0.8, alpha=0.7)
    ax.set_yticks(range(len(result.top_idx)))
    ax.set_yticklabels(result.top_names, fontsize=8)
    ax.set_xlabel("Time before detection (s)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="log10 |attribution|")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
