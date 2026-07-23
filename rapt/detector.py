"""Hierarchical OOD detection: calibration and the three statistical gates.

Gates (fired independently, decision = OR):

1. **Local (per-dimension)** — ``loss_i > cal_max_i + k_dim * (cal_max_i -
   cal_median_i)``: catches localized deviations in a few channels.
2. **Global (mean)** — ``mean_i(loss_i) > max(cal_mean) + k_global *
   std(cal_mean)``: catches systemic drift.
3. **Range** (optional, "Hybrid") — raw observation outside
   ``[min - b*range, max + b*range]`` from the calibration data.

The continuous ``risk`` score is ``max(local, global, range)`` where each term
is normalized so 1.0 is the calibrated decision boundary — usable both for
thresholded deployment and threshold-free metrics (AUROC).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .config import RaptConfig
from .data import NormStats, SequenceData
from .model import RaptModel, nll_per_dim

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@torch.no_grad()
def score_sequence(
    model: RaptModel,
    stats: NormStats,
    obs: np.ndarray,
    actions: np.ndarray | None = None,
    device: str = "cpu",
) -> np.ndarray:
    """Per-step per-dimension anomaly scores for one sequence.

    Returns ``loss [T, D]`` aligned so ``loss[t]`` scores ``obs[t]``:
    reconstruction scores each step directly; forward dynamics scores
    ``obs[t]`` with the prediction made at ``t-1`` (``loss[0]`` is zero).
    """
    model.eval().to(device)
    if len(obs) < 2 and model.cfg.train_dynamics:
        # 1-step sequences (e.g. real runs where the fault destabilized the
        # robot immediately) have no transition to score.
        return np.zeros((len(obs), model.cfg.obs_dim), dtype=np.float32)
    o = torch.from_numpy(stats.normalize(obs).astype(np.float32)).to(device)
    if model.cfg.train_dynamics:
        if actions is None:
            raise ValueError("Forward-dynamics model requires the action sequence.")
        a = torch.from_numpy(actions.astype(np.float32)).to(device)
        x = torch.cat([o[:-1], a[:-1]], dim=1)
        target = o[1:]
    else:
        x, target = o, o
    mu, log_var, _ = model(x.unsqueeze(0))
    loss = nll_per_dim(mu[0], log_var[0], target).cpu().numpy()
    if model.cfg.train_dynamics:
        loss = np.concatenate([np.zeros((1, loss.shape[1]), dtype=loss.dtype), loss])
    return loss


def score_dataset(
    model: RaptModel,
    stats: NormStats,
    data: SequenceData,
    device: str = "cpu",
) -> list[np.ndarray]:
    return [
        score_sequence(model, stats, data.obs[i], data.actions[i] if data.actions else None, device)
        for i in range(len(data))
    ]


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


@dataclass
class Calibration:
    """Thresholds derived from a nominal calibration batch (sim or real)."""

    per_dim_median: np.ndarray
    per_dim_max: np.ndarray
    per_dim_thresh: np.ndarray
    mean_max: float
    mean_std: float
    mean_thresh: float
    obs_min: np.ndarray  # raw units, from calibration data
    obs_max: np.ndarray
    range_buffer: float = 1.0
    k_dim: float = 2.0
    k_global: float = 3.0
    # Operating-point scale: raw gate risk is divided by this before the
    # risk > 1 decision. Set by target-FPR calibration (the (1 - fpr)
    # quantile of nominal episode risk); 1.0 = use the gate margins as-is.
    risk_scale: float = 1.0
    extras: dict = field(default_factory=dict)

    @property
    def range_lo(self) -> np.ndarray:
        return self.obs_min - self.range_buffer * (self.obs_max - self.obs_min)

    @property
    def range_hi(self) -> np.ndarray:
        return self.obs_max + self.range_buffer * (self.obs_max - self.obs_min)

    def to_dict(self) -> dict:
        d = {
            "per_dim_median": self.per_dim_median.tolist(),
            "per_dim_max": self.per_dim_max.tolist(),
            "per_dim_thresh": self.per_dim_thresh.tolist(),
            "mean_max": self.mean_max,
            "mean_std": self.mean_std,
            "mean_thresh": self.mean_thresh,
            "obs_min": self.obs_min.tolist(),
            "obs_max": self.obs_max.tolist(),
            "range_buffer": self.range_buffer,
            "k_dim": self.k_dim,
            "k_global": self.k_global,
            "risk_scale": self.risk_scale,
        }
        d.update(self.extras)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        arrays = {
            k: np.asarray(d[k], dtype=np.float64)
            for k in ("per_dim_median", "per_dim_max", "per_dim_thresh", "obs_min", "obs_max")
        }
        return cls(
            **arrays,
            mean_max=float(d["mean_max"]),
            mean_std=float(d["mean_std"]),
            mean_thresh=float(d["mean_thresh"]),
            range_buffer=float(d.get("range_buffer", 1.0)),
            k_dim=float(d.get("k_dim", 2.0)),
            k_global=float(d.get("k_global", 3.0)),
            risk_scale=float(d.get("risk_scale", 1.0)),
        )


def calibrate(
    model: RaptModel,
    stats: NormStats,
    nominal: SequenceData,
    cfg: RaptConfig,
    device: str = "cpu",
) -> Calibration:
    """Fit gate thresholds on nominal data.

    Use a large nominal **simulation** batch, or a brief verified **real**
    nominal run (the paper uses 3×1 min) to absorb static deployment offsets
    such as sensor noise, latency, and contact dynamics.
    """
    losses = score_dataset(model, stats, nominal, device)
    flat = np.concatenate(losses, axis=0)  # [sum_T, D]
    per_dim_median = np.median(flat, axis=0)
    per_dim_max = flat.max(axis=0)
    per_dim_thresh = per_dim_max + cfg.k_dim * (per_dim_max - per_dim_median)
    mean_series = flat.mean(axis=1)
    mean_max, mean_std = float(mean_series.max()), float(mean_series.std())
    raw = np.concatenate(nominal.obs, axis=0)
    return Calibration(
        per_dim_median=per_dim_median,
        per_dim_max=per_dim_max,
        per_dim_thresh=per_dim_thresh,
        mean_max=mean_max,
        mean_std=mean_std,
        mean_thresh=mean_max + cfg.k_global * mean_std,
        obs_min=raw.min(axis=0),
        obs_max=raw.max(axis=0),
        range_buffer=cfg.range_buffer,
        k_dim=cfg.k_dim,
        k_global=cfg.k_global,
    )


# ---------------------------------------------------------------------------
# Risk evaluation
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    risk: float  # max of the gate ratios; > 1.0 = anomaly
    is_anomaly: bool
    local_risk: float
    global_risk: float
    range_risk: float
    per_dim_loss: np.ndarray
    top_dim: int  # most anomalous dimension (argmax of loss / per-dim thresh)


def evaluate_step(
    loss_row: np.ndarray,
    obs_row: np.ndarray,
    cal: Calibration,
    use_range: bool = True,
) -> StepResult:
    ratios = loss_row / (cal.per_dim_thresh + _EPS)
    local = float(ratios.max())
    global_ = float(loss_row.mean() / (cal.mean_thresh + _EPS))
    if use_range:
        width = cal.range_buffer * (cal.obs_max - cal.obs_min) + _EPS
        excess = np.maximum(cal.range_lo - obs_row, obs_row - cal.range_hi) / width
        range_ = float(1.0 + excess.max())
    else:
        range_ = 0.0
    risk = max(local, global_, range_) / cal.risk_scale
    return StepResult(
        risk=risk,
        is_anomaly=risk > 1.0,
        local_risk=local,
        global_risk=global_,
        range_risk=range_,
        per_dim_loss=loss_row,
        top_dim=int(np.argmax(ratios)),
    )


def risk_trajectory(
    losses: np.ndarray,
    obs: np.ndarray,
    cal: Calibration,
    use_range: bool = True,
) -> np.ndarray:
    """Vectorized per-step risk score for a whole sequence ``[T]``."""
    local = (losses / (cal.per_dim_thresh + _EPS)).max(axis=1)
    global_ = losses.mean(axis=1) / (cal.mean_thresh + _EPS)
    risk = np.maximum(local, global_)
    if use_range:
        width = cal.range_buffer * (cal.obs_max - cal.obs_min) + _EPS
        excess = np.maximum(cal.range_lo - obs, obs - cal.range_hi) / width
        risk = np.maximum(risk, 1.0 + excess.max(axis=1))
    return risk / cal.risk_scale


class RaptDetector:
    """Streaming detector: feed one observation (and action) per control step.

    Maintains the GRU hidden state and previous inputs internally, mirroring
    the on-robot C++ runtime. ~1.6 ms per step on CPU in the paper's setup.
    """

    def __init__(
        self,
        model: RaptModel,
        stats: NormStats,
        cal: Calibration,
        use_range: bool = True,
        device: str = "cpu",
    ):
        self.model = model.eval().to(device)
        self.stats = stats
        self.cal = cal
        self.use_range = use_range
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.hidden = self.model.init_hidden(1, self.device)
        self._prev: tuple[torch.Tensor, torch.Tensor | None] | None = None

    @torch.no_grad()
    def step(self, obs: np.ndarray, action: np.ndarray | None = None) -> StepResult:
        o = torch.from_numpy(self.stats.normalize(obs).astype(np.float32)).to(self.device)
        a = None
        if self.model.cfg.train_dynamics:
            if action is None:
                raise ValueError("Forward-dynamics model requires an action per step.")
            a = torch.from_numpy(np.asarray(action, dtype=np.float32)).to(self.device)

        if self.model.cfg.train_dynamics:
            if self._prev is None:  # first step: nothing to predict yet
                self._prev = (o, a)
                zeros = np.zeros(self.model.cfg.obs_dim)
                return evaluate_step(zeros, obs, self.cal, self.use_range)
            prev_o, prev_a = self._prev
            x = torch.cat([prev_o, prev_a]).unsqueeze(0)
            target = o
            self._prev = (o, a)
        else:
            x, target = o.unsqueeze(0), o

        mu, log_var, self.hidden = self.model(x, self.hidden)
        loss = nll_per_dim(mu[0], log_var[0], target).cpu().numpy()
        return evaluate_step(loss, obs, self.cal, self.use_range)
