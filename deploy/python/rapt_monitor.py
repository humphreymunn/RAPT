"""Dependency-light runtime monitor: onnxruntime + numpy only (no torch).

Drop this file into your control stack, point it at a RAPT checkpoint
directory, and call ``step(obs, action)`` once per control tick. Designed for
the 50 Hz loop of the paper (~1.6 ms/step on CPU); call ``reset()`` at
episode boundaries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

_EPS = 1e-8


@dataclass
class MonitorResult:
    risk: float  # >1.0 = OOD
    is_anomaly: bool
    local_risk: float  # per-dimension gate ratio (localized spikes)
    global_risk: float  # mean-NLL gate ratio (systemic drift)
    range_risk: float  # physical range gate ratio
    per_dim_loss: np.ndarray
    top_dim: int


class RaptMonitor:
    def __init__(self, checkpoint_dir: str | Path, use_range: bool = True, threads: int = 1):
        ckpt = Path(checkpoint_dir)
        cfg = json.loads((ckpt / "config.json").read_text())
        self.obs_dim = cfg["obs_dim"]
        self.action_dim = cfg["action_dim"]
        self.embed_dim = cfg["embed_dim"]
        self.dynamics = cfg["train_dynamics"]
        self.dim_names = cfg.get("dim_names") or [f"dim_{i}" for i in range(self.obs_dim)]

        stats = json.loads((ckpt / "obs_stats.json").read_text())
        self.mean = np.asarray(stats["mean"], dtype=np.float32)
        self.std = np.asarray(stats["std"], dtype=np.float32)

        cal = json.loads((ckpt / "calibration.json").read_text())
        self.per_dim_thresh = np.asarray(cal["per_dim_thresh"], dtype=np.float64)
        self.mean_thresh = float(cal["mean_thresh"])
        self.risk_scale = float(cal.get("risk_scale", 1.0))
        obs_min = np.asarray(cal["obs_min"], dtype=np.float64)
        obs_max = np.asarray(cal["obs_max"], dtype=np.float64)
        buffer = float(cal.get("range_buffer", 1.0))
        span = obs_max - obs_min
        self.range_lo = obs_min - buffer * span
        self.range_hi = obs_max + buffer * span
        self.range_width = buffer * span + _EPS
        self.use_range = use_range

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        self.session = ort.InferenceSession(
            str(ckpt / "rapt.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self.reset()

    def reset(self) -> None:
        self.hidden = np.zeros((1, 1, self.embed_dim), dtype=np.float32)
        self._prev: tuple[np.ndarray, np.ndarray] | None = None

    def step(self, obs: np.ndarray, action: np.ndarray | None = None) -> MonitorResult:
        obs = np.asarray(obs, dtype=np.float32)
        o_norm = (obs - self.mean) / self.std

        if self.dynamics:
            if action is None:
                raise ValueError("Forward-dynamics checkpoint requires an action per step.")
            if self._prev is None:
                self._prev = (o_norm, np.asarray(action, dtype=np.float32))
                return self._evaluate(np.zeros(self.obs_dim), obs)
            prev_o, prev_a = self._prev
            x = np.concatenate([prev_o, prev_a])[None]
            target = o_norm
            self._prev = (o_norm, np.asarray(action, dtype=np.float32))
        else:
            x, target = o_norm[None], o_norm

        out, self.hidden = self.session.run(
            None, {"input": x, "hidden_in": self.hidden}
        )
        mu, log_var = out[0, : self.obs_dim], out[0, self.obs_dim :]
        loss = np.exp(-log_var) * (target - mu) ** 2
        return self._evaluate(loss.astype(np.float64), obs)

    def _evaluate(self, loss: np.ndarray, obs: np.ndarray) -> MonitorResult:
        ratios = loss / (self.per_dim_thresh + _EPS)
        local = float(ratios.max())
        global_ = float(loss.mean() / (self.mean_thresh + _EPS))
        range_ = 0.0
        if self.use_range:
            excess = np.maximum(self.range_lo - obs, obs - self.range_hi) / self.range_width
            range_ = float(1.0 + excess.max())
        risk = max(local, global_, range_) / self.risk_scale
        return MonitorResult(
            risk=risk,
            is_anomaly=risk > 1.0,
            local_risk=local,
            global_risk=global_,
            range_risk=range_,
            per_dim_loss=loss,
            top_dim=int(np.argmax(ratios)),
        )
