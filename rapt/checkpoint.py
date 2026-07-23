"""Checkpoint format: one directory bundling everything a RAPT system needs.

```
checkpoint_dir/
├── config.json         # RaptConfig (architecture, training, gates, dims)
├── model.pt            # PyTorch state_dict
├── rapt.onnx           # ONNX export (input/hidden_in → reconstruction/hidden_out)
├── obs_stats.json      # per-dim mean/std/min/max + dimension names
├── calibration.json    # gate thresholds (written by calibrate)
├── calibration_losses.csv  # per-dim nominal losses (C++ runtime compatible)
└── history.json        # training curves
```
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .config import RaptConfig
from .data import NormStats, SequenceData
from .detector import Calibration, RaptDetector, calibrate, score_sequence
from .model import RaptModel
from .saliency import SaliencyResult, integrated_gradients_bptt


class RaptSystem:
    """Model + normalization + calibration, loadable as one unit."""

    def __init__(
        self,
        cfg: RaptConfig,
        model: RaptModel,
        stats: NormStats,
        cal: Calibration | None = None,
        history: dict | None = None,
    ):
        self.cfg = cfg
        self.model = model
        self.stats = stats
        self.cal = cal
        self.history = history or {}

    # -- persistence --------------------------------------------------------

    def save(self, ckpt_dir: str | Path, export_onnx: bool = True) -> Path:
        ckpt = Path(ckpt_dir)
        ckpt.mkdir(parents=True, exist_ok=True)
        self.cfg.to_json(ckpt / "config.json")
        torch.save(self.model.state_dict(), ckpt / "model.pt")
        obs_stats = self.stats.to_dict()
        obs_stats["dim_names"] = self.cfg.dim_names
        (ckpt / "obs_stats.json").write_text(json.dumps(obs_stats, indent=2))
        if self.cal is not None:
            (ckpt / "calibration.json").write_text(json.dumps(self.cal.to_dict(), indent=2))
        if self.history:
            (ckpt / "history.json").write_text(json.dumps(self.history, indent=2))
        if export_onnx:
            from .onnx_export import export_onnx as _export

            _export(self.model, ckpt / "rapt.onnx")
        self._write_deploy_bundle(ckpt)
        return ckpt

    def _write_deploy_bundle(self, ckpt: Path) -> None:
        """Flat CSV twins of the JSON files for dependency-free C++ loading."""
        cfg = self.cfg
        lines = [
            f"obs_dim,{cfg.obs_dim}",
            f"action_dim,{cfg.action_dim}",
            f"embed_dim,{cfg.embed_dim}",
            f"train_dynamics,{int(cfg.train_dynamics)}",
            f"use_range_gate,{int(cfg.use_range_gate)}",
            f"dt,{cfg.dt}",
        ]
        (ckpt / "deploy_config.csv").write_text("\n".join(lines) + "\n")
        rows = []
        for key in ("mean", "std", "min", "max"):
            vals = ",".join(f"{v:.8g}" for v in getattr(self.stats, key))
            rows.append(f"{key},{vals}")
        (ckpt / "obs_stats.csv").write_text("\n".join(rows) + "\n")
        if self.cal is not None:
            c = self.cal
            rows = [
                f"{name}," + ",".join(f"{v:.8g}" for v in arr)
                for name, arr in (
                    ("per_dim_thresh", c.per_dim_thresh),
                    ("obs_min", c.obs_min),
                    ("obs_max", c.obs_max),
                )
            ]
            rows.append(f"mean_thresh,{c.mean_thresh:.8g}")
            rows.append(f"range_buffer,{c.range_buffer:.8g}")
            rows.append(f"risk_scale,{c.risk_scale:.8g}")
            (ckpt / "calibration.csv").write_text("\n".join(rows) + "\n")

    @classmethod
    def load(cls, ckpt_dir: str | Path, device: str = "cpu") -> "RaptSystem":
        ckpt = Path(ckpt_dir)
        cfg = RaptConfig.from_json(ckpt / "config.json")
        model = RaptModel(cfg)
        model.load_state_dict(torch.load(ckpt / "model.pt", map_location=device))
        model.eval().to(device)
        stats = NormStats.from_dict(json.loads((ckpt / "obs_stats.json").read_text()))
        cal = None
        if (ckpt / "calibration.json").exists():
            cal = Calibration.from_dict(json.loads((ckpt / "calibration.json").read_text()))
        history = {}
        if (ckpt / "history.json").exists():
            history = json.loads((ckpt / "history.json").read_text())
        return cls(cfg, model, stats, cal, history)

    # -- convenience --------------------------------------------------------

    def calibrate(
        self, nominal: SequenceData, ckpt_dir: str | Path | None = None, device: str = "cpu"
    ) -> Calibration:
        """Fit gate thresholds on nominal data and optionally persist them."""
        self.cal = calibrate(self.model, self.stats, nominal, self.cfg, device)
        if ckpt_dir is not None:
            ckpt = Path(ckpt_dir)
            (ckpt / "calibration.json").write_text(json.dumps(self.cal.to_dict(), indent=2))
            self._write_calibration_losses(nominal, ckpt / "calibration_losses.csv", device)
            self._write_deploy_bundle(ckpt)
        return self.cal

    def _write_calibration_losses(
        self, nominal: SequenceData, path: Path, device: str = "cpu"
    ) -> None:
        rows = [
            score_sequence(
                self.model,
                self.stats,
                nominal.obs[i],
                nominal.actions[i] if nominal.actions else None,
                device,
            )
            for i in range(len(nominal))
        ]
        flat = np.concatenate(rows, axis=0)
        header = "timestamp," + ",".join(f"dim_{i}" for i in range(flat.shape[1]))
        stamped = np.concatenate([np.arange(len(flat))[:, None] * self.cfg.dt, flat], axis=1)
        np.savetxt(path, stamped, delimiter=",", header=header, comments="", fmt="%.6g")

    def detector(self, use_range: bool | None = None, device: str = "cpu") -> RaptDetector:
        if self.cal is None:
            raise RuntimeError("System is not calibrated — run calibrate() first.")
        if use_range is None:
            use_range = self.cfg.use_range_gate
        return RaptDetector(self.model, self.stats, self.cal, use_range, device)

    def score(
        self, obs: np.ndarray, actions: np.ndarray | None = None, device: str = "cpu"
    ) -> np.ndarray:
        return score_sequence(self.model, self.stats, obs, actions, device)

    def first_detection(
        self, obs: np.ndarray, actions: np.ndarray | None = None, device: str = "cpu"
    ) -> int | None:
        """Step index of the first gate trigger in a sequence, or None."""
        if self.cal is None:
            raise RuntimeError("System is not calibrated — run calibrate() first.")
        from .detector import risk_trajectory

        losses = score_sequence(self.model, self.stats, obs, actions, device)
        risk = risk_trajectory(losses, obs, self.cal, self.cfg.use_range_gate)
        hits = np.nonzero(risk > 1.0)[0]
        return int(hits[0]) if len(hits) else None

    def attribute(
        self,
        obs: np.ndarray,
        actions: np.ndarray | None = None,
        window: int | None = None,
        device: str = "cpu",
    ) -> SaliencyResult:
        """Saliency for the window ending at the last row of ``obs``."""
        h = window or self.cfg.saliency_window
        if len(obs) < 2:
            raise ValueError("Need at least 2 timesteps for attribution.")
        obs_w = obs[-h:]
        act_w = actions[-h:] if actions is not None else None
        return integrated_gradients_bptt(
            self.model,
            self.stats,
            obs_w,
            act_w,
            ig_steps=self.cfg.ig_steps,
            top_k=self.cfg.top_k,
            dim_names=self.cfg.dim_names,
            device=device,
        )
