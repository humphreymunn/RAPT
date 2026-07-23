#!/usr/bin/env python3
"""(Re)calibrate a RAPT checkpoint's detection gates on nominal data.

Run this against a brief verified nominal recording from *your* deployment
(the paper uses 3×1-minute real-robot runs) to absorb static sim-to-real
offsets: sensor noise, latency, contact dynamics, hardware differences.
Updates calibration.json and calibration_losses.csv in the checkpoint.

Examples:
  python scripts/calibrate.py checkpoints/robot real_nominal_logs/
  python scripts/calibrate.py checkpoints/robot cal.npz --k-dim 2 --k-global 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from rapt import RaptSystem, load_sequences


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("checkpoint", help="checkpoint directory")
    ap.add_argument("nominal", help="nominal dataset (any labeled-anomalous sequences are dropped)")
    ap.add_argument("--k-dim", type=float, help="per-dimension gate margin (default from config)")
    ap.add_argument("--k-global", type=float, help="global-mean gate margin")
    ap.add_argument("--range-buffer", type=float, help="range-gate buffer fraction")
    ap.add_argument("--target-fpr", type=float, default=None,
                    help="set the operating point at the (1 - fpr) quantile of nominal "
                         "episode risk on this calibration data (paper sim protocol)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    system = RaptSystem.load(args.checkpoint, args.device)
    if args.k_dim is not None:
        system.cfg.k_dim = args.k_dim
    if args.k_global is not None:
        system.cfg.k_global = args.k_global
    if args.range_buffer is not None:
        system.cfg.range_buffer = args.range_buffer

    data = load_sequences(args.nominal).nominal()
    total_steps = sum(len(o) for o in data.obs)
    cal = system.calibrate(data, args.checkpoint, args.device)

    if args.target_fpr is not None:
        from rapt.detector import risk_trajectory, score_dataset

        losses = score_dataset(system.model, system.stats, data, args.device)
        ep_risk = np.array([
            risk_trajectory(l, o, cal, system.cfg.use_range_gate).max()
            for l, o in zip(losses, data.obs)
        ])
        q = float(np.quantile(ep_risk, 1.0 - args.target_fpr))
        cal.risk_scale = cal.risk_scale * q
        import json as _json

        (Path(args.checkpoint) / "calibration.json").write_text(
            _json.dumps(cal.to_dict(), indent=2))
        system._write_deploy_bundle(Path(args.checkpoint))
        print(f"Target-FPR calibration: nominal episode risk q{100 * (1 - args.target_fpr):.1f} "
              f"= {q:.3f} over {len(ep_risk)} episodes -> risk_scale {cal.risk_scale:.3f}")

    system.cfg.to_json(Path(args.checkpoint) / "config.json")
    print(f"Calibrated on {len(data)} nominal sequences ({total_steps} steps, "
          f"{total_steps * system.cfg.dt:.0f} s at {1 / system.cfg.dt:.0f} Hz).")
    print(f"  global mean-NLL threshold: {cal.mean_thresh:.4g} "
          f"(max {cal.mean_max:.4g} + {cal.k_global:.1f} std)")
    print(f"  per-dim thresholds: median-of-max {float(np.median(cal.per_dim_max)):.4g}, "
          f"margin k={cal.k_dim:.1f}")
    print(f"  range gate: ±{cal.range_buffer:.0%} of calibration min/max span")
    print(f"Updated {Path(args.checkpoint) / 'calibration.json'}")


if __name__ == "__main__":
    main()
