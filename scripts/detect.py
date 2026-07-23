#!/usr/bin/env python3
"""Run streaming OOD detection over a recorded sequence.

Feeds the sequence step-by-step through the calibrated detector (as the
on-robot runtime does) and reports the first trigger: time, which gate fired,
and the most anomalous dimension.

Examples:
  python scripts/detect.py checkpoints/quickstart data/sample/eval.npz --index 1
  python scripts/detect.py checkpoints/robot my_run.csv --plot risk.png
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
    ap.add_argument("sequence", help="a .csv/.npy file or a dataset (.npz/.h5/dir)")
    ap.add_argument("--index", type=int, default=0, help="sequence index within a dataset")
    ap.add_argument("--no-range", action="store_true", help="disable the range gate")
    ap.add_argument("--plot", help="save a risk-trajectory plot to this path")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    system = RaptSystem.load(args.checkpoint, args.device)
    data = load_sequences(args.sequence)
    obs = data.obs[args.index]
    act = data.actions[args.index] if data.actions else None
    names = system.cfg.dim_names or data.dim_names

    det = system.detector(use_range=not args.no_range, device=args.device)
    risks, first = [], None
    for t in range(len(obs)):
        r = det.step(obs[t], act[t] if act is not None else None)
        risks.append(r.risk)
        if r.is_anomaly and first is None:
            first = (t, r)
    risks = np.array(risks)

    dt = system.cfg.dt
    print(f"Sequence '{data.names[args.index]}': {len(obs)} steps ({len(obs) * dt:.1f} s)")
    if first is None:
        print(f"No anomaly detected (peak risk {risks.max():.3f}, threshold 1.0).")
    else:
        t, r = first
        gate = max(
            ("local per-dim", r.local_risk),
            ("global mean", r.global_risk),
            ("range", r.range_risk),
            key=lambda g: g[1],
        )[0]
        dim = names[r.top_dim] if r.top_dim < len(names) else f"dim_{r.top_dim}"
        print(f"OOD DETECTED at step {t} (t={t * dt:.2f} s)")
        print(f"  gate: {gate} | risk {r.risk:.2f} (threshold 1.0)")
        print(f"  most anomalous dimension: {dim}")
        print(f"  → run scripts/attribute.py to generate root-cause saliency")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 3))
        ts = np.arange(len(risks)) * dt
        ax.plot(ts, risks, lw=0.8)
        ax.axhline(1.0, color="r", ls="--", lw=0.8, label="detection threshold")
        if first is not None:
            ax.axvline(first[0] * dt, color="orange", lw=0.8, label="first trigger")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("risk")
        ax.set_yscale("log")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"Risk plot saved to {args.plot}")


if __name__ == "__main__":
    main()
