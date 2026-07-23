#!/usr/bin/env python3
"""Train and evaluate the paper's baselines on the common dataset format.

Runs LSTM-VAE, Deep SVDD, and Isolation Forest with the paper's
hyperparameters on the same train/eval splits used for RAPT, and prints a
comparison table (AUROC + Safety Score @ 0.5% FPR). If a RAPT checkpoint is
given, its numbers are included in the table.

PatchAD is not vendored (third-party): see reproduce/README.md for the
configuration used with the official implementation.

Example:
  python reproduce/run_baselines.py data/sample/train.npz data/sample/eval.npz \\
      --rapt checkpoints/quickstart
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from rapt import auroc, load_sequences, safety_score
from reproduce.baselines import DeepSVDDDetector, IsolationForestDetector, LSTMVAEDetector


def episode_metrics(scores_per_seq: list[np.ndarray], labels: np.ndarray, fpr: float):
    ep = np.array([s.max() for s in scores_per_seq])
    return (
        auroc(ep, labels),
        safety_score(ep[labels == 0], ep[labels == 1], fpr),
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("train_dataset", help="nominal training dataset")
    ap.add_argument("eval_dataset", help="labeled evaluation dataset")
    ap.add_argument("--methods", nargs="*", default=["lstm_vae", "deep_svdd", "isolation_forest"])
    ap.add_argument("--rapt", help="optional RAPT checkpoint to include in the table")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--fpr", type=float, default=0.005)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    train = load_sequences(args.train_dataset).nominal()
    eval_data = load_sequences(args.eval_dataset)
    if eval_data.labels is None:
        ap.error("Evaluation dataset needs labels.")
    labels = np.array(eval_data.labels)
    rows = []

    for method in args.methods:
        print(f"Training {method} ...")
        if method == "lstm_vae":
            det = LSTMVAEDetector(train.obs_dim, device=device)
            det.fit(train.obs, epochs=args.epochs)
        elif method == "deep_svdd":
            det = DeepSVDDDetector(train.obs_dim, device=device)
            det.fit(train.obs, epochs=args.epochs)
        elif method == "isolation_forest":
            det = IsolationForestDetector()
            det.fit(train.obs)
        else:
            ap.error(f"Unknown method {method}")
        scores = [det.score(o) for o in eval_data.obs]
        a, s = episode_metrics(scores, labels, args.fpr)
        rows.append((method, a, s))

    if args.rapt:
        from rapt import RaptSystem
        from rapt.detector import risk_trajectory, score_dataset

        system = RaptSystem.load(args.rapt, device)
        losses = score_dataset(system.model, system.stats, eval_data, device)
        risks = [
            risk_trajectory(l, o, system.cal, system.cfg.use_range_gate)
            for l, o in zip(losses, eval_data.obs)
        ]
        a, s = episode_metrics(risks, labels, args.fpr)
        rows.append(("RAPT", a, s))

    print(f"\n{'method':20s} {'AUROC':>8s} {'Safety Score':>14s}   (TPR @ {args.fpr:.1%} FPR)")
    for name, a, s in rows:
        print(f"{name:20s} {a:8.3f} {s:14.3f}")


if __name__ == "__main__":
    main()
