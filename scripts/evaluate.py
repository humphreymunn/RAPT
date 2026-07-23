#!/usr/bin/env python3
"""Evaluate a trained RAPT checkpoint on a labeled dataset.

Reports the paper's metrics: AUROC, Safety Score (TPR @ fixed episode-level
FPR, default 0.5%), the confusion counts at the calibrated operating point
(risk > 1), and PADD when the dataset provides anomaly onset steps (an
``onset`` array in .npz, -1 for nominal sequences).

Example:
  python scripts/evaluate.py checkpoints/quickstart data/sample/eval.npz
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from rapt import RaptSystem, auroc, load_sequences, padd, safety_score
from rapt.detector import risk_trajectory, score_dataset
from rapt.metrics import confusion_from_flags


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("checkpoint", help="checkpoint directory")
    ap.add_argument("dataset", help="labeled dataset (needs a `labels` array or labels.json)")
    ap.add_argument("--fpr", type=float, default=0.005, help="Safety Score operating FPR")
    ap.add_argument("--no-range", action="store_true", help="disable the range gate (Model Only)")
    ap.add_argument("--save-json", help="write all metrics to this JSON file")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    system = RaptSystem.load(args.checkpoint, args.device)
    if system.cal is None:
        ap.error("Checkpoint has no calibration.json — run scripts/calibrate.py first.")
    data = load_sequences(args.dataset)
    if data.labels is None:
        ap.error("Dataset has no labels — evaluation needs nominal/anomalous labels.")
    use_range = not args.no_range and system.cfg.use_range_gate

    onsets = faults = None
    if Path(args.dataset).suffix == ".npz":
        extras = np.load(args.dataset)
        if "onset" in extras.files:
            onsets = extras["onset"].astype(int)
        if "fault" in extras.files:
            faults = [str(f) for f in extras["fault"]]

    losses = score_dataset(system.model, system.stats, data, args.device)
    risks = [
        risk_trajectory(l, o, system.cal, use_range) for l, o in zip(losses, data.obs)
    ]
    ep_scores = np.array([r.max() for r in risks])
    labels = np.array(data.labels)

    metrics: dict = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "gates": "hybrid" if use_range else "model_only",
        "n_sequences": len(data),
        "n_nominal": int((labels == 0).sum()),
        "n_anomalous": int(labels.sum()),
        "operating_fpr": args.fpr,
        "auroc": auroc(ep_scores, labels),
        "safety_score": safety_score(ep_scores[labels == 0], ep_scores[labels == 1], args.fpr),
    }
    print(f"Evaluated {len(data)} sequences "
          f"({metrics['n_nominal']} nominal / {metrics['n_anomalous']} anomalous), "
          f"gates: {'Hybrid (with range)' if use_range else 'Model Only'}")
    print(f"AUROC (episode-level):        {metrics['auroc']:.3f}")
    print(f"Safety Score (TPR@{args.fpr:.1%} FPR): {metrics['safety_score']:.3f}")

    flagged = [bool(r.max() > 1.0) for r in risks]
    rep = confusion_from_flags(flagged, labels.tolist())
    metrics.update(tp=rep.tp, fn=rep.fn, tn=rep.tn, fp=rep.fp, tpr=rep.tpr, fpr=rep.fpr)
    print(f"Calibrated operating point:   {rep}")

    if onsets is not None:
        trig, ons, hor = [], [], []
        for r, onset, label in zip(risks, onsets, labels):
            if not label:
                continue
            steps = np.nonzero(r[onset:] > 1.0)[0]
            trig.append(int(onset + steps[0]) if len(steps) else None)
            ons.append(int(onset))
            hor.append(len(r))
        metrics["padd_seconds"] = padd(trig, ons, hor, dt=system.cfg.dt)
        print(f"PADD:                         "
              f"{metrics['padd_seconds']:.2f} s over {len(ons)} anomalous eps")

    if faults is not None:
        by_fault: dict[str, list[bool]] = defaultdict(list)
        for f, flag, label in zip(faults, flagged, labels):
            if label:
                by_fault[f].append(flag)
        metrics["detection_rate_by_fault"] = {
            f: {"rate": float(np.mean(flags)), "detected": int(sum(flags)), "total": len(flags)}
            for f, flags in sorted(by_fault.items())
        }
        print("Detection rate by fault type:")
        for f, flags in sorted(by_fault.items()):
            print(f"  {f:18s} {np.mean(flags):5.1%} ({sum(flags)}/{len(flags)})")

    if args.save_json:
        import json

        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_json).write_text(json.dumps(metrics, indent=2))
        print(f"Metrics saved to {args.save_json}")


if __name__ == "__main__":
    main()
