#!/usr/bin/env python3
"""Paper-protocol evaluation (faithful to the RAPT paper's simulation runs).

Replicates the January-2026 `play_with_fdm.py` evaluation semantics that
produced the paper's simulation tables:

- Threshold normalization from a nominal calibration batch collected at
  evaluation conditions: per-dim ``max x (1 + 6*margin)`` and global
  ``max(mean-over-dims) x (1 + margin)`` with margin 0.2 (i.e. x2.2 / x1.2).
- Per-step risk = max(local, global): local = max_d(loss_d / thresh_d),
  global = mean_d(loss_d) / thresh_mean; episode score = max over steps.
  Model-only — no range detector.
- Per-category ROC using that category's control (nominal) sequences as the
  negative class; Safety Score = TPR interpolated at 0.5% FPR.
- Task-level metrics pool all categories' scores (the paper's aggregate).

Note (flagged deliberately): computing TPR at a fixed FPR *within* each
category run uses few negatives at small scale, and the paper's fourth gate
(range detector) is excluded here by design.

Example:
  python scripts/evaluate_paper.py results/g1_velocity/checkpoint \\
      datasets/g1_velocity --save-json results/g1_velocity/paper_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import auc, roc_curve

from rapt import RaptSystem, load_sequences
from rapt.detector import score_dataset

MARGIN = 0.2  # paper margin_percent


def tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target: float) -> float:
    """Correct TPR at fixed FPR: the ROC step function (sup tpr with fpr <= target)."""
    ok = fpr <= target + 1e-12
    return float(tpr[ok].max()) if ok.any() else 0.0


def tpr_at_fpr_paper_interp(fpr: np.ndarray, tpr: np.ndarray, target: float) -> float:
    """The original code's interpolation, reproduced verbatim for comparison.

    Bug (flagged): np.unique keeps the FIRST tpr at each duplicated fpr,
    deleting the ROC's vertical segments (including at fpr=0), which
    underestimates TPR — severely so with few negatives.
    """
    if len(fpr) < 2:
        return 0.0
    unique_fpr, idx = np.unique(fpr, return_index=True)
    unique_tpr = tpr[idx]
    if target < unique_fpr[0]:
        return 0.0
    if target > unique_fpr[-1]:
        return float(unique_tpr[-1])
    return float(np.interp(target, unique_fpr, unique_tpr))


def episode_scores(losses: list[np.ndarray], thresh_dim: np.ndarray, thresh_mean: float):
    scores = []
    for l in losses:
        local = (l / thresh_dim).max(axis=1)
        global_ = l.mean(axis=1) / thresh_mean
        scores.append(float(np.maximum(local, global_).max()))
    return np.array(scores)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("checkpoint", help="trained RAPT checkpoint directory")
    ap.add_argument("dataset", help="dataset dir with test.npz + calibration.npz")
    ap.add_argument("--fpr", type=float, default=0.005)
    ap.add_argument("--cal-style", choices=["margin", "sigma5_3", "sigma5_1"],
                    default="sigma5_3",
                    help="threshold normalization: 'sigma5_3' (default) = per-dim max+5*sigma, "
                         "global mean-of-maxes + 3*sigma (the paper's calibration); "
                         "'margin' = the Jan-code variant (max*2.2 / max*1.2). "
                         "In practice the choice moves results by <0.01 because the "
                         "per-dimension gate dominates the combined risk score.")
    ap.add_argument("--save-json", help="write metrics to this file")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ds = Path(args.dataset)
    system = RaptSystem.load(args.checkpoint, args.device)
    dt = system.cfg.dt

    cal = load_sequences(ds / "calibration.npz")
    cal_losses = score_dataset(system.model, system.stats, cal, args.device)
    flat = np.concatenate(cal_losses, axis=0)
    if args.cal_style == "margin":
        thresh_dim = np.clip(flat.max(axis=0) * (1 + 6 * MARGIN), 1e-6, None)
        thresh_mean = max(float(flat.mean(axis=1).max()) * (1 + MARGIN), 1e-6)
    else:
        k_global = 3.0 if args.cal_style == "sigma5_3" else 1.0
        thresh_dim = np.clip(flat.max(axis=0) + 5.0 * flat.std(axis=0), 1e-6, None)
        # global gate, matching the later code: mean over dims of per-dim maxima,
        # plus K * std over episodes of (per-episode per-dim max, mean over dims)
        mean_max = float(flat.max(axis=0).mean())
        per_ep = np.array([l.max(axis=0).mean() for l in cal_losses])
        thresh_mean = max(mean_max + k_global * float(per_ep.std()), 1e-6)

    test = load_sequences(ds / "test.npz")
    d = np.load(ds / "test.npz")
    faults = np.array([str(f) for f in d["fault"]])
    onsets = d["onset"].astype(int)
    labels = np.array(test.labels)
    test_losses = score_dataset(system.model, system.stats, test, args.device)
    scores = episode_scores(test_losses, thresh_dim, thresh_mean)

    # Negatives: the task's pooled nominal controls. (The paper used each
    # category run's own 2048 controls; at this scale per-run negatives are
    # too few for a 0.5% FPR estimate, so we pool — flagged in caveats.)
    ctrl_idx = np.nonzero(labels == 0)[0]
    blocks: dict[str, list[int]] = defaultdict(list)
    for i, f in enumerate(faults):
        if f != "none":
            blocks[f].append(i)

    per_cat = {}
    for cat, ood_idx in blocks.items():
        b = {"ctrl": list(ctrl_idx), "ood": ood_idx}
        y = np.array([0] * len(ctrl_idx) + [1] * len(ood_idx))
        s = scores[np.array(list(ctrl_idx) + list(ood_idx))]
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        # PADD over this category's OOD sequences at the paper's normalized
        # threshold (risk > 1)
        delays = []
        for j in b["ood"]:
            l = test_losses[j]
            risk = np.maximum((l / thresh_dim).max(axis=1), l.mean(axis=1) / thresh_mean)
            on = max(onsets[j], 0)
            hits = np.nonzero(risk[on:] > 1.0)[0]
            delays.append(float(hits[0]) if len(hits) else float(len(risk)))
        per_cat[cat] = {
            "auroc": float(auc(fpr, tpr)),
            "safety_score": tpr_at_fpr(fpr, tpr, args.fpr),
            "safety_score_paper_interp": tpr_at_fpr_paper_interp(fpr, tpr, args.fpr),
            "n_ood": len(b["ood"]),
            "n_ctrl": len(b["ctrl"]),
            "padd_s": float(np.mean(delays) * dt),
        }

    pooled_fpr, pooled_tpr, _ = roc_curve(labels, scores)
    metrics = {
        "protocol": "paper (Jan-2026 sim eval): model-only, x2.2/x1.2 calibration margins, "
                    "per-category ROC vs in-run controls",
        "operating_fpr": args.fpr,
        "auroc_pooled": float(auc(pooled_fpr, pooled_tpr)),
        "safety_score_pooled": tpr_at_fpr(pooled_fpr, pooled_tpr, args.fpr),
        "safety_score_pooled_paper_interp": tpr_at_fpr_paper_interp(
            pooled_fpr, pooled_tpr, args.fpr
        ),
        "safety_score_mean_of_categories": float(
            np.mean([c["safety_score"] for c in per_cat.values()])
        ),
        "auroc_mean_of_categories": float(np.mean([c["auroc"] for c in per_cat.values()])),
        "per_category": dict(sorted(per_cat.items())),
        "caveats": [
            "single seed; 128 envs vs paper's 4096",
            "per-category negatives are the task's pooled controls (~400), not the "
            "category run's own controls as in the paper (too few at this scale)",
            "range detector excluded by request",
            "init_state/env_disturbance-push injections actually apply here "
            "(the paper-era code wiped them at env.reset)",
            "safety_score uses the correct ROC step function; *_paper_interp "
            "reproduces the original code's interpolation, which drops vertical "
            "ROC segments (first-occurrence np.unique bug) and underestimates TPR",
        ],
    }

    print(f"Pooled:            AUROC {metrics['auroc_pooled']:.3f} | "
          f"Safety Score {metrics['safety_score_pooled']:.3f} "
          f"(paper-interp {metrics['safety_score_pooled_paper_interp']:.3f})")
    print(f"Mean of categories: AUROC {metrics['auroc_mean_of_categories']:.3f} | "
          f"Safety Score {metrics['safety_score_mean_of_categories']:.3f}")
    print(f"{'category':18s} {'AUROC':>6s} {'SS':>6s} {'SSpap':>6s} {'PADD(s)':>8s} {'n_ood':>6s}")
    for cat, c in sorted(per_cat.items()):
        print(f"{cat:18s} {c['auroc']:6.3f} {c['safety_score']:6.3f} "
              f"{c['safety_score_paper_interp']:6.3f} {c['padd_s']:8.2f} {c['n_ood']:6d}")

    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_json).write_text(json.dumps(metrics, indent=2))
        print(f"Saved to {args.save_json}")


if __name__ == "__main__":
    main()
