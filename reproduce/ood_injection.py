#!/usr/bin/env python3
"""Observation-space OOD injection for building labeled evaluation sets.

Implements the paper's observation-level corruption categories with the same
sampling ranges, applicable to any sequence dataset (the physics-level
categories — actuator dynamics, pushes, friction, initial state — require the
simulator; see reproduce/isaaclab/play_with_rapt.py):

  sensor_drift    per-step bias U[5e-4, 5e-3] on 1-2 dims
  sensor_zero     zero out 1-5 dims
  scale_half      multiply 1-10 dims by 0.5
  scale_double    multiply 1-10 dims by 2.0
  obs_swap        swap two observation dims
  noise           additive Gaussian, std U[0.05, 0.15], on 1-5 dims
  latency_offset  observations delayed 1-5 steps
  latency_slow    observations update only every 2-10 steps
  frozen_sensor   1-2 dims hold their onset value

Takes nominal sequences, injects a category from a random onset in half of
them, and writes a labeled .npz (with `labels`, `onset`, `fault`) compatible
with scripts/evaluate.py.

Example:
  python reproduce/ood_injection.py nominal_test.npz --out eval_injected.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from rapt import load_sequences

CATEGORIES = [
    "sensor_drift",
    "sensor_zero",
    "scale_half",
    "scale_double",
    "obs_swap",
    "noise",
    "latency_offset",
    "latency_slow",
    "frozen_sensor",
]


def inject(obs: np.ndarray, category: str, onset: int, rng: np.random.Generator) -> np.ndarray:
    obs = obs.copy()
    D = obs.shape[1]
    tail = slice(onset, None)
    n_tail = len(obs) - onset
    if category == "sensor_drift":
        dims = rng.choice(D, rng.integers(1, 3), replace=False)
        for d in dims:
            obs[tail, d] += np.arange(n_tail) * rng.uniform(5e-4, 5e-3)
    elif category == "sensor_zero":
        dims = rng.choice(D, rng.integers(1, 6), replace=False)
        obs[onset:, dims] = 0.0
    elif category in ("scale_half", "scale_double"):
        factor = 0.5 if category == "scale_half" else 2.0
        dims = rng.choice(D, rng.integers(1, min(11, D + 1)), replace=False)
        obs[onset:, dims] *= factor
    elif category == "obs_swap":
        d1, d2 = rng.choice(D, 2, replace=False)
        obs[onset:, [d1, d2]] = obs[onset:, [d2, d1]]
    elif category == "noise":
        dims = rng.choice(D, rng.integers(1, 6), replace=False)
        std = rng.uniform(0.05, 0.15)
        obs[onset:, dims] += rng.normal(0, std, (n_tail, len(dims)))
    elif category == "latency_offset":
        delay = min(int(rng.integers(1, 6)), onset)
        obs[onset:] = obs[onset - delay : len(obs) - delay]
    elif category == "latency_slow":
        every = int(rng.integers(2, 11))
        held = obs[onset:].copy()
        for t in range(n_tail):
            held[t] = obs[onset + (t // every) * every]
        obs[onset:] = held
    elif category == "frozen_sensor":
        dims = rng.choice(D, rng.integers(1, 3), replace=False)
        obs[onset:, dims] = obs[onset, dims]
    else:
        raise ValueError(f"Unknown category {category}")
    return obs


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dataset", help="nominal sequences to corrupt")
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--categories", nargs="*", default=CATEGORIES)
    ap.add_argument("--anomalous-fraction", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = load_sequences(args.dataset).nominal()
    rng = np.random.default_rng(args.seed)
    out: dict[str, np.ndarray] = {"dim_names": np.array(data.dim_names)}
    labels, onsets, faults = [], [], []
    for i in range(len(data)):
        obs = data.obs[i]
        anomalous = rng.random() < args.anomalous_fraction
        onset, fault = -1, "none"
        if anomalous:
            onset = int(rng.integers(len(obs) // 4, 3 * len(obs) // 4))
            fault = args.categories[int(rng.integers(len(args.categories)))]
            obs = inject(obs, fault, onset, rng)
        out[f"seq_{i:05d}"] = obs
        if data.actions is not None:
            out[f"act_{i:05d}"] = data.actions[i]
        labels.append(int(anomalous))
        onsets.append(onset)
        faults.append(fault)
    out["labels"] = np.array(labels)
    out["onset"] = np.array(onsets)
    out["fault"] = np.array(faults)
    np.savez_compressed(args.out, **out)
    print(f"Wrote {args.out}: {sum(labels)}/{len(labels)} anomalous "
          f"({', '.join(sorted(set(faults) - {'none'}))})")


if __name__ == "__main__":
    main()
