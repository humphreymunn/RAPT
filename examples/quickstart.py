#!/usr/bin/env python3
"""End-to-end RAPT quickstart on synthetic data (~2 min on CPU).

Generates a toy multichannel dataset, trains a small RAPT model, calibrates
the gates, evaluates detection, and runs root-cause attribution on one
anomalous sequence. Mirrors what the CLI scripts do, as a single readable
script.

  python examples/quickstart.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from rapt import RaptConfig, RaptSystem, auroc, load_sequences, train_rapt
from rapt.detector import risk_trajectory, score_dataset
from rapt.saliency import plot_saliency_heatmap

OUT = ROOT / "examples" / "quickstart_output"


def main() -> None:
    # 1. Synthetic dataset (N ragged sequences, ~10 s each at 50 Hz)
    data_dir = OUT / "data"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_synthetic_data.py"),
         "--out", str(data_dir)],
        check=True,
    )
    train = load_sequences(data_dir / "train.npz")
    eval_data = load_sequences(data_dir / "eval.npz")

    # 2. Train (small config so the demo is fast; defaults match the paper)
    tr, val, _ = train.split(val_fraction=0.2, test_fraction=0.0)
    cfg = RaptConfig(
        obs_dim=train.obs_dim,
        dim_names=train.dim_names,
        embed_dim=64,
        num_blocks=2,
        num_epochs=30,
        seq_len=50,
        saliency_window=100,
    )
    model, stats, history = train_rapt(cfg, tr, val)
    system = RaptSystem(cfg, model, stats, history=history)
    system.calibrate(val)
    system.save(OUT / "checkpoint")

    # 3. Detect: episode-level AUROC + per-sequence triggers
    losses = score_dataset(system.model, system.stats, eval_data)
    risks = [risk_trajectory(l, o, system.cal) for l, o in zip(losses, eval_data.obs)]
    ep_scores = np.array([r.max() for r in risks])
    labels = np.array(eval_data.labels)
    print(f"\nEpisode AUROC on labeled eval set: {auroc(ep_scores, labels):.3f}")

    detected = [(i, np.nonzero(r > 1.0)[0]) for i, r in enumerate(risks) if labels[i]]
    hits = [i for i, steps in detected if len(steps)]
    print(f"Detected {len(hits)}/{len(detected)} anomalous sequences at the "
          "calibrated operating point.")

    # 4. Root-cause attribution on the first detected anomaly
    if hits:
        i = hits[0]
        step = int(np.nonzero(risks[i] > 1.0)[0][0])
        sal = system.attribute(eval_data.obs[i][: step + 1])
        heatmap = OUT / "saliency_heatmap.png"
        plot_saliency_heatmap(sal, str(heatmap), dt=cfg.dt)
        print(f"\nSequence '{eval_data.names[i]}' detected at step {step}; "
              f"top salient dims: {', '.join(sal.top_names[:3])}")
        print(f"Saliency heatmap: {heatmap}")
        print("\nNext: run the LLM diagnosis on it —")
        print(f"  python scripts/diagnose.py {OUT / 'checkpoint'} "
              f"{data_dir / 'eval.npz'} --index {i} --provider anthropic")


if __name__ == "__main__":
    main()
