#!/usr/bin/env python3
"""Replay a recorded log through the runtime monitor and measure latency.

Example:
  python deploy/python/replay_example.py checkpoints/quickstart data/sample/eval.npz --index 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rapt_monitor import RaptMonitor


def load_obs(path: str, index: int):
    from rapt.data import load_sequences  # torch-free loaders would also work

    data = load_sequences(path)
    return (
        data.obs[index],
        data.actions[index] if data.actions else None,
        data.names[index],
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("log", help="recorded sequence (.csv/.npy/.npz/.h5/dir)")
    ap.add_argument("--index", type=int, default=0)
    args = ap.parse_args()

    obs, act, name = load_obs(args.log, args.index)
    monitor = RaptMonitor(args.checkpoint)

    latencies, first = [], None
    for t in range(len(obs)):
        t0 = time.perf_counter()
        r = monitor.step(obs[t], act[t] if act is not None else None)
        latencies.append(time.perf_counter() - t0)
        if r.is_anomaly and first is None:
            first = (t, r)

    lat = np.array(latencies[10:]) * 1e3  # skip warmup
    print(f"Replayed '{name}': {len(obs)} steps")
    print(f"Latency: mean {lat.mean():.2f} ms | p99 {np.percentile(lat, 99):.2f} ms "
          f"(paper: ~1.6 ms at 50 Hz → budget 20 ms)")
    if first:
        t, r = first
        print(f"OOD detected at step {t}: risk {r.risk:.2f}, "
              f"top dimension '{monitor.dim_names[r.top_dim]}'")
    else:
        print("No anomaly detected.")


if __name__ == "__main__":
    main()
