#!/usr/bin/env python3
"""Generate a small synthetic time-series dataset for the quickstart.

Simulates a periodic multi-channel process (a stand-in for gait-like
proprioception): coupled oscillator channels + derivative channels + command
channels, with actions that track the commands. Anomalous sequences get one of
several injected faults in their second half (drift, noise burst, scaling,
frozen channel, channel swap), mirroring the paper's simulation OOD suite.

Outputs an .npz per split:
  train.npz  — nominal only (RAPT trains on nominal data)
  eval.npz   — labeled mix of nominal + anomalous, with `onset` steps
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

FAULTS = ["drift", "noise", "scale", "freeze", "swap"]


def make_sequence(rng: np.random.Generator, T: int, n_osc: int = 8, n_cmd: int = 2):
    # A gait-like process: all channels are driven by one shared phase with
    # fixed per-channel offsets/amplitudes, so the joint structure is
    # learnable and structural faults (freeze/swap/scale) are detectable.
    offsets = np.linspace(0, 2 * np.pi, n_osc, endpoint=False)
    amps = 1.0 + 0.5 * np.sin(np.arange(n_osc))
    freq = rng.uniform(0.9, 1.1)
    t = np.arange(T) * 0.02
    cmd = np.clip(np.cumsum(rng.normal(0, 0.01, (T, n_cmd)), axis=0), -1, 1)
    phase = 2 * np.pi * np.cumsum(freq * (1 + 0.2 * cmd[:, 0]) * 0.02)
    osc = amps * np.sin(phase[:, None] + offsets) * (1 + 0.3 * cmd[:, 1:2])
    vel = np.gradient(osc, axis=0) / 0.02
    obs = np.concatenate([osc, 0.1 * vel, cmd], axis=1)
    obs += rng.normal(0, 0.02, obs.shape)
    actions = 0.5 * cmd + rng.normal(0, 0.05, (T, n_cmd))
    return obs.astype(np.float32), actions.astype(np.float32)


def inject(obs: np.ndarray, fault: str, onset: int, rng: np.random.Generator) -> np.ndarray:
    obs = obs.copy()
    d = rng.integers(0, obs.shape[1])
    if fault == "drift":
        obs[onset:, d] += np.arange(len(obs) - onset) * rng.uniform(5e-3, 2e-2)
    elif fault == "noise":
        obs[onset:, d] += rng.normal(0, rng.uniform(0.5, 1.5), len(obs) - onset)
    elif fault == "scale":
        obs[onset:, d] *= rng.choice([0.3, 2.5])
    elif fault == "freeze":
        obs[onset:, d] = obs[onset, d]
    elif fault == "swap":
        d2 = (d + 1) % obs.shape[1]
        obs[onset:, [d, d2]] = obs[onset:, [d2, d]]
    return obs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/sample", help="output directory")
    ap.add_argument("--n_train", type=int, default=60)
    ap.add_argument("--n_eval", type=int, default=30, help="eval sequences (half anomalous)")
    ap.add_argument("--steps", type=int, default=500, help="mean sequence length")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_osc, n_cmd = 8, 2
    dim_names = (
        [f"osc_{i}" for i in range(n_osc)]
        + [f"vel_{i}" for i in range(n_osc)]
        + [f"cmd_{i}" for i in range(n_cmd)]
    )

    def varying_T() -> int:
        return int(rng.integers(int(0.7 * args.steps), int(1.3 * args.steps)))

    train: dict[str, np.ndarray] = {"dim_names": np.array(dim_names)}
    for i in range(args.n_train):
        o, a = make_sequence(rng, varying_T(), n_osc, n_cmd)
        train[f"seq_{i:05d}"], train[f"act_{i:05d}"] = o, a
    np.savez_compressed(out / "train.npz", **train)

    evald: dict[str, np.ndarray] = {"dim_names": np.array(dim_names)}
    labels, onsets, faults = [], [], []
    for i in range(args.n_eval):
        o, a = make_sequence(rng, varying_T(), n_osc, n_cmd)
        anomalous = i % 2 == 1
        onset = -1
        fault = "none"
        if anomalous:
            onset = int(rng.integers(len(o) // 3, 2 * len(o) // 3))
            fault = FAULTS[i % len(FAULTS)]
            o = inject(o, fault, onset, rng)
        evald[f"seq_{i:05d}"], evald[f"act_{i:05d}"] = o, a
        labels.append(int(anomalous))
        onsets.append(onset)
        faults.append(fault)
    evald["labels"] = np.array(labels)
    evald["onset"] = np.array(onsets)
    evald["fault"] = np.array(faults)
    np.savez_compressed(out / "eval.npz", **evald)

    print(f"Wrote {out / 'train.npz'} ({args.n_train} nominal sequences)")
    print(f"Wrote {out / 'eval.npz'} ({args.n_eval} labeled sequences, D={len(dim_names)})")


if __name__ == "__main__":
    main()
