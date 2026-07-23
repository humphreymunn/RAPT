#!/usr/bin/env python3
"""Convert on-robot logger output into a standard sequence dataset (.npz).

The C++ deployment stack writes one directory per run containing
``observations.csv`` and ``actions.csv`` (header row, leading timestamp
column, one row per 50 Hz step). This script bundles one or more run
directories into a dataset usable by every other script.

Note on actions: the logger stores *scaled* actions (post action-scale); the
forward-dynamics model was trained on raw policy outputs. Pass the policy's
action scale (0.25 for the paper's G1 tasks) so actions are un-scaled.

Examples:
  python scripts/import_deploy_logs.py logs/cali_real_new_1 logs/cal_real_new_2 \\
      --action-scale 0.25 --out real_calibration.npz
  python scripts/import_deploy_logs.py logs/nominal_* logs/latency* \\
      --action-scale 0.25 --label-pattern "latency=1" --out real_eval.npz
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


def load_run(run_dir: Path, action_scale: float):
    obs = np.loadtxt(run_dir / "observations.csv", delimiter=",", skiprows=1, ndmin=2)[:, 1:]
    actions = None
    act_file = run_dir / "actions.csv"
    if act_file.exists():
        actions = np.loadtxt(act_file, delimiter=",", skiprows=1, ndmin=2)[:, 1:]
        actions = actions / action_scale
        n = min(len(obs), len(actions))
        obs, actions = obs[:n], actions[:n]
    return obs.astype(np.float32), None if actions is None else actions.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("runs", nargs="+", help="run directories (each with observations.csv)")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--action-scale", type=float, default=1.0,
                    help="divide logged actions by this (0.25 for the paper's G1 policies)")
    ap.add_argument("--label-pattern", action="append", default=[],
                    help="'regex=label' applied to run names, e.g. 'latency|push=1'; "
                         "unmatched runs get label 0 (nominal)")
    args = ap.parse_args()

    patterns = []
    for spec in args.label_pattern:
        regex, _, label = spec.rpartition("=")
        patterns.append((re.compile(regex), int(label)))

    out: dict[str, np.ndarray] = {}
    labels, names = [], []
    have_actions = True
    runs = [Path(r) for r in args.runs if (Path(r) / "observations.csv").exists()]
    if not runs:
        ap.error("No run directory contains observations.csv")
    for i, run in enumerate(runs):
        obs, act = load_run(run, args.action_scale)
        out[f"seq_{i:05d}"] = obs
        if act is None:
            have_actions = False
        else:
            out[f"act_{i:05d}"] = act
        label = 0
        for regex, lab in patterns:
            if regex.search(run.name):
                label = lab
                break
        labels.append(label)
        names.append(run.name)
        print(f"  {run.name}: {len(obs)} steps ({len(obs) * 0.02:.0f} s @50 Hz), label {label}")
    if not have_actions:
        out = {k: v for k, v in out.items() if not k.startswith("act_")}
    if any(labels):
        out["labels"] = np.array(labels)
    np.savez_compressed(args.out, **out)
    print(f"Wrote {args.out} ({len(runs)} sequences"
          f"{', with actions' if have_actions else ''}).")


if __name__ == "__main__":
    main()
