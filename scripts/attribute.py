#!/usr/bin/env python3
"""Root-cause attribution (temporal saliency) for a sequence of observations.

Finds the detection point (or uses --at), then attributes the model's NLL at
that step to the preceding history window via integrated gradients through
time. Saves the saliency heatmap, the raw attribution map, and prints the
top-K dimensions with their saliency-onset times.

Note: attribution quality degrades for very short histories — prefer at least
one full saliency window (config `saliency_window`, paper: 200 steps = 4 s).

Examples:
  python scripts/attribute.py checkpoints/quickstart data/sample/eval.npz --index 1
  python scripts/attribute.py checkpoints/robot run.csv --at 1200 --out diag/
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
    ap.add_argument("--at", type=int, help="attribute at this step (default: first detection)")
    ap.add_argument("--window", type=int, help="history window H (default from config)")
    ap.add_argument("--out", default="attribution", help="output directory")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    system = RaptSystem.load(args.checkpoint, args.device)
    data = load_sequences(args.sequence)
    obs = data.obs[args.index]
    act = data.actions[args.index] if data.actions else None
    dt = system.cfg.dt

    step = args.at
    if step is None:
        if system.cal is None:
            ap.error("Checkpoint is uncalibrated — pass --at or calibrate first.")
        step = system.first_detection(obs, act, args.device)
        if step is None:
            print("No detection in this sequence; attributing at the final step.")
            step = len(obs) - 1
        else:
            print(f"First detection at step {step} (t={step * dt:.2f} s).")

    h = args.window or system.cfg.saliency_window
    if step + 1 < min(h, 20):
        print(f"Warning: only {step + 1} steps of history available — "
              "attribution over very short windows is unreliable.")
    result = system.attribute(
        obs[: step + 1], act[: step + 1] if act is not None else None, window=h, device=args.device
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from rapt.saliency import plot_saliency_heatmap

    plot_saliency_heatmap(result, str(out / "saliency_heatmap.png"), dt=dt)
    np.savez_compressed(
        out / "attribution.npz",
        attribution=result.attribution,
        top_idx=result.top_idx,
        top_names=np.array(result.top_names),
        detection_step=step,
    )

    print(f"\nTop-{len(result.top_idx)} salient dimensions (|IG| max over window):")
    win_len = result.attribution.shape[0]
    for rank, (i, name) in enumerate(zip(result.top_idx, result.top_names), 1):
        mag = float(np.abs(result.attribution[:, i]).max())
        onset = result.onset_steps.get(int(i))
        onset_s = f"{(onset - win_len) * dt:+.2f} s" if onset is not None else "-"
        print(f"  {rank:2d}. {name:30s} |IG|max {mag:9.3g}   onset {onset_s}")
    print(f"\nSaved {out / 'saliency_heatmap.png'} and {out / 'attribution.npz'}")


if __name__ == "__main__":
    main()
