#!/usr/bin/env python3
"""Run the full RAPT pipeline over one or more dataset directories.

Each dataset directory must contain ``train.npz`` (nominal), ``test.npz``
(labeled), and optionally ``metadata.json`` (as produced by the Isaac Lab
collection script). For each dataset this trains a RAPT model (forward
dynamics for velocity-style tasks, reconstruction otherwise), calibrates the
gates, evaluates on the test split, and saves:

  <out>/<name>/checkpoint/     full RAPT checkpoint (.pt + .onnx + gates)
  <out>/<name>/train.log       training output (per-epoch NLL)
  <out>/<name>/metrics.json    AUROC / Safety Score / confusion / PADD / per-fault
  <out>/summary.json, summary.md

Example:
  python scripts/benchmark.py datasets/g1_velocity datasets/g1_mimic_* --out results
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_logged(cmd: list[str], log_path: Path) -> None:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    with open(log_path, "w", buffering=1) as log:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in proc.stdout:
            log.write(line)
            if any(k in line for k in ("epoch", "NLL", "AUROC", "Safety", "PADD",
                                       "Calibrated", "operating", "saved", "Checkpoint")):
                print("  " + line.rstrip())
        if proc.wait() != 0:
            raise SystemExit(f"Command failed (see {log_path})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("datasets", nargs="+", help="dataset directories (train.npz + test.npz)")
    ap.add_argument("--out", default="results", help="results output directory")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--fpr", type=float, default=0.005)
    ap.add_argument("--skip-train", action="store_true",
                    help="reuse existing checkpoints, only evaluate")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    out_root = Path(args.out)
    rows = []
    for ds in args.datasets:
        ds = Path(ds)
        name = ds.name
        meta = {}
        if (ds / "metadata.json").exists():
            meta = json.loads((ds / "metadata.json").read_text())
        dynamics = "velocity" in meta.get("task", name).lower()
        run_dir = out_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt = run_dir / "checkpoint"
        print(f"\n=== {name} ({'forward-dynamics' if dynamics else 'reconstruction'}) ===")

        if not args.skip_train or not (ckpt / "model.pt").exists():
            cmd = [sys.executable, "-u", str(ROOT / "scripts" / "train.py"), str(ds / "train.npz"),
                   "--out", str(ckpt), "--epochs", str(args.epochs),
                   "--dt", str(meta.get("dt", 0.02))]
            if dynamics:
                cmd.append("--dynamics")
            if args.device:
                cmd += ["--device", args.device]
            run_logged(cmd, run_dir / "train.log")

        def dev_args() -> list[str]:
            if args.device:
                return ["--device", args.device]
            try:
                import torch

                if torch.cuda.is_available():
                    return ["--device", "cuda"]
            except ImportError:
                pass
            return []

        # Paper sim protocol: calibrate the operating point on a dedicated
        # nominal batch collected under evaluation conditions, at target FPR.
        cal_npz = ds / "calibration.npz"
        if cal_npz.exists():
            cmd = [sys.executable, "-u", str(ROOT / "scripts" / "calibrate.py"), str(ckpt),
                   str(cal_npz), "--target-fpr", str(args.fpr)] + dev_args()
            run_logged(cmd, run_dir / "calibrate.log")

        metrics_path = run_dir / "metrics.json"
        cmd = [sys.executable, "-u", str(ROOT / "scripts" / "evaluate.py"), str(ckpt),
               str(ds / "test.npz"), "--fpr", str(args.fpr),
               "--save-json", str(metrics_path)] + dev_args()
        run_logged(cmd, run_dir / "eval.log")

        m = json.loads(metrics_path.read_text())
        m["dataset_name"] = name
        m["task"] = meta.get("task", name)
        m["objective"] = "dynamics" if dynamics else "reconstruction"
        rows.append(m)

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(rows, indent=2))
    lines = [
        "| dataset | objective | AUROC | Safety Score | TPR | FPR | PADD (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in rows:
        lines.append(
            f"| {m['dataset_name']} | {m['objective']} | {m['auroc']:.3f} | "
            f"{m['safety_score']:.3f} | {100 * m['tpr']:.1f}% | {100 * m['fpr']:.1f}% | "
            f"{m.get('padd_seconds', float('nan')):.2f} |"
        )
    summary_md = "\n".join(lines)
    (out_root / "summary.md").write_text(summary_md + "\n")
    print(f"\n{summary_md}")
    print(f"\nSummary written to {out_root}/summary.json and summary.md")


if __name__ == "__main__":
    main()
