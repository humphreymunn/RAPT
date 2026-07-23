#!/usr/bin/env python3
"""Train a full RAPT system on a sequence dataset.

Splits the data per-sequence into train/val/test, trains on nominal data,
prints NLL metrics per epoch, calibrates the detection gates on the
validation split, reports held-out test metrics, and saves a complete
checkpoint (PyTorch + ONNX + normalization stats + calibration thresholds).

Examples:
  python scripts/train.py data/sample/train.npz --out checkpoints/quickstart
  python scripts/train.py my_logs_dir/ --dynamics --epochs 100 --seq-len 50 \\
      --dim-names my_labels.json --out checkpoints/robot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from rapt import RaptConfig, RaptSystem, load_sequences, train_rapt
from rapt.data import make_windows
from rapt.train import evaluate_nll


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dataset", help="dataset path (.npz / .h5 / directory of .csv|.npy)")
    ap.add_argument("--out", required=True, help="checkpoint output directory")
    ap.add_argument("--dynamics", action="store_true", help="forward-dynamics target (needs actions)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seq-len", type=int, default=50)
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--num-blocks", type=int, default=4)
    ap.add_argument("--compression", type=float, default=0.25)
    ap.add_argument("--dt", type=float, default=0.02, help="timestep seconds (default 50 Hz)")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--test-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim-names", help="JSON file with a list of dimension names (optional)")
    ap.add_argument("--no-onnx", action="store_true", help="skip ONNX export")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    data = load_sequences(args.dataset)
    if data.labels is not None:
        n_anom = sum(data.labels)
        if n_anom:
            print(f"Note: dropping {n_anom} labeled-anomalous sequences (RAPT trains on nominal).")
        data = data.nominal()
    dim_names = data.dim_names
    if args.dim_names:
        dim_names = json.loads(Path(args.dim_names).read_text())
        assert len(dim_names) == data.obs_dim, (
            f"--dim-names has {len(dim_names)} entries but data has {data.obs_dim} dims"
        )
    if args.dynamics and data.actions is None:
        ap.error("--dynamics requires the dataset to contain aligned action arrays")

    cfg = RaptConfig(
        obs_dim=data.obs_dim,
        action_dim=data.action_dim if args.dynamics else 0,
        seq_len=args.seq_len,
        dt=args.dt,
        dim_names=dim_names,
        embed_dim=args.embed_dim,
        num_blocks=args.num_blocks,
        compression_ratio=args.compression,
        train_dynamics=args.dynamics,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    train, val, test = data.split(cfg.val_fraction, cfg.test_fraction, cfg.seed)
    print(f"Sequences: {len(train)} train / {len(val)} val / {len(test)} test (D={data.obs_dim})")

    model, stats, history = train_rapt(cfg, train, val, device=args.device)
    system = RaptSystem(cfg, model, stats, history=history)

    if len(test):
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        x, y = make_windows(test, cfg.seq_len, stats, cfg.train_dynamics)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
            batch_size=cfg.batch_size,
        )
        test_nll = evaluate_nll(model, loader, device)
        history["test_nll"] = test_nll
        print(f"Held-out test NLL: {test_nll:+.4f}")

    ckpt = system.save(args.out, export_onnx=not args.no_onnx)
    cal_data = val if len(val) else train
    system.calibrate(cal_data, ckpt)
    print(f"Calibrated gates on {len(cal_data)} nominal sequences "
          f"(global thresh {system.cal.mean_thresh:.4g}).")

    if not args.no_onnx:
        try:
            from rapt.onnx_export import verify_onnx

            err = verify_onnx(model.cpu(), ckpt / "rapt.onnx")
            print(f"ONNX parity check passed (max abs err {err:.2e}).")
        except ImportError:
            print("onnxruntime not installed — skipped ONNX parity check.")
    print(f"Checkpoint saved to {ckpt}")


if __name__ == "__main__":
    main()
