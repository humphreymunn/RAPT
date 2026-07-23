#!/usr/bin/env python3
"""Convert a paper-era checkpoint directory (fdm_models_deploy/... /
mae_models_deploy/...) into this repo's checkpoint format.

Reads experiment_config.json + final_model.pth + obs_stats.h5 (+ optional
calibration_dataset_loss.csv), renames the state dict to the reconciled
RaptModel layout, computes compact calibration thresholds, verifies parity
against the original rapt.onnx if present, and writes a standard checkpoint.

Example:
  python scripts/convert_legacy_checkpoint.py \\
      /path/to/fdm_models_deploy/DYN_Unitree-G1-29dof-Velocity_20260325_125357 \\
      --out checkpoints/g1_29dof_velocity
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from rapt import NormStats, RaptConfig, RaptModel, RaptSystem
from rapt.detector import Calibration

G1_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def g1_velocity_dim_names() -> list[str]:
    names = ["root_vel_x", "root_vel_y", "root_vel_z",
             "gravity_x", "gravity_y", "gravity_z",
             "cmd_vel_x", "cmd_vel_y", "cmd_vel_yaw"]
    for prefix in ("pos_", "vel_", "action_"):
        names += [prefix + j for j in G1_JOINTS]
    return names


def rename_state_dict(sd: dict) -> dict:
    """Legacy UniversalModel names → RaptModel names."""
    out = {}
    for k, v in sd.items():
        nk = k
        if k.startswith("encoder_mlp.0."):
            nk = k.replace("encoder_mlp.0.", "encoder_in.0.")
        elif k.startswith("encoder_mlp."):
            idx = int(k.split(".")[1])
            nk = k.replace(f"encoder_mlp.{idx}.", f"encoder_blocks.{idx - 2}.")
        elif k.startswith("decoder_mlp."):
            idx = int(k.split(".")[1])
            nk = k.replace(f"decoder_mlp.{idx}.", f"decoder_blocks.{idx}.")
        out[nk] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("legacy_dir", help="paper-era model directory")
    ap.add_argument("--out", required=True, help="output checkpoint directory")
    ap.add_argument("--weights", default="final_model.pth", help="which weights file")
    ap.add_argument("--dim-names", help="JSON list of observation names "
                    "(default: G1 29-DoF velocity names when obs_dim=96)")
    ap.add_argument("--dt", type=float, default=0.02)
    args = ap.parse_args()

    legacy = Path(args.legacy_dir)
    exp = json.loads((legacy / "experiment_config.json").read_text())
    sd = torch.load(legacy / args.weights, map_location="cpu")
    sd = rename_state_dict(sd)

    obs_dim = sd["head.weight"].shape[0] // (2 if exp.get("use_probabilistic", True) else 1)
    input_dim = sd["encoder_in.0.weight"].shape[1]
    action_dim = input_dim - obs_dim if exp.get("train_dynamics") else 0

    if args.dim_names:
        dim_names = json.loads(Path(args.dim_names).read_text())
    elif obs_dim == 96:
        dim_names = g1_velocity_dim_names()
    else:
        dim_names = [f"dim_{i}" for i in range(obs_dim)]

    cfg = RaptConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        seq_len=int(exp.get("seq_len", 50)),
        dt=args.dt,
        dim_names=dim_names,
        embed_dim=int(exp.get("embed_dim", 256)),
        num_blocks=int(exp.get("num_blocks", 4)),
        compression_ratio=float(exp.get("mask_ratio", 0.25)),
        dropout=float(exp.get("dropout", 0.0)),
        use_residual=bool(exp.get("use_residual", True)),
        use_probabilistic=bool(exp.get("use_probabilistic", True)),
        use_temporal=bool(exp.get("use_temporal", True)),
        reconstruction_type=str(exp.get("reconstruction_type", "bottleneck")),
        train_dynamics=bool(exp.get("train_dynamics", False)),
        lr=float(exp.get("lr", 1e-3)),
        batch_size=int(exp.get("batch_size", 256)),
        num_epochs=int(exp.get("num_epochs", 100)),
        noise_scale=float(exp.get("noise_scale", 0.01)),
    )
    model = RaptModel(cfg)
    model.load_state_dict(sd)
    model.eval()
    print(f"Loaded weights: obs_dim={obs_dim}, action_dim={action_dim}, "
          f"dynamics={cfg.train_dynamics}, embed={cfg.embed_dim}")

    import h5py

    with h5py.File(legacy / "obs_stats.h5") as f:
        mean = np.asarray(f["mean"], dtype=np.float32)
        std = np.asarray(f["std"], dtype=np.float32)
        mn = np.asarray(f["min"], dtype=np.float32)
        mx = np.asarray(f["max"], dtype=np.float32)
    if mn.ndim == 2:  # per-episode rows → global bounds
        mn, mx = mn.min(axis=0), mx.max(axis=0)
    stats = NormStats(mean=mean, std=std, min=mn, max=mx)

    cal = None
    cal_csv = legacy / "calibration_dataset_loss.csv"
    if cal_csv.exists():
        print(f"Computing thresholds from {cal_csv.name} ...")
        import pandas as pd

        losses = None
        mean_max, mean_sum, mean_sumsq, mean_n = -np.inf, 0.0, 0.0, 0
        chunks_max, sample = [], []
        for chunk in pd.read_csv(cal_csv, chunksize=200_000, dtype=np.float32):
            arr = chunk.iloc[:, 1:].to_numpy()  # drop timestamp
            chunks_max.append(arr.max(axis=0))
            step = max(1, len(arr) // 20_000)
            sample.append(arr[::step])
            m = arr.mean(axis=1)
            mean_max = max(mean_max, float(m.max()))
            mean_sum += float(m.sum())
            mean_sumsq += float((m**2).sum())
            mean_n += len(m)
        per_dim_max = np.max(chunks_max, axis=0)
        per_dim_median = np.median(np.concatenate(sample, axis=0), axis=0)
        mean_mu = mean_sum / mean_n
        mean_std = float(np.sqrt(max(mean_sumsq / mean_n - mean_mu**2, 0.0)))
        cal = Calibration(
            per_dim_median=per_dim_median,
            per_dim_max=per_dim_max,
            per_dim_thresh=per_dim_max + cfg.k_dim * (per_dim_max - per_dim_median),
            mean_max=mean_max,
            mean_std=mean_std,
            mean_thresh=mean_max + cfg.k_global * mean_std,
            obs_min=mn.astype(np.float64),
            obs_max=mx.astype(np.float64),
            range_buffer=cfg.range_buffer,
            k_dim=cfg.k_dim,
            k_global=cfg.k_global,
        )
        print(f"  global mean-NLL threshold {cal.mean_thresh:.4g} "
              f"({mean_n} calibration steps)")

    system = RaptSystem(cfg, model, stats, cal)
    out = system.save(args.out, export_onnx=True)

    legacy_onnx = legacy / "rapt.onnx"
    if legacy_onnx.exists():
        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(str(legacy_onnx), providers=["CPUExecutionProvider"])
            x = np.random.randn(1, cfg.input_dim).astype(np.float32)
            h = np.zeros((1, 1, cfg.embed_dim), dtype=np.float32)
            ref, _ = sess.run(None, {"input": x, "hidden_in": h})
            with torch.no_grad():
                mu, log_var, _ = model(torch.from_numpy(x), torch.from_numpy(h))
                got = torch.cat([mu, log_var], dim=-1).numpy()
            err = float(np.abs(ref - got).max())
            print(f"Parity vs original rapt.onnx: max abs err {err:.2e}")
            if err > 1e-3:
                raise SystemExit("Conversion mismatch against the original ONNX!")
        except ImportError:
            print("onnxruntime not installed — skipped original-ONNX parity check.")
    print(f"Converted checkpoint written to {out}")


if __name__ == "__main__":
    main()
