#!/usr/bin/env python3
"""Export (or re-export) a checkpoint's model to ONNX and verify parity.

Example:
  python scripts/export_onnx.py checkpoints/quickstart
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rapt import RaptSystem
from rapt.onnx_export import export_onnx, verify_onnx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", help="checkpoint directory")
    ap.add_argument("--out", help="output path (default <checkpoint>/rapt.onnx)")
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args()

    system = RaptSystem.load(args.checkpoint)
    out = Path(args.out) if args.out else Path(args.checkpoint) / "rapt.onnx"
    export_onnx(system.model, out, args.opset)
    print(f"Exported {out} (input [B,{system.cfg.input_dim}] + hidden [1,B,{system.cfg.embed_dim}] "
          f"→ [mu|log_var] [B,{2 * system.cfg.obs_dim}] + hidden)")
    try:
        err = verify_onnx(system.model, out)
        print(f"Parity check passed (max abs err {err:.2e}).")
    except ImportError:
        print("onnxruntime not installed — skipped parity check.")


if __name__ == "__main__":
    main()
