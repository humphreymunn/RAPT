"""ONNX export for deployment (C++/Python onnxruntime).

Graph signature (matches the on-robot runtime):
  inputs:  ``input     [B, input_dim]``  — normalized obs (‖ raw action)
           ``hidden_in [1, B, embed_dim]`` — GRU state (zeros at episode start)
  outputs: ``reconstruction [B, 2*obs_dim]`` — ``[mu | log_var]``
           ``hidden_out     [1, B, embed_dim]``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .model import RaptModel


class _OnnxWrapper(nn.Module):
    def __init__(self, model: RaptModel):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, hidden: torch.Tensor):
        mu, log_var, hidden_out = self.model(x, hidden)
        if hidden_out is None:  # non-temporal ablation: pass state through
            hidden_out = hidden
        return torch.cat([mu, log_var], dim=-1), hidden_out


def export_onnx(model: RaptModel, path: str | Path, opset: int = 13) -> Path:
    path = Path(path)
    model = model.eval().cpu()
    wrapper = _OnnxWrapper(model)
    x = torch.zeros(1, model.cfg.input_dim)
    h = model.init_hidden(1)
    torch.onnx.export(
        wrapper,
        (x, h),
        str(path),
        input_names=["input", "hidden_in"],
        output_names=["reconstruction", "hidden_out"],
        dynamic_axes={
            "input": {0: "batch"},
            "hidden_in": {1: "batch"},
            "reconstruction": {0: "batch"},
            "hidden_out": {1: "batch"},
        },
        opset_version=opset,
    )
    return path


def verify_onnx(model: RaptModel, path: str | Path, atol: float = 1e-4) -> float:
    """Compare ONNX output against PyTorch on random inputs; returns max abs
    error. Requires ``onnxruntime``."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.random.randn(1, model.cfg.input_dim).astype(np.float32)
    h = np.zeros((1, 1, model.cfg.embed_dim), dtype=np.float32)
    out, h_out = sess.run(None, {"input": x, "hidden_in": h})
    with torch.no_grad():
        mu, log_var, h_t = model(torch.from_numpy(x), torch.from_numpy(h))
        ref = torch.cat([mu, log_var], dim=-1).numpy()
    err = float(np.abs(out - ref).max())
    if h_t is not None:
        err = max(err, float(np.abs(h_out - h_t.numpy()).max()))
    if err > atol:
        raise AssertionError(f"ONNX/PyTorch mismatch: max abs error {err:.2e} > {atol:.0e}")
    return err
