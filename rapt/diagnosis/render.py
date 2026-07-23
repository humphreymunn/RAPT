"""Render the diagnostic inputs sent to the LLM as images.

The paper's multimodal query attaches: (1) the log-scale temporal-saliency
heatmap; (2) a heatmap of raw signal values over the final window (e.g. joint
positions, downsampled to ~20 columns); (3) a command/context heatmap over a
longer horizon; and optionally one synchronized camera keyframe.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..saliency import SaliencyResult, plot_saliency_heatmap


def _downsample(x: np.ndarray, cols: int) -> np.ndarray:
    if len(x) <= cols:
        return x
    idx = np.linspace(0, len(x) - 1, cols).astype(int)
    return x[idx]


def plot_signal_heatmap(
    values: np.ndarray,
    names: list[str],
    path: str | Path,
    dt: float = 0.02,
    cols: int = 20,
    title: str = "Signal history",
    cmap: str = "viridis",
) -> None:
    """Heatmap of ``values [T, D]`` (rows = dimensions), downsampled in time."""
    v = _downsample(values, cols)
    duration = len(values) * dt
    fig, ax = plt.subplots(figsize=(10, 0.35 * v.shape[1] + 2))
    im = ax.imshow(
        v.T, aspect="auto", cmap=cmap, extent=(-duration, 0.0, v.shape[1] - 0.5, -0.5)
    )
    ax.set_yticks(range(v.shape[1]))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Time before detection (s)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_diagnosis_inputs(
    saliency: SaliencyResult,
    obs_window: np.ndarray,
    out_dir: str | Path,
    dim_names: list[str],
    dt: float = 0.02,
    signal_dims: list[int] | None = None,
    command_window: np.ndarray | None = None,
    command_names: list[str] | None = None,
    keyframe: str | Path | None = None,
) -> list[Path]:
    """Write the image set for one diagnostic query; returns the paths.

    - ``signal_dims``: which raw dimensions to show as "kinematics" (defaults
      to the saliency top-K).
    - ``command_window`` / ``command_names``: optional command/context signals
      over a longer horizon (paper: 15 s of command velocity).
    - ``keyframe``: optional path to a camera image, copied alongside.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    p = out / "saliency_heatmap.png"
    plot_saliency_heatmap(saliency, str(p), dt=dt)
    paths.append(p)

    dims = signal_dims
    if dims is None:
        dims = [i for i in saliency.top_idx if i < obs_window.shape[1]]
    p = out / "signal_history.png"
    plot_signal_heatmap(
        obs_window[:, dims],
        [dim_names[i] for i in dims],
        p,
        dt=dt,
        title="Raw signal history (most salient dims)",
    )
    paths.append(p)

    if command_window is not None:
        p = out / "command_history.png"
        plot_signal_heatmap(
            command_window,
            command_names or [f"cmd_{i}" for i in range(command_window.shape[1])],
            p,
            dt=dt,
            cols=20,
            title="Command history",
            cmap="coolwarm",
        )
        paths.append(p)

    if keyframe is not None:
        import shutil

        p = out / ("keyframe" + Path(keyframe).suffix)
        shutil.copy(keyframe, p)
        paths.append(p)
    return paths
