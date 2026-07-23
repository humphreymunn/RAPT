"""Dataset handling for RAPT.

A dataset is a collection of **sequences** — each a ``[T_i, D]`` float array
(time × dimensions). ``T_i`` may differ between sequences. The data can be
anything with temporal structure: proprioceptive robot state, tabular sensor
streams, or a learned latent space. Optionally each sequence has:

- an aligned action array ``[T_i, A]`` (required for the forward-dynamics
  training target),
- a binary label (0 = nominal, 1 = anomalous) for evaluation only — RAPT
  itself trains purely on nominal data,
- per-dimension names for attribution/diagnosis.

Supported on-disk formats (``load_sequences``):

- **Directory of .csv files** — one file per sequence, header row = dimension
  names, optional leading ``timestamp`` column (dropped). Files whose names
  contain ``anomal``/``ood``/``fail`` are labeled 1; a ``labels.json``
  (``{"filename": 0|1}``) in the directory overrides this.
- **Directory of .npy files** — one ``[T, D]`` array per sequence; optional
  ``<name>_actions.npy`` per sequence; ``dim_names.json`` and ``labels.json``
  are honored.
- **Single .npz** — either ragged (``seq_00000``, ``seq_00001``, … with
  optional ``act_%05d`` twins) or dense 3D ``data``/``observations``
  ``[N, T, D]`` with optional ``actions`` ``[N, T, A]``; optional
  ``dim_names`` (array of str) and ``labels`` (``[N]``).
- **Single .h5/.hdf5** — datasets ``observations`` and optionally ``actions``
  shaped ``[N, T, D]`` or ``[T, N, D]`` (auto-detected; Isaac Lab collection
  uses time-major). Requires ``h5py``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SequenceData:
    """In-memory dataset: ragged sequences with optional actions and labels."""

    obs: list[np.ndarray]  # each [T_i, D]
    actions: list[np.ndarray] | None = None  # each [T_i, A]
    labels: list[int] | None = None  # 0 nominal / 1 anomalous
    dim_names: list[str] = field(default_factory=list)
    names: list[str] = field(default_factory=list)  # per-sequence identifiers

    def __post_init__(self):
        if not self.dim_names and self.obs:
            self.dim_names = [f"dim_{i}" for i in range(self.obs[0].shape[1])]
        if not self.names:
            self.names = [f"seq_{i:05d}" for i in range(len(self.obs))]

    def __len__(self) -> int:
        return len(self.obs)

    @property
    def obs_dim(self) -> int:
        return self.obs[0].shape[1] if self.obs else 0

    @property
    def action_dim(self) -> int:
        return self.actions[0].shape[1] if self.actions else 0

    def subset(self, idx: list[int]) -> "SequenceData":
        return SequenceData(
            obs=[self.obs[i] for i in idx],
            actions=[self.actions[i] for i in idx] if self.actions else None,
            labels=[self.labels[i] for i in idx] if self.labels is not None else None,
            dim_names=self.dim_names,
            names=[self.names[i] for i in idx],
        )

    def nominal(self) -> "SequenceData":
        if self.labels is None:
            return self
        return self.subset([i for i, l in enumerate(self.labels) if l == 0])

    def anomalous(self) -> "SequenceData":
        if self.labels is None:
            return SequenceData([], dim_names=self.dim_names)
        return self.subset([i for i, l in enumerate(self.labels) if l == 1])

    def split(
        self, val_fraction: float = 0.1, test_fraction: float = 0.1, seed: int = 0
    ) -> tuple["SequenceData", "SequenceData", "SequenceData"]:
        """Random per-sequence train/val/test split."""
        n = len(self)
        order = np.random.default_rng(seed).permutation(n)
        n_test = int(round(n * test_fraction))
        n_val = int(round(n * val_fraction))
        test = order[:n_test].tolist()
        val = order[n_test : n_test + n_val].tolist()
        train = order[n_test + n_val :].tolist()
        if not train:
            raise ValueError(f"Split left no training sequences (N={n}).")
        return self.subset(train), self.subset(val), self.subset(test)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_ANOMALY_NAME_RE = re.compile(r"anomal|ood|fail", re.IGNORECASE)


def _label_from_name(name: str, overrides: dict[str, int]) -> int | None:
    for key, val in overrides.items():
        if Path(key).stem == Path(name).stem:
            return int(val)
    return 1 if _ANOMALY_NAME_RE.search(name) else None


def _load_csv(path: Path) -> tuple[np.ndarray, list[str]]:
    with open(path) as f:
        header = f.readline().strip().split(",")
    has_header = not all(_is_float(h) for h in header)
    data = np.loadtxt(path, delimiter=",", skiprows=1 if has_header else 0, ndmin=2)
    names = header if has_header else [f"dim_{i}" for i in range(data.shape[1])]
    if names and names[0].lower() in ("timestamp", "time", "t"):
        data, names = data[:, 1:], names[1:]
    return data.astype(np.float32), names


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def load_sequences(path: str | Path) -> SequenceData:
    """Load a dataset from any supported format (see module docstring)."""
    path = Path(path)
    if path.is_dir():
        return _load_dir(path)
    if path.suffix == ".npz":
        return _load_npz(path)
    if path.suffix in (".h5", ".hdf5"):
        return _load_h5(path)
    if path.suffix == ".csv":
        obs, names = _load_csv(path)
        return SequenceData(obs=[obs], dim_names=names, names=[path.stem])
    raise ValueError(f"Unsupported dataset path: {path}")


def _load_dir(path: Path) -> SequenceData:
    overrides = {}
    labels_file = path / "labels.json"
    if labels_file.exists():
        overrides = json.loads(labels_file.read_text())
    dim_names: list[str] = []
    dim_names_file = path / "dim_names.json"
    if dim_names_file.exists():
        dim_names = json.loads(dim_names_file.read_text())

    obs, actions, labels, names = [], [], [], []
    files = sorted(p for p in path.iterdir() if p.suffix in (".csv", ".npy"))
    files = [p for p in files if not p.stem.endswith("_actions")]
    if not files:
        raise ValueError(f"No .csv/.npy sequence files found in {path}")
    any_label = False
    for p in files:
        if p.suffix == ".csv":
            arr, csv_names = _load_csv(p)
            if not dim_names:
                dim_names = csv_names
        else:
            arr = np.load(p).astype(np.float32)
        obs.append(arr)
        act_file = p.with_name(p.stem + "_actions.npy")
        actions.append(np.load(act_file).astype(np.float32) if act_file.exists() else None)
        lab = _label_from_name(p.name, overrides)
        any_label = any_label or lab is not None
        labels.append(0 if lab is None else lab)
        names.append(p.stem)

    have_actions = all(a is not None for a in actions)
    return SequenceData(
        obs=obs,
        actions=actions if have_actions else None,
        labels=labels if any_label else None,
        dim_names=dim_names,
        names=names,
    )


def _load_npz(path: Path) -> SequenceData:
    data = np.load(path, allow_pickle=False)
    dim_names = [str(s) for s in data["dim_names"]] if "dim_names" in data else []
    labels = data["labels"].astype(int).tolist() if "labels" in data else None
    seq_keys = sorted(k for k in data.files if k.startswith("seq_"))
    if seq_keys:
        obs = [data[k].astype(np.float32) for k in seq_keys]
        act_keys = [k.replace("seq_", "act_") for k in seq_keys]
        actions = None
        if all(k in data.files for k in act_keys):
            actions = [data[k].astype(np.float32) for k in act_keys]
        return SequenceData(obs, actions, labels, dim_names, [k for k in seq_keys])
    key = "data" if "data" in data.files else "observations"
    dense = data[key].astype(np.float32)  # [N, T, D]
    obs = list(dense)
    actions = list(data["actions"].astype(np.float32)) if "actions" in data.files else None
    return SequenceData(obs, actions, labels, dim_names)


def _load_h5(path: Path) -> SequenceData:
    import h5py

    with h5py.File(path, "r") as f:
        obs = np.asarray(f["observations"], dtype=np.float32)
        actions = np.asarray(f["actions"], dtype=np.float32) if "actions" in f else None
    # Isaac Lab collection is time-major [T, N, D]; assume the smaller of the
    # first two axes is the sequence axis N.
    if obs.ndim != 3:
        raise ValueError(f"HDF5 observations must be 3D, got {obs.shape}")
    if obs.shape[0] > obs.shape[1]:  # [T, N, D] → [N, T, D]
        obs = obs.transpose(1, 0, 2)
        if actions is not None:
            actions = actions.transpose(1, 0, 2)
    return SequenceData(list(obs), list(actions) if actions is not None else None)


# ---------------------------------------------------------------------------
# Normalization + windowing
# ---------------------------------------------------------------------------


@dataclass
class NormStats:
    """Per-dimension statistics computed on nominal training data. ``min`` /
    ``max`` additionally drive the deployment range gate."""

    mean: np.ndarray
    std: np.ndarray
    min: np.ndarray
    max: np.ndarray

    @classmethod
    def from_data(cls, data: SequenceData) -> "NormStats":
        flat = np.concatenate(data.obs, axis=0)
        std = flat.std(axis=0)
        std[std < 1e-6] = 1e-6
        return cls(flat.mean(axis=0), std, flat.min(axis=0), flat.max(axis=0))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def to_dict(self) -> dict:
        return {k: getattr(self, k).tolist() for k in ("mean", "std", "min", "max")}

    @classmethod
    def from_dict(cls, d: dict) -> "NormStats":
        return cls(**{k: np.asarray(d[k], dtype=np.float32) for k in ("mean", "std", "min", "max")})


def make_windows(
    data: SequenceData,
    seq_len: int,
    stats: NormStats,
    dynamics: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Cut sequences into non-overlapping normalized windows.

    Returns ``(inputs, targets)``:

    - reconstruction: inputs ``[M, seq_len, D]``, targets = inputs.
    - dynamics: inputs ``[M, seq_len, D+A]`` (normalized obs ‖ raw action at
      ``t``), targets ``[M, seq_len, D]`` (normalized obs at ``t+1``).
    """
    xs, ys = [], []
    for i, o in enumerate(data.obs):
        o_n = stats.normalize(o)
        if dynamics:
            if data.actions is None:
                raise ValueError("Forward-dynamics target requires action arrays.")
            a = data.actions[i]
            inp = np.concatenate([o_n[:-1], a[:-1]], axis=1)
            tgt = o_n[1:]
        else:
            inp = tgt = o_n
        n_win = len(inp) // seq_len
        if n_win == 0:
            continue
        xs.append(inp[: n_win * seq_len].reshape(n_win, seq_len, -1))
        ys.append(tgt[: n_win * seq_len].reshape(n_win, seq_len, -1))
    if not xs:
        raise ValueError(
            f"No sequence is at least seq_len={seq_len} steps long — "
            "use shorter windows or longer sequences."
        )
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)
