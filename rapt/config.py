"""Configuration for the RAPT model, training, and detector."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class RaptConfig:
    """Model + training + detection configuration.

    Defaults match the final RAPT configuration from the paper (bottleneck
    reconstruction, probabilistic decoder, GRU recurrence, residual blocks,
    25% latent compression).
    """

    # --- data ---
    obs_dim: int = 0
    action_dim: int = 0  # only used when train_dynamics=True
    seq_len: int = 50  # training-unroll window length
    dt: float = 0.02  # timestep in seconds (50 Hz in the paper)
    dim_names: list[str] = field(default_factory=list)

    # --- architecture ---
    embed_dim: int = 256
    num_blocks: int = 4  # encoder blocks (= decoder blocks)
    compression_ratio: float = 0.25  # bottleneck removes this fraction of dims
    dropout: float = 0.0
    use_residual: bool = True
    use_probabilistic: bool = True
    use_temporal: bool = True
    reconstruction_type: str = "bottleneck"  # "bottleneck" | "masked"
    # False: reconstruct o_t from o_t (default). True: forward-dynamics —
    # predict o_{t+1} from (o_t, a_t); best for command-conditioned locomotion.
    train_dynamics: bool = False

    # --- training ---
    lr: float = 1e-3
    batch_size: int = 256
    num_epochs: int = 100
    weight_decay: float = 1e-2
    noise_scale: float = 0.01  # denoising: Gaussian noise added to inputs
    grad_clip: float = 1.0
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 0

    # --- detection gates ---
    # per-dimension gate: loss_i > cal_max_i + k_dim * (cal_max_i - cal_median_i)
    k_dim: float = 2.0
    # global gate: mean_loss > max(cal_mean) + k_global * std(cal_mean)
    k_global: float = 3.0
    # range gate: obs outside [min - buffer*range, max + buffer*range]
    range_buffer: float = 1.0
    use_range_gate: bool = True  # "Hybrid" in the paper; False = "Model Only"

    # --- saliency ---
    saliency_window: int = 200  # H: history steps for IG-BPTT (4 s @ 50 Hz)
    ig_steps: int = 50  # Riemann steps for integrated gradients
    top_k: int = 10  # salient dimensions kept for diagnosis

    @property
    def input_dim(self) -> int:
        return self.obs_dim + (self.action_dim if self.train_dynamics else 0)

    @property
    def bottleneck_dim(self) -> int:
        return int(self.embed_dim * (1.0 - self.compression_ratio))

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "RaptConfig":
        data = json.loads(Path(path).read_text())
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)
