"""RAPT: Model-Predictive Out-of-Distribution Detection and Failure Diagnosis.

Typical use:

    from rapt import RaptConfig, RaptSystem, load_sequences, train_rapt

    data = load_sequences("my_dataset/")            # N ragged [T_i, D] sequences
    train, val, test = data.nominal().split()
    cfg = RaptConfig(obs_dim=data.obs_dim, dim_names=data.dim_names)
    model, stats, history = train_rapt(cfg, train, val)
    system = RaptSystem(cfg, model, stats, history=history)
    system.calibrate(val)                            # or a real-world nominal run
    system.save("checkpoints/my_rapt")               # .pt + .onnx + stats + gates

    detector = system.detector()
    result = detector.step(obs_t)                    # streaming, per control step
    if result.is_anomaly:
        sal = system.attribute(recent_obs_window)
"""

from .checkpoint import RaptSystem
from .config import RaptConfig
from .data import NormStats, SequenceData, load_sequences, make_windows
from .detector import Calibration, RaptDetector, calibrate, risk_trajectory, score_sequence
from .metrics import DetectionReport, auroc, episode_scores, padd, safety_score
from .model import RaptModel, nll_loss, nll_per_dim
from .saliency import SaliencyResult, integrated_gradients_bptt, plot_saliency_heatmap
from .train import train_rapt

__version__ = "1.0.0"

__all__ = [
    "RaptConfig",
    "RaptModel",
    "RaptSystem",
    "RaptDetector",
    "SequenceData",
    "NormStats",
    "Calibration",
    "SaliencyResult",
    "DetectionReport",
    "load_sequences",
    "make_windows",
    "train_rapt",
    "calibrate",
    "score_sequence",
    "risk_trajectory",
    "integrated_gradients_bptt",
    "plot_saliency_heatmap",
    "nll_loss",
    "nll_per_dim",
    "auroc",
    "episode_scores",
    "safety_score",
    "padd",
]
