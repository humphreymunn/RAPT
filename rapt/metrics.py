"""Evaluation metrics used in the RAPT paper.

- AUROC over per-step (or per-episode) anomaly scores.
- Safety Score: TPR at a fixed episode-level FPR (0.5% in the paper).
- PADD: Penalized Average Detection Delay — time from OOD onset to first
  trigger, with missed detections penalized by the episode horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC of anomaly ``scores`` against binary ``labels`` (1 = OOD)."""
    labels = np.asarray(labels)
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, np.asarray(scores)))


def episode_scores(step_scores: Sequence[np.ndarray]) -> np.ndarray:
    """Reduce per-step scores to one score per episode (max over time)."""
    return np.array([float(np.max(s)) for s in step_scores])


def safety_score(
    episode_scores_nominal: np.ndarray,
    episode_scores_ood: np.ndarray,
    fpr: float = 0.005,
) -> float:
    """TPR at a fixed episode-level FPR (paper default 0.5%).

    The detection threshold is set to the (1 - fpr) quantile of the nominal
    episode scores; the returned value is the fraction of OOD episodes whose
    score exceeds it.
    """
    nominal = np.asarray(episode_scores_nominal, dtype=np.float64)
    ood = np.asarray(episode_scores_ood, dtype=np.float64)
    if len(nominal) == 0 or len(ood) == 0:
        return float("nan")
    thresh = np.quantile(nominal, 1.0 - fpr)
    return float(np.mean(ood > thresh))


def padd(
    trigger_steps: Sequence[int | None],
    onset_steps: Sequence[int],
    horizons: Sequence[int],
    dt: float = 1.0,
) -> float:
    """Penalized Average Detection Delay.

    For each episode, the delay is ``trigger - onset`` (clipped at 0 for
    triggers before onset); a missed detection (``trigger is None`` or before
    onset with no later trigger) incurs the full episode horizon. ``dt``
    converts steps to seconds.
    """
    delays = []
    for trig, onset, horizon in zip(trigger_steps, onset_steps, horizons):
        if trig is None:
            delays.append(horizon)
        else:
            delays.append(max(0, trig - onset))
    return float(np.mean(np.asarray(delays, dtype=np.float64) * dt))


@dataclass
class DetectionReport:
    """Confusion counts at the calibrated operating point (real-world eval)."""

    tp: int = 0
    fn: int = 0
    tn: int = 0
    fp: int = 0

    @property
    def tpr(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else float("nan")

    @property
    def fpr(self) -> float:
        d = self.fp + self.tn
        return self.fp / d if d else float("nan")

    def __str__(self) -> str:
        return (
            f"TPR {100 * self.tpr:.1f}% | FPR {100 * self.fpr:.1f}% | "
            f"TP {self.tp} FN {self.fn} TN {self.tn} FP {self.fp}"
        )


def confusion_from_flags(
    episode_flagged: Sequence[bool], episode_labels: Sequence[int]
) -> DetectionReport:
    """Episode-level confusion counts from binary detector decisions."""
    rep = DetectionReport()
    for flagged, label in zip(episode_flagged, episode_labels):
        if label:
            rep.tp += int(flagged)
            rep.fn += int(not flagged)
        else:
            rep.fp += int(flagged)
            rep.tn += int(not flagged)
    return rep
