"""Isolation Forest baseline (100 trees, contamination 0.01, as in the paper)."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationForestDetector:
    def __init__(self, n_estimators: int = 100, contamination: float = 0.01, seed: int = 0):
        self.model = IsolationForest(
            n_estimators=n_estimators, contamination=contamination, random_state=seed
        )

    def fit(self, sequences: list[np.ndarray], log_fn=print) -> None:
        flat = np.concatenate(sequences, 0)
        self.model.fit(flat)
        log_fn(f"  Isolation Forest fit on {len(flat)} samples.")

    def score(self, seq: np.ndarray) -> np.ndarray:
        return -self.model.decision_function(seq)
