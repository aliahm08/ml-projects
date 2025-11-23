from __future__ import annotations

import numpy as np


class StandardScaler:
    """Minimal StandardScaler compatible with scikit-learn's interface."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        array = np.asarray(X, dtype=np.float32)
        self.mean_ = array.mean(axis=0)
        self.scale_ = array.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fit before calling transform.")
        array = np.asarray(X, dtype=np.float32)
        return (array - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
