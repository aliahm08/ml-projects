from __future__ import annotations

import numpy as np


class LogisticRegression:
    """A very small multinomial logistic regression for educational usage."""

    def __init__(
        self,
        max_iter: int = 200,
        learning_rate: float = 0.1,
        batch_size: int = 256,
        multi_class: str = "auto",
        solver: str = "lbfgs",
        verbose: int = 0,
        random_state: int = 42,
    ) -> None:
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.multi_class = multi_class
        self.solver = solver
        self.verbose = verbose
        self.random_state = random_state
        self.classes_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        features = np.asarray(X, dtype=np.float32)
        labels = np.asarray(y, dtype=np.int64)
        classes, y_indices = np.unique(labels, return_inverse=True)
        self.classes_ = classes
        n_samples, n_features = features.shape
        n_classes = classes.size

        rng = np.random.default_rng(self.random_state)
        self.coef_ = rng.normal(scale=0.01, size=(n_classes, n_features)).astype(
            np.float32
        )
        self.intercept_ = np.zeros(n_classes, dtype=np.float32)

        y_one_hot = np.eye(n_classes, dtype=np.float32)[y_indices]
        batch_size = max(1, min(self.batch_size, n_samples))

        for epoch in range(self.max_iter):
            order = rng.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                batch_idx = order[start : start + batch_size]
                X_batch = features[batch_idx]
                y_batch = y_one_hot[batch_idx]

                logits = self._logits(X_batch)
                probs = _softmax(logits)
                error = probs - y_batch

                grad_w = error.T @ X_batch / X_batch.shape[0]
                grad_b = error.mean(axis=0)

                self.coef_ -= self.learning_rate * grad_w
                self.intercept_ -= self.learning_rate * grad_b

            if self.verbose and (epoch % 10 == 0 or epoch == self.max_iter - 1):
                preds = self.predict(features)
                acc = float((preds == labels).mean())
                print(f"[mini_sklearn] epoch={epoch} accuracy={acc:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits = self._logits(np.asarray(X, dtype=np.float32))
        indices = np.argmax(logits, axis=1)
        return self.classes_[indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self._logits(np.asarray(X, dtype=np.float32))
        return _softmax(logits)

    def _logits(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Call fit before predict.")
        return X @ self.coef_.T + self.intercept_


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z, dtype=np.float32)
    return exp / exp.sum(axis=1, keepdims=True)
