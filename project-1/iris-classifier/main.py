"""Train a simple logistic regression classifier on the TensorFlow MNIST dataset."""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Tuple
from urllib.request import urlretrieve

import numpy as np

try:  # Prefer the real scikit-learn modules when available.
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import StandardScaler
    USING_MINI_SKLEARN = False
except ImportError:  # pragma: no cover - fallback to lightweight stand-ins.
    from mini_sklearn.linear_model import LogisticRegression
    from mini_sklearn.metrics import accuracy_score, classification_report
    from mini_sklearn.preprocessing import StandardScaler

    USING_MINI_SKLEARN = True

try:
    from tensorflow.keras.datasets import mnist as tf_mnist  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    tf_mnist = None

ARTIFACT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ARTIFACT_DIR / "model.pkl"
SCALER_PATH = ARTIFACT_DIR / "scaler.pkl"
DATA_DIR = ARTIFACT_DIR / "data"
TF_MNIST_ARCHIVE = DATA_DIR / "mnist.npz"
TF_MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
TRAIN_FRACTION = float(os.environ.get("TRAIN_FRACTION", "1.0"))
TEST_FRACTION = float(os.environ.get("TEST_FRACTION", "1.0"))
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Download MNIST via TensorFlow (or its hosted archive) and return arrays."""
    (X_train, y_train), (X_test, y_test) = download_tensorflow_mnist()

    # Flatten 28x28 images into 784-length vectors and cast to float32.
    X_train = X_train.reshape((X_train.shape[0], -1)).astype("float32")
    X_test = X_test.reshape((X_test.shape[0], -1)).astype("float32")

    rng = np.random.default_rng(RANDOM_SEED)
    X_train, y_train = _subset_split(
        X_train, y_train, TRAIN_FRACTION, "TRAIN_FRACTION", rng
    )
    X_test, y_test = _subset_split(
        X_test, y_test, TEST_FRACTION, "TEST_FRACTION", rng
    )

    return X_train, X_test, y_train, y_test


def download_tensorflow_mnist():
    """Use tf.keras if available; otherwise fetch the official TensorFlow archive."""
    if tf_mnist is not None:
        return tf_mnist.load_data()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not TF_MNIST_ARCHIVE.exists():
        print("Downloading MNIST archive from TensorFlow storage...")
        urlretrieve(TF_MNIST_URL, TF_MNIST_ARCHIVE)

    with np.load(TF_MNIST_ARCHIVE) as data:
        X_train = data["x_train"]
        y_train = data["y_train"]
        X_test = data["x_test"]
        y_test = data["y_test"]
    return (X_train, y_train), (X_test, y_test)


def scale_data(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Standardize feature values with zero mean and unit variance."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Fit a multinomial logistic regression classifier."""
    model = LogisticRegression(
        max_iter=200,
        solver="lbfgs",
        verbose=1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model: LogisticRegression, X_test: np.ndarray, y_test: np.ndarray) -> float:
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification report:\n")
    print(classification_report(y_test, preds))
    return acc


def save_artifacts(model: LogisticRegression, scaler: StandardScaler) -> None:
    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(model, model_file)
    with SCALER_PATH.open("wb") as scaler_file:
        pickle.dump(scaler, scaler_file)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved scaler to {SCALER_PATH}")


def _subset_split(
    X: np.ndarray,
    y: np.ndarray,
    fraction: float,
    fraction_name: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Shuffle and subsample arrays while keeping every class represented."""
    if not 0 < fraction <= 1:
        raise ValueError(f"{fraction_name} must be in (0, 1].")

    limit = int(fraction * X.shape[0])
    if limit < 1:
        raise ValueError(
            f"{fraction_name}={fraction} results in 0 samples. Increase the fraction."
        )

    if limit == X.shape[0]:
        return X, y

    indices = rng.choice(X.shape[0], size=limit, replace=False)
    return X[indices], y[indices]


def main() -> None:
    print("1. Loading MNIST data via TensorFlow...")
    X_train, X_test, y_train, y_test = load_data()

    print("2. Scaling pixels with StandardScaler...")
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    trainer = "scikit-learn" if not USING_MINI_SKLEARN else "mini_sklearn fallback"
    print(f"3. Training logistic regression classifier ({trainer})...")
    model = train_model(X_train_scaled, y_train)

    print("4. Evaluating on the hold-out test set...")
    evaluate(model, X_test_scaled, y_test)

    print("5. Saving artifacts with pickle...")
    save_artifacts(model, scaler)


if __name__ == "__main__":
    main()
