"""Lightweight fallbacks that mimic a subset of scikit-learn APIs."""

from . import linear_model, metrics, preprocessing  # noqa: F401

__all__ = ["linear_model", "metrics", "preprocessing"]
