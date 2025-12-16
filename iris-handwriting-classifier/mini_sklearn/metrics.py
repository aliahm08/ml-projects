from __future__ import annotations

import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Return a simple text report of precision/recall/f1 per class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    headers = ("precision", "recall", "f1-score", "support")
    line_fmt = "{label:>12} {precision:>9.2f} {recall:>6.2f} {f1:>9.2f} {support:>9}"
    lines = ["              " + " ".join(f"{h:>9}" for h in headers)]

    precision_vals: list[float] = []
    recall_vals: list[float] = []
    f1_vals: list[float] = []
    supports: list[int] = []

    for label in labels:
        mask_true = y_true == label
        mask_pred = y_pred == label
        tp = np.sum(mask_true & mask_pred)
        fp = np.sum(~mask_true & mask_pred)
        fn = np.sum(mask_true & ~mask_pred)
        support = int(mask_true.sum())

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        precision_vals.append(precision)
        recall_vals.append(recall)
        f1_vals.append(f1)
        supports.append(support)

        lines.append(
            line_fmt.format(
                label=str(label),
                precision=precision,
                recall=recall,
                f1=f1,
                support=support,
            )
        )

    total_support = int(np.sum(supports))
    accuracy = accuracy_score(y_true, y_pred)
    lines.append("")
    lines.append(f"accuracy{'':>24} {accuracy:>9.2f} {total_support:>9}")

    avg_fmt = "{label:>12} {precision:>9.2f} {recall:>6.2f} {f1:>9.2f} {support:>9}"
    macro = avg_fmt.format(
        label="macro avg",
        precision=np.mean(precision_vals) if precision_vals else 0.0,
        recall=np.mean(recall_vals) if recall_vals else 0.0,
        f1=np.mean(f1_vals) if f1_vals else 0.0,
        support=total_support,
    )
    weighted_precision = _weighted_average(precision_vals, supports)
    weighted_recall = _weighted_average(recall_vals, supports)
    weighted_f1 = _weighted_average(f1_vals, supports)
    weighted = avg_fmt.format(
        label="weighted avg",
        precision=weighted_precision,
        recall=weighted_recall,
        f1=weighted_f1,
        support=total_support,
    )
    lines.append(macro)
    lines.append(weighted)
    return "\n".join(lines)


def _weighted_average(values: list[float], weights: list[int]) -> float:
    if not values or not weights:
        return 0.0
    total = np.sum(weights)
    if total == 0:
        return 0.0
    return float(np.sum(np.asarray(values) * np.asarray(weights) / total))
