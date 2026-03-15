"""Metric computation and aggregation utilities."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute core and class-specific binary classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "precision_pinkeye": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "recall_pinkeye": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "f1_pinkeye": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics


def to_metrics_frame(rows: List[Dict[str, float]]) -> pd.DataFrame:
    """Convert per-fold metric dictionaries to a dataframe."""
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Generate mean±std summary table over CV folds."""
    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "precision_pinkeye",
        "recall_pinkeye",
        "f1_pinkeye",
    ]

    rows = []
    for col in metric_cols:
        if col not in metrics_df.columns:
            continue
        values = metrics_df[col].astype(float)
        rows.append(
            {
                "metric": col,
                "mean": values.mean(),
                "std": values.std(ddof=1) if len(values) > 1 else 0.0,
                "mean_pm_std": f"{values.mean():.4f} ± "
                f"{(values.std(ddof=1) if len(values) > 1 else 0.0):.4f}",
            }
        )
    return pd.DataFrame(rows)
