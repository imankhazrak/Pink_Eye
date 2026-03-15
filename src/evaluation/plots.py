"""Plotting utilities for training/evaluation artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


def save_training_curves(history: Dict[str, List[float]], output_path: Path) -> None:
    """Save train/validation loss and F1 curves for one fold."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.get("train_loss", []), label="train_loss")
    plt.plot(epochs, history.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.get("val_f1_pinkeye", []), label="val_f1_pinkeye")
    plt.plot(epochs, history.get("val_recall_pinkeye", []), label="val_recall_pinkeye")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Curves")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path, title: str) -> None:
    """Save ROC curve figure for a fold."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_confusion_matrix_plot(
    confusion_values: np.ndarray,
    output_path: Path,
    class_names: List[str],
    title: str,
) -> None:
    """Save confusion matrix heatmap using matplotlib only."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.imshow(confusion_values, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    threshold = confusion_values.max() / 2.0 if confusion_values.size else 0.0
    for i in range(confusion_values.shape[0]):
        for j in range(confusion_values.shape[1]):
            plt.text(
                j,
                i,
                f"{int(confusion_values[i, j])}",
                ha="center",
                va="center",
                color="white" if confusion_values[i, j] > threshold else "black",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
