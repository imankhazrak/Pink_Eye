"""Training loop, loss functions, and fold-level fitting logic."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import WeightedRandomSampler

from src.evaluation.metrics import compute_binary_metrics


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional class weighting."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = None,
        weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, reduction="none", weight=self.weight)
        p_t = torch.exp(-ce)
        focal = ((1 - p_t) ** self.gamma) * ce
        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
            focal = alpha_t * focal
        return focal.mean()


@dataclass
class FoldResult:
    """Container for fold outputs used by cross-validation orchestrator."""

    best_epoch: int
    checkpoint_path: str
    history: Dict[str, List[float]]
    best_metrics: Dict[str, float]
    y_true: np.ndarray
    y_prob: np.ndarray
    val_filenames: List[str]


def build_class_weights(labels: List[int], device: torch.device) -> torch.Tensor:
    """Compute inverse-frequency class weights from labels."""
    labels_arr = np.asarray(labels)
    counts = np.bincount(labels_arr, minlength=2).astype(np.float32)
    if np.any(counts == 0):
        counts = counts + 1e-6
    weights = counts.sum() / (2.0 * counts)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    return weights


def build_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    """Create weighted sampler to rebalance minority class during training."""
    labels_arr = np.asarray(labels)
    class_counts = np.bincount(labels_arr, minlength=2).astype(np.float32)
    class_weights = 1.0 / np.maximum(class_counts, 1.0)
    sample_weights = class_weights[labels_arr]
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def create_criterion(config, class_weights: torch.Tensor) -> nn.Module:
    """Create loss function based on config and imbalance strategy."""
    if config.loss_type == "focal":
        return FocalLoss(
            gamma=config.focal_gamma,
            alpha=config.focal_alpha,
            weight=class_weights,
        )
    if config.loss_type == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    raise ValueError(
        f"Unsupported loss_type '{config.loss_type}'. Use weighted_ce or focal."
    )


def _run_train_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    total = 0

    for images, labels, _ in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        running_loss += float(loss.item()) * bs
        total += bs

    return running_loss / max(total, 1)


def _run_val_epoch(
    model: nn.Module,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    running_loss = 0.0
    total = 0
    y_true: List[int] = []
    y_prob: List[float] = []
    val_filenames: List[str] = []

    with torch.no_grad():
        for images, labels, filenames in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)[:, 1]

            bs = images.size(0)
            running_loss += float(loss.item()) * bs
            total += bs

            y_true.extend(labels.cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())
            val_filenames.extend([str(f) for f in filenames])

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float32)
    metrics = compute_binary_metrics(y_true=y_true_arr, y_prob=y_prob_arr, threshold=0.5)

    return {
        "val_loss": running_loss / max(total, 1),
        "metrics": metrics,
        "y_true": y_true_arr,
        "y_prob": y_prob_arr,
        "val_filenames": val_filenames,
    }


def fit_one_fold(
    model: nn.Module,
    fold_index: int,
    train_loader,
    val_loader,
    config,
    device: torch.device,
    checkpoint_dir: Path,
    logger,
    train_labels: List[int],
) -> FoldResult:
    """Train and validate a model for one cross-validation fold."""
    class_weights = build_class_weights(train_labels, device=device)
    criterion = create_criterion(config=config, class_weights=class_weights)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    model.to(device)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_f1_pinkeye": [],
        "val_recall_pinkeye": [],
    }

    best_score = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    best_metrics: Dict[str, float] = {}
    best_y_true = None
    best_y_prob = None
    best_val_filenames: List[str] = []
    epochs_without_improvement = 0

    for epoch in range(1, config.num_epochs + 1):
        train_loss = _run_train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_out = _run_val_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        val_loss = float(val_out["val_loss"])
        metrics = val_out["metrics"]
        val_f1_pinkeye = float(metrics["f1_pinkeye"])
        val_recall_pinkeye = float(metrics["recall_pinkeye"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1_pinkeye"].append(val_f1_pinkeye)
        history["val_recall_pinkeye"].append(val_recall_pinkeye)

        logger.info(
            "[Fold %d] Epoch %d/%d | train_loss=%.4f val_loss=%.4f "
            "f1_pinkeye=%.4f recall_pinkeye=%.4f",
            fold_index,
            epoch,
            config.num_epochs,
            train_loss,
            val_loss,
            val_f1_pinkeye,
            val_recall_pinkeye,
        )

        if val_f1_pinkeye > (best_score + config.min_delta):
            best_score = val_f1_pinkeye
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = metrics
            best_y_true = val_out["y_true"]
            best_y_prob = val_out["y_prob"]
            best_val_filenames = val_out["val_filenames"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            logger.info(
                "[Fold %d] Early stopping at epoch %d (best_epoch=%d, best_f1_pinkeye=%.4f)",
                fold_index,
                epoch,
                best_epoch,
                best_score,
            )
            break

    model.load_state_dict(best_state)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{config.model_name}_fold{fold_index}_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "config": vars(config),
        },
        checkpoint_path,
    )

    if best_y_true is None or best_y_prob is None:
        raise RuntimeError("No best validation state found during training.")

    return FoldResult(
        best_epoch=best_epoch,
        checkpoint_path=str(checkpoint_path),
        history=history,
        best_metrics=best_metrics,
        y_true=best_y_true,
        y_prob=best_y_prob,
        val_filenames=best_val_filenames,
    )
