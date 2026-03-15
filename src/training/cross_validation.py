"""Cross-validation orchestration for Step 2 experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import (
    ClassMapping,
    EyeCropsDataset,
    build_metadata_table,
    get_train_transforms,
    get_val_transforms,
    make_fold_indices,
)
from src.evaluation.metrics import summarize_metrics, to_metrics_frame
from src.evaluation.plots import (
    save_confusion_matrix_plot,
    save_roc_curve,
    save_training_curves,
)
from src.explainability.gradcam import save_gradcam_examples
from src.models import build_model, get_gradcam_target_layer
from src.training.trainer import build_weighted_sampler, fit_one_fold


def _resolve_dataset_path(project_root: Path, dataset_root: str) -> Path:
    path = Path(dataset_root)
    if not path.is_absolute():
        path = project_root / dataset_root
    return path


def run_stratified_cross_validation(config, output_dirs: Dict[str, Path], logger) -> Dict[str, str]:
    """Run full stratified K-fold experiment and persist all artifacts."""
    project_root = Path(__file__).resolve().parents[2]
    dataset_root = _resolve_dataset_path(project_root, config.dataset_root)

    class_mapping = ClassMapping(
        healthy=config.healthy_dir_name,
        pinkeye=config.pinkeye_dir_name,
        healthy_label=0,
        pinkeye_label=1,
    )
    metadata = build_metadata_table(dataset_root=dataset_root, class_mapping=class_mapping)
    labels = metadata["label"].astype(int).tolist()
    fold_indices = make_fold_indices(labels, config.num_folds, config.random_seed)

    logger.info(
        "Loaded dataset from %s | total=%d | healthy=%d | pinkeye=%d",
        dataset_root,
        len(metadata),
        int((metadata["label"] == 0).sum()),
        int((metadata["label"] == 1).sum()),
    )

    all_fold_metrics: List[Dict[str, float]] = []
    all_prediction_rows: List[Dict[str, object]] = []
    device = torch.device(
        config.device if config.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    for fold_id, (train_idx, val_idx) in enumerate(fold_indices, start=1):
        logger.info("Starting fold %d/%d", fold_id, config.num_folds)
        fold_logger = logger

        train_records = metadata.iloc[train_idx].reset_index(drop=True)
        val_records = metadata.iloc[val_idx].reset_index(drop=True)

        if config.augment_minority_only:
            train_dataset = EyeCropsDataset(
                records=train_records,
                transform=get_val_transforms(config.image_size),
                minority_transform=get_train_transforms(config.image_size),
                minority_label=config.minority_label,
            )
            logger.info(
                "Fold %d: class-conditional augmentation enabled (minority_label=%d)",
                fold_id,
                config.minority_label,
            )
        else:
            train_dataset = EyeCropsDataset(
                records=train_records,
                transform=get_train_transforms(config.image_size),
            )
        val_dataset = EyeCropsDataset(
            records=val_records,
            transform=get_val_transforms(config.image_size),
        )

        train_labels = train_records["label"].astype(int).tolist()
        sampler = build_weighted_sampler(train_labels) if config.use_weighted_sampler else None

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        model = build_model(
            model_name=config.model_name,
            pretrained=config.pretrained,
            num_classes=2,
        )

        fold_result = fit_one_fold(
            model=model,
            fold_index=fold_id,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            checkpoint_dir=output_dirs["models"],
            logger=fold_logger,
            train_labels=train_labels,
        )

        row = {"model": config.model_name, "fold": fold_id}
        row.update(fold_result.best_metrics)
        all_fold_metrics.append(row)

        # Persist prediction-level details for reproducibility.
        for filename, y_t, y_p in zip(
            fold_result.val_filenames,
            fold_result.y_true.tolist(),
            fold_result.y_prob.tolist(),
        ):
            all_prediction_rows.append(
                {
                    "model": config.model_name,
                    "fold": fold_id,
                    "filename": filename,
                    "y_true": int(y_t),
                    "y_prob_pinkeye": float(y_p),
                    "y_pred": int(y_p >= 0.5),
                }
            )

        # Plot fold-level artifacts
        save_training_curves(
            history=fold_result.history,
            output_path=output_dirs["plots"] / f"{config.model_name}_fold{fold_id}_curves.png",
        )
        save_roc_curve(
            y_true=fold_result.y_true,
            y_prob=fold_result.y_prob,
            output_path=output_dirs["plots"] / f"{config.model_name}_fold{fold_id}_roc.png",
            title=f"{config.model_name} Fold {fold_id} ROC",
        )
        conf = np.array(
            [
                [fold_result.best_metrics["tn"], fold_result.best_metrics["fp"]],
                [fold_result.best_metrics["fn"], fold_result.best_metrics["tp"]],
            ]
        )
        save_confusion_matrix_plot(
            confusion_values=conf,
            output_path=output_dirs["plots"] / f"{config.model_name}_fold{fold_id}_confusion.png",
            class_names=["healthy", "pinkeye"],
            title=f"{config.model_name} Fold {fold_id} Confusion Matrix",
        )

        # Grad-CAM for CNN baselines.
        if config.save_gradcam:
            target_layer = get_gradcam_target_layer(config.model_name, model)
            _ = save_gradcam_examples(
                model=model,
                target_layer=target_layer,
                dataloader=val_loader,
                device=device,
                output_dir=output_dirs["explainability"]
                / config.model_name
                / f"fold_{fold_id}",
                max_examples=config.gradcam_examples_per_fold,
            )

    metrics_df = to_metrics_frame(all_fold_metrics)
    summary_df = summarize_metrics(metrics_df)
    if not summary_df.empty:
        summary_df.insert(0, "model", config.model_name)

    predictions_df = pd.DataFrame(all_prediction_rows)

    per_fold_csv = output_dirs["metrics"] / f"{config.model_name}_fold_metrics.csv"
    summary_csv = output_dirs["metrics"] / f"{config.model_name}_summary_metrics.csv"
    predictions_csv = output_dirs["metrics"] / f"{config.model_name}_fold_predictions.csv"

    metrics_df.to_csv(per_fold_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    predictions_df.to_csv(predictions_csv, index=False)

    logger.info("Saved fold metrics: %s", per_fold_csv)
    logger.info("Saved summary metrics: %s", summary_csv)
    logger.info("Saved fold predictions: %s", predictions_csv)

    return {
        "per_fold_metrics_csv": str(per_fold_csv),
        "summary_metrics_csv": str(summary_csv),
        "predictions_csv": str(predictions_csv),
    }
