"""Entrypoint for Step 2 pinkeye classification experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.training.cross_validation import run_stratified_cross_validation
from src.utils.config import TrainingConfig, save_run_config, set_global_seed
from src.utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Step 2 classifiers with stratified 5-fold CV."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["resnet18", "efficientnet", "cnn_transformer", "cnn_transformer_strong"],
        help="Model architecture.",
    )
    parser.add_argument("--dataset-root", type=str, default="outputs/eye_crops_classification")
    parser.add_argument("--healthy-dir-name", type=str, default="healthy")
    parser.add_argument("--pinkeye-dir-name", type=str, default="pink_eye")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument(
        "--loss-type",
        type=str,
        default="weighted_ce",
        choices=["weighted_ce", "focal"],
    )
    parser.add_argument("--use-weighted-sampler", action="store_true")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--save-gradcam", action="store_true", default=True)
    parser.add_argument("--no-gradcam", dest="save_gradcam", action="store_false")
    parser.add_argument("--gradcam-examples-per-fold", type=int, default=6)
    parser.add_argument(
        "--augment-minority-only",
        action="store_true",
        help="Apply training augmentations only to minority class samples.",
    )
    parser.add_argument(
        "--minority-label",
        type=int,
        default=1,
        help="Class label treated as minority for class-conditional augmentation.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainingConfig:
    model_name = args.model
    if model_name == "efficientnet":
        model_name = "efficientnet_b0"

    return TrainingConfig(
        model_name=model_name,
        dataset_root=args.dataset_root,
        healthy_dir_name=args.healthy_dir_name,
        pinkeye_dir_name=args.pinkeye_dir_name,
        image_size=args.image_size,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_folds=args.num_folds,
        num_workers=args.num_workers,
        random_seed=args.seed,
        patience=args.patience,
        min_delta=args.min_delta,
        device=args.device,
        output_root=args.output_root,
        loss_type=args.loss_type,
        use_weighted_sampler=args.use_weighted_sampler,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        pretrained=args.pretrained,
        save_gradcam=args.save_gradcam,
        gradcam_examples_per_fold=args.gradcam_examples_per_fold,
        augment_minority_only=args.augment_minority_only,
        minority_label=args.minority_label,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)

    project_root = Path(__file__).resolve().parent
    output_dirs = config.resolve_paths(project_root=project_root)
    set_global_seed(config.random_seed)

    log_file = output_dirs["logs"] / f"{config.model_name}_train.log"
    logger = get_logger(name=f"step2.{config.model_name}", log_file=log_file)
    logger.info("Starting Step 2 run with config: %s", vars(config))

    save_run_config(
        config=config,
        output_path=output_dirs["logs"] / f"{config.model_name}_run_config.json",
    )

    artifacts = run_stratified_cross_validation(
        config=config,
        output_dirs=output_dirs,
        logger=logger,
    )
    logger.info("Run complete. Artifacts: %s", artifacts)


if __name__ == "__main__":
    main()
