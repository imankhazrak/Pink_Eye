"""Sample class-conditional synthetic eye crops from trained DDPM."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from src.datasets.eye_dataset import IDX_TO_CLASS
from src.diffusion.ddpm import DDPM
from src.diffusion.unet import ConditionalUNet
from src.utils.io import load_yaml
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample from trained DDPM")
    parser.add_argument(
        "--config",
        type=str,
        default="/users/PCS0229/imankhazrak/Pink_Eye/configs/ddpm_sample.yaml",
        help="Path to DDPM sample YAML config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    seed_everything(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")

    model = ConditionalUNet(
        num_classes=cfg["model"]["num_classes"],
        in_channels=cfg["model"].get("in_channels", 3),
        out_channels=cfg["model"].get("out_channels", 3),
        base_ch=cfg["model"]["base_ch"],
        time_dim=cfg["model"]["time_dim"],
    ).to(device)

    checkpoint = torch.load(cfg["checkpoint"]["path"], map_location=device)
    state_key = cfg["checkpoint"].get("state_key", "ema_state_dict")
    if state_key not in checkpoint:
        raise KeyError(f"Checkpoint missing '{state_key}'")
    model.load_state_dict(checkpoint[state_key])
    model.eval()

    ddpm = DDPM(
        model=model,
        timesteps=cfg["diffusion"]["timesteps"],
        beta_start=cfg["diffusion"].get("beta_start", 1e-4),
        beta_end=cfg["diffusion"].get("beta_end", 0.02),
        schedule=cfg["diffusion"].get("schedule", "cosine"),
    ).to(device)

    out_root = Path(cfg["output"]["dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    n_per_class = int(cfg["sample"]["num_per_class"])
    batch_size = int(cfg["sample"]["batch_size"])
    image_size = int(cfg["data"]["image_size"])

    channels = cfg["model"].get("out_channels", 3)
    first_batch = True

    with torch.no_grad():
        for class_idx, class_name in IDX_TO_CLASS.items():
            class_dir = out_root / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            saved = 0
            pbar = tqdm(total=n_per_class, desc=f"Sampling {class_name}", leave=False)
            while saved < n_per_class:
                cur_batch = min(batch_size, n_per_class - saved)
                y = torch.full((cur_batch,), class_idx, device=device, dtype=torch.long)
                samples = ddpm.sample(
                    image_size=image_size,
                    batch_size=cur_batch,
                    channels=channels,
                    y=y,
                    device=device.type,
                    clamp=True,
                    denormalize=True,
                )
                if first_batch:
                    print("samples min:", samples.min().item())
                    print("samples max:", samples.max().item())
                    print("samples mean:", samples.mean().item())
                    first_batch = False
                for i in range(cur_batch):
                    save_image(samples[i], class_dir / f"{class_name}_{saved + i + 1:05d}.png")
                saved += cur_batch
                pbar.update(cur_batch)
            pbar.close()

    print(f"Sampling finished. Saved to: {out_root}")


if __name__ == "__main__":
    main()
