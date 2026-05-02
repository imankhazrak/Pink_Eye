"""Train class-conditional DDPM on cropped eye dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from src.datasets.eye_dataset import EyeCropsDataset, IDX_TO_CLASS
from src.diffusion.ddpm import DDPM
from src.diffusion.ema import EMA
from src.diffusion.unet import ConditionalUNet
from src.utils.io import load_yaml
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DDPM for Step 2.5")
    parser.add_argument(
        "--config",
        type=str,
        default="/users/PCS0229/imankhazrak/Pink_Eye/configs/ddpm_train.yaml",
        help="Path to DDPM train YAML config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    seed_everything(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")

    dataset = EyeCropsDataset(
        root_dir=cfg["data"]["root_dir"],
        image_size=cfg["data"]["image_size"],
        class_dirs=cfg["data"].get("class_dirs"),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    model = ConditionalUNet(
        num_classes=cfg["model"]["num_classes"],
        in_channels=cfg["model"].get("in_channels", 3),
        out_channels=cfg["model"].get("out_channels", 3),
        base_ch=cfg["model"]["base_ch"],
        time_dim=cfg["model"]["time_dim"],
    ).to(device)
    ddpm = DDPM(
        model=model,
        timesteps=cfg["diffusion"]["timesteps"],
        beta_start=cfg["diffusion"].get("beta_start", 1e-4),
        beta_end=cfg["diffusion"].get("beta_end", 0.02),
        schedule=cfg["diffusion"].get("schedule", "cosine"),
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    ema = EMA(model, decay=cfg["train"]["ema_decay"])
    optimizer = AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    sample_dir = Path(cfg["output"]["sample_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    epochs = cfg["train"]["epochs"]
    save_every = cfg["train"]["save_every"]
    sample_every = cfg["train"]["sample_every"]
    grad_clip = cfg["train"].get("grad_clip", None)

    for epoch in range(1, epochs + 1):
        ddpm.train()
        running_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for images, labels, _ in pbar:
            images = images.to(device)
            labels = labels.to(device)
            t = torch.randint(0, ddpm.timesteps, (images.shape[0],), device=device).long()

            loss = ddpm.p_losses(images, t, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            ema.update(model)

            running_loss += loss.item()
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / max(len(loader), 1)
        print(f"[Epoch {epoch:03d}] loss={avg_loss:.6f}")

        if epoch % save_every == 0 or epoch == epochs:
            ckpt_path = ckpt_dir / f"ddpm_epoch{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.ema_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg,
                },
                ckpt_path,
            )

        if epoch % sample_every == 0 or epoch == epochs:
            ema_model = ema.ema_model.to(device).eval()
            sampler = DDPM(
                model=ema_model,
                timesteps=cfg["diffusion"]["timesteps"],
                beta_start=cfg["diffusion"].get("beta_start", 1e-4),
                beta_end=cfg["diffusion"].get("beta_end", 0.02),
                schedule=cfg["diffusion"].get("schedule", "cosine"),
            ).to(device)
            channels = cfg["model"].get("out_channels", 3)
            with torch.no_grad():
                for class_idx, class_name in IDX_TO_CLASS.items():
                    y = torch.full((cfg["output"]["sample_batch_size"],), class_idx, device=device, dtype=torch.long)
                    samples = sampler.sample(
                        image_size=cfg["data"]["image_size"],
                        batch_size=cfg["output"]["sample_batch_size"],
                        channels=channels,
                        y=y,
                        device=device.type,
                        clamp=True,
                        denormalize=True,
                    )
                    grid = make_grid(samples, nrow=cfg["output"]["sample_grid_nrow"])
                    save_image(grid, sample_dir / f"epoch{epoch:03d}_{class_name}.png")

    print("Training finished.")


if __name__ == "__main__":
    main()
