# DDPM Synthetic Generation Pipeline for Pink-Eye Eye Crops

This document provides a complete project structure and starter code for **Step 2.5: synthetic image generation using DDPM** before moving to Step 3. The goal is to train a DDPM on the cropped eye images and generate additional images, especially for the minority `pink_eye` class, so you can later test whether synthetic augmentation improves classification.

---

## 1. Recommended experimental goal

Use DDPM primarily for **class-conditional data generation** on the cropped eye dataset:

- `healthy`
- `pink_eye`

Because your dataset is imbalanced, the most practical first experiment is:

1. Train a **class-conditional DDPM** on the cropped eye images.
2. Generate synthetic images for the **minority class (`pink_eye`)**.
3. Optionally generate a smaller number for `healthy`.
4. Build a new dataset version such as:

```text
data/
  eye_crops_real/
    healthy/
    pink_eye/
  eye_crops_synthetic/
    healthy/
    pink_eye/
  eye_crops_augmented/
    healthy/
    pink_eye/
```

Then compare:

- real only
- real + synthetic minority balancing
- real + synthetic both classes

---

## 2. Project structure

```text
pinkeye_ddpm/
├── configs/
│   ├── ddpm_train.yaml
│   └── ddpm_sample.yaml
├── data/
│   ├── eye_crops_real/
│   │   ├── healthy/
│   │   └── pink_eye/
│   ├── eye_crops_synthetic/
│   │   ├── healthy/
│   │   └── pink_eye/
│   └── eye_crops_augmented/
│       ├── healthy/
│       └── pink_eye/
├── outputs/
│   ├── ddpm/
│   │   ├── checkpoints/
│   │   ├── samples/
│   │   ├── grids/
│   │   └── logs/
│   └── metadata/
│       ├── synthetic_images.csv
│       └── generation_summary.csv
├── src/
│   ├── datasets/
│   │   └── eye_dataset.py
│   ├── diffusion/
│   │   ├── unet.py
│   │   ├── ddpm.py
│   │   ├── scheduler.py
│   │   └── ema.py
│   ├── utils/
│   │   ├── seed.py
│   │   ├── image_utils.py
│   │   └── io.py
│   ├── train_ddpm.py
│   ├── sample_ddpm.py
│   ├── build_augmented_dataset.py
│   └── preview_samples.py
├── requirements.txt
└── README.md
```

---

## 3. Data assumptions

Your current cropped-eye classification dataset looks like:

```text
eye_crops/
  healthy/
  pink_eye/
```

Use exactly that as the DDPM training source.

Because DDPM training is harder than classification, start with:

- image size: **64x64** or **128x128**
- RGB images
- class-conditional generation

For your first run, I recommend:

- **64x64** for fast stability testing
- then **128x128** if results look promising

---

## 4. Code

## 4.1 `src/datasets/eye_dataset.py`

```python
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


CLASS_TO_IDX = {
    "healthy": 0,
    "pink_eye": 1,
}

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


class EyeCropsDataset(Dataset):
    def __init__(self, root_dir: str | Path, image_size: int = 64) -> None:
        self.root_dir = Path(root_dir)
        self.samples: List[Tuple[Path, int]] = []

        for class_name, label in CLASS_TO_IDX.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.glob("*")):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    self.samples.append((img_path, label))

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long), str(img_path)
```

---

## 4.2 `src/diffusion/unet.py`

```python
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / max(half_dim - 1, 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, class_dim: int | None = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.class_mlp = nn.Linear(class_dim, out_ch) if class_dim is not None else None
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, c_emb: torch.Tensor | None = None) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        if self.class_mlp is not None and c_emb is not None:
            h = h + self.class_mlp(c_emb)[:, :, None, None]
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        return h + self.res_conv(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, class_dim: int | None = None) -> None:
        super().__init__()
        self.block = ResBlock(in_ch, out_ch, time_dim, class_dim)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t_emb, c_emb=None):
        x = self.block(x, t_emb, c_emb)
        skip = x
        x = self.pool(x)
        return x, skip


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, time_dim: int, class_dim: int | None = None) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block = ResBlock(in_ch + skip_ch, out_ch, time_dim, class_dim)

    def forward(self, x, skip, t_emb, c_emb=None):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x, t_emb, c_emb)
        return x


class ConditionalUNet(nn.Module):
    def __init__(self, num_classes: int = 2, base_ch: int = 64, time_dim: int = 256) -> None:
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.class_embed = nn.Embedding(num_classes, time_dim)

        self.in_conv = nn.Conv2d(3, base_ch, 3, padding=1)
        self.down1 = Down(base_ch, base_ch, time_dim, time_dim)
        self.down2 = Down(base_ch, base_ch * 2, time_dim, time_dim)
        self.down3 = Down(base_ch * 2, base_ch * 4, time_dim, time_dim)

        self.mid = ResBlock(base_ch * 4, base_ch * 4, time_dim, time_dim)

        self.up3 = Up(base_ch * 4, base_ch * 4, base_ch * 2, time_dim, time_dim)
        self.up2 = Up(base_ch * 2, base_ch * 2, base_ch, time_dim, time_dim)
        self.up1 = Up(base_ch, base_ch, base_ch, time_dim, time_dim)

        self.out_conv = nn.Conv2d(base_ch, 3, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        c_emb = self.class_embed(y)

        x = self.in_conv(x)
        x, s1 = self.down1(x, t_emb, c_emb)
        x, s2 = self.down2(x, t_emb, c_emb)
        x, s3 = self.down3(x, t_emb, c_emb)

        x = self.mid(x, t_emb, c_emb)

        x = self.up3(x, s3, t_emb, c_emb)
        x = self.up2(x, s2, t_emb, c_emb)
        x = self.up1(x, s1, t_emb, c_emb)
        return self.out_conv(x)
```

---

## 4.3 `src/diffusion/scheduler.py`

```python
from __future__ import annotations

import torch


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)
```

---

## 4.4 `src/diffusion/ema.py`

```python
from __future__ import annotations

import copy
import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.ema_model.state_dict().items():
            if k in msd:
                v.copy_(self.decay * v + (1.0 - self.decay) * msd[k].detach())
```

---

## 4.5 `src/diffusion/ddpm.py`

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .scheduler import linear_beta_schedule


class DDPM(nn.Module):
    def __init__(self, model: nn.Module, timesteps: int = 1000, device: str = "cuda") -> None:
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.device = device

        betas = linear_beta_schedule(timesteps).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_ac = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_ac * x_start + sqrt_om * noise

    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, y: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t.float(), y)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, t_index: int) -> torch.Tensor:
        betas_t = self.betas[t][:, None, None, None]
        sqrt_one_minus_ac_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t][:, None, None, None]

        model_mean = sqrt_recip_alpha_t * (
            x - betas_t * self.model(x, t.float(), y) / sqrt_one_minus_ac_t
        )

        if t_index == 0:
            return model_mean
        posterior_var_t = self.posterior_variance[t][:, None, None, None]
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_var_t) * noise

    @torch.no_grad()
    def sample(self, image_size: int, batch_size: int, channels: int, y: torch.Tensor) -> torch.Tensor:
        x = torch.randn((batch_size, channels, image_size, image_size), device=self.device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t, y, i)
        return x
```

---

## 4.6 `src/utils/seed.py`

```python
from __future__ import annotations

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

---

## 4.7 `src/utils/image_utils.py`

```python
from __future__ import annotations

from pathlib import Path
import torch
from torchvision.utils import save_image, make_grid


def denormalize(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1) / 2


def save_batch_images(images: torch.Tensor, out_dir: str | Path, prefix: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = denormalize(images)
    for i, img in enumerate(images):
        save_image(img, out_dir / f"{prefix}_{i:05d}.png")


def save_grid(images: torch.Tensor, out_path: str | Path, nrow: int = 8) -> None:
    grid = make_grid(denormalize(images), nrow=nrow)
    save_image(grid, out_path)
```

---

## 4.8 `src/train_ddpm.py`

```python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.eye_dataset import EyeCropsDataset
from diffusion.unet import ConditionalUNet
from diffusion.ddpm import DDPM
from diffusion.ema import EMA
from utils.seed import set_seed
from utils.image_utils import save_grid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--base_ch", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="outputs/ddpm")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    grid_dir = output_dir / "grids"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    grid_dir.mkdir(parents=True, exist_ok=True)

    dataset = EyeCropsDataset(args.data_root, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = ConditionalUNet(num_classes=args.num_classes, base_ch=args.base_ch).to(device)
    diffusion = DDPM(model=model, timesteps=args.timesteps, device=str(device)).to(device)
    ema = EMA(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0.0

        for images, labels, _ in pbar:
            images = images.to(device)
            labels = labels.to(device)
            t = torch.randint(0, args.timesteps, (images.size(0),), device=device).long()
            loss = diffusion.p_losses(images, t, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(model)

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(len(loader), 1)
        print(f"Epoch {epoch+1}: avg loss = {avg_loss:.6f}")

        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.ema_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                },
                ckpt_dir / f"ddpm_epoch_{epoch+1}.pt",
            )

            ema.ema_model.eval()
            with torch.no_grad():
                y = torch.tensor([0] * 8 + [1] * 8, device=device)
                samples = DDPM(ema.ema_model, timesteps=args.timesteps, device=str(device)).to(device).sample(
                    image_size=args.image_size,
                    batch_size=16,
                    channels=3,
                    y=y,
                )
                save_grid(samples, grid_dir / f"samples_epoch_{epoch+1}.png", nrow=4)


if __name__ == "__main__":
    main()
```

---

## 4.9 `src/sample_ddpm.py`

```python
from __future__ import annotations

import argparse
from pathlib import Path
import csv

import torch

from diffusion.unet import ConditionalUNet
from diffusion.ddpm import DDPM
from utils.image_utils import save_batch_images


IDX_TO_CLASS = {0: "healthy", 1: "pink_eye"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--base_ch", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--num_healthy", type=int, default=0)
    parser.add_argument("--num_pink_eye", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--out_dir", type=str, default="data/eye_crops_synthetic")
    parser.add_argument("--metadata_csv", type=str, default="outputs/metadata/synthetic_images.csv")
    return parser.parse_args()


def generate_class(ddpm: DDPM, class_id: int, total: int, batch_size: int, image_size: int, out_dir: Path, rows: list[dict]):
    class_name = IDX_TO_CLASS[class_id]
    class_dir = out_dir / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    counter = 0
    while generated < total:
        current_bs = min(batch_size, total - generated)
        y = torch.full((current_bs,), class_id, device=ddpm.device, dtype=torch.long)
        samples = ddpm.sample(image_size=image_size, batch_size=current_bs, channels=3, y=y)
        prefix = f"synthetic_{class_name}"
        save_batch_images(samples, class_dir, prefix=f"{prefix}_{generated:05d}")

        for i in range(current_bs):
            filename = f"{prefix}_{generated:05d}_{i:05d}.png"
            rows.append({
                "filename": filename,
                "class_id": class_id,
                "class_name": class_name,
                "source": "ddpm",
            })
        generated += current_bs
        counter += 1
        print(f"Generated {generated}/{total} for {class_name}")


def main():
    args = parse_args()
    device = torch.device(args.device)

    model = ConditionalUNet(num_classes=args.num_classes, base_ch=args.base_ch).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("ema_state_dict", ckpt["model_state_dict"])
    model.load_state_dict(state)
    model.eval()

    ddpm = DDPM(model=model, timesteps=args.timesteps, device=str(device)).to(device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(args.metadata_csv)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    generate_class(ddpm, 0, args.num_healthy, args.batch_size, args.image_size, out_dir, rows)
    generate_class(ddpm, 1, args.num_pink_eye, args.batch_size, args.image_size, out_dir, rows)

    with metadata_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "class_id", "class_name", "source"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
```

---

## 4.10 `src/build_augmented_dataset.py`

```python
from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def copy_images(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for img_path in src.glob("*"):
        if img_path.is_file():
            shutil.copy2(img_path, dst / img_path.name)
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_root", type=str, required=True)
    parser.add_argument("--synthetic_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    args = parser.parse_args()

    real_root = Path(args.real_root)
    synthetic_root = Path(args.synthetic_root)
    out_root = Path(args.out_root)

    for cls in ["healthy", "pink_eye"]:
        out_cls = out_root / cls
        copy_images(real_root / cls, out_cls)
        copy_images(synthetic_root / cls, out_cls)

    print(f"Augmented dataset created at: {out_root}")


if __name__ == "__main__":
    main()
```

---

## 4.11 `configs/ddpm_train.yaml`

```yaml
data_root: data/eye_crops_real
image_size: 64
epochs: 200
batch_size: 16
lr: 0.0002
timesteps: 1000
base_ch: 64
num_classes: 2
output_dir: outputs/ddpm
seed: 42
```

---

## 4.12 `configs/ddpm_sample.yaml`

```yaml
checkpoint: outputs/ddpm/checkpoints/ddpm_epoch_200.pt
image_size: 64
timesteps: 1000
base_ch: 64
num_classes: 2
num_healthy: 0
num_pink_eye: 224
batch_size: 16
out_dir: data/eye_crops_synthetic
metadata_csv: outputs/metadata/synthetic_images.csv
```

---

## 5. Recommended training procedure

### Phase A — Train DDPM on real cropped eyes

Example:

```bash
python src/train_ddpm.py \
  --data_root data/eye_crops_real \
  --image_size 64 \
  --epochs 200 \
  --batch_size 16 \
  --lr 2e-4 \
  --output_dir outputs/ddpm
```

### Phase B — Generate minority-class synthetic images

You currently have roughly:

- healthy: 309
- pink_eye: 85

To balance pink_eye up toward healthy, generate about:

```text
309 - 85 = 224 synthetic pink_eye images
```

Example:

```bash
python src/sample_ddpm.py \
  --checkpoint outputs/ddpm/checkpoints/ddpm_epoch_200.pt \
  --image_size 64 \
  --num_healthy 0 \
  --num_pink_eye 224 \
  --out_dir data/eye_crops_synthetic \
  --metadata_csv outputs/metadata/synthetic_images.csv
```

### Phase C — Build augmented dataset

```bash
python src/build_augmented_dataset.py \
  --real_root data/eye_crops_real \
  --synthetic_root data/eye_crops_synthetic \
  --out_root data/eye_crops_augmented
```

---

## 6. Recommended experiments after generation

Once the augmented dataset exists, compare these classification conditions:

### Experiment 1 — Real only

```text
data/eye_crops_real
```

### Experiment 2 — Real + synthetic minority balancing

```text
data/eye_crops_augmented
```

### Experiment 3 — Real + selected synthetic samples only

Filter synthetic images manually or by confidence/quality before adding them.

---

## 7. Very important quality-control step

Do **not** automatically trust all synthetic images.

Before using synthetic data in classification:

1. preview synthetic image grids
2. manually inspect samples from both classes
3. remove unrealistic or degenerate samples
4. optionally ask Cursor to build a small filtering script

In diffusion projects, low-quality synthetic images can hurt the classifier.

---

## 8. Recommended practical choices for your project

For your dataset size, I recommend:

### First run
- image size: **64x64**
- class-conditional DDPM
- generate only **pink_eye** first

### Second run
- image size: **128x128**
- compare whether image quality improves

### Best initial target
Generate **224 pink_eye images** to approximately balance the dataset.

---

## 9. Suggested paper framing

If the synthetic experiment works, you can frame it as:

> We investigated whether class-conditional synthetic eye-image generation using DDPM can alleviate the scarcity of diseased cattle-eye samples and improve downstream pink-eye classification.

Then compare:

- no synthetic augmentation
- DDPM-balanced minority augmentation
- optional selective synthetic augmentation

---

## 10. Important warning

Because your current dataset is relatively small, DDPM may:

- overfit
- generate near-duplicates
- produce blurry samples

So the generation stage should be treated as an **experiment**, not assumed to help automatically.

That is why the right evaluation is:

```text
real only
vs
real + DDPM synthetic
```

on the same classifier protocol.

---

## 11. Best next step in practice

1. Train the DDPM on cropped eye images.
2. Generate synthetic `pink_eye` images.
3. Manually inspect quality.
4. Build an augmented dataset.
5. Re-run your best classifier (for example ResNet18 and cnn_transformer_strong).
6. Compare whether synthetic augmentation improves:
   - recall_pinkeye
   - F1_pinkeye
   - balanced accuracy

---

## 12. Optional next improvements

After this baseline DDPM works, you can upgrade to:

- classifier-free guidance
- better UNet attention blocks
- DDIM sampling
- FID / KID evaluation
- class-specific quality filtering
- latent diffusion if you scale up later

---

If you want next, I can turn this into a **single copy-paste Cursor prompt** that tells Cursor exactly how to generate all these files in your repository.

