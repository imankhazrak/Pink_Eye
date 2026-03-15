# Step 2 Controlled Augmentation Report (Flip + Rotation)

**Project:** Cattle Pinkeye Detection - Step 2  
**Date:** March 13, 2026  
**Experiment goal:** Test whether Gaussian noise contributed to performance drops by running a controlled geometric-only augmentation experiment.

---

## 1. Dataset and Protocol

### Dataset
- Root: `outputs/eye_crops_classification`
- Total: **394** images
  - `healthy` (label 0): **309**
  - `pink_eye` (label 1): **85**

### Training protocol (kept fixed)
- 5-fold stratified cross-validation
- 5 epochs
- Batch size: 8
- Loss: weighted cross-entropy
- Early stopping: enabled
- Same class imbalance handling and optimizer settings as prior runs

### Models evaluated
1. `resnet18`
2. `efficientnet_b0`
3. `cnn_transformer_strong`

---

## 2. Augmentation Policies Compared

### A) Previous run (no_color_aug)
Training-only transforms:
- Horizontal flip (`p=0.5`)
- Rotation ±10 deg (`p=0.3`)
- Gaussian noise (`p=0.1`)

No color jitter / brightness / contrast changes.

### B) Current controlled run (flip_rotation only)
Training-only transforms:
- Horizontal flip (`p=0.5`)
- Rotation ±10 deg (`p=0.3`)

Removed:
- Gaussian noise
- any color/brightness/contrast augmentation

Validation/test for both setups:
- resize + normalization only

---

## 3. Controlled Run Results (Flip + Rotation only)

**Output root:** `outputs/aug_flip_rotation/`

| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1-score | ROC-AUC | Recall (pinkeye) | F1 (pinkeye) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `resnet18` | 0.9543 ± 0.0147 | 0.9452 ± 0.0229 | 0.8712 ± 0.0516 | 0.9294 ± 0.0492 | 0.8979 ± 0.0312 | 0.9842 ± 0.0134 | 0.9294 ± 0.0492 | **0.8979 ± 0.0312** |
| `efficientnet_b0` | 0.9365 ± 0.0269 | 0.9041 ± 0.0501 | 0.8788 ± 0.1129 | 0.8471 ± 0.1220 | 0.8518 ± 0.0549 | 0.9735 ± 0.0238 | 0.8471 ± 0.1220 | 0.8518 ± 0.0549 |
| `cnn_transformer_strong` | 0.9467 ± 0.0226 | 0.9234 ± 0.0241 | 0.8835 ± 0.0904 | 0.8824 ± 0.0588 | 0.8788 ± 0.0428 | 0.9756 ± 0.0117 | 0.8824 ± 0.0588 | 0.8788 ± 0.0428 |

Best model under this controlled setting (primary metric `F1_pinkeye`): **`resnet18`**

---

## 4. Primary Comparison Table (Noise Ablation)

Comparison requested:
- **No Aug (here interpreted as previous no-color run):** flip + rotation + Gaussian noise
- **Flip+Rotation:** current controlled run
- Metric of interest: `F1_pinkeye`

| Model | No Aug (`F1_pinkeye`) | Flip+Rotation (`F1_pinkeye`) | Delta (Flip+Rotation - No Aug) |
|---|---:|---:|---:|
| `resnet18` | 0.9190 ± 0.0374 | 0.8979 ± 0.0312 | **-0.0210** |
| `efficientnet_b0` | 0.8721 ± 0.0347 | 0.8518 ± 0.0549 | **-0.0202** |
| `cnn_transformer_strong` | 0.8672 ± 0.0285 | 0.8788 ± 0.0428 | **+0.0117** |

---

## 5. Interpretation

1. Removing Gaussian noise **improved** `cnn_transformer_strong` (`+0.0117` F1_pinkeye).
2. Removing Gaussian noise **decreased** `resnet18` and `efficientnet_b0` F1_pinkeye (both around `-0.02`).
3. This indicates Gaussian noise was likely detrimental for the stronger hybrid model, but acted as useful regularization for the CNN baselines in this setup.
4. Under flip+rotation only, `resnet18` remains the strongest of the three models tested.

---

## 6. Output Locations

- Controlled run:
  - `outputs/aug_flip_rotation/resnet18/`
  - `outputs/aug_flip_rotation/efficientnet/`
  - `outputs/aug_flip_rotation/cnn_transformer_strong/`

- Reference run (with Gaussian noise):
  - `outputs/benchmark_gpu_nocolor_aug/resnet18/`
  - `outputs/benchmark_gpu_nocolor_aug/efficientnet/`
  - `outputs/benchmark_gpu_nocolor_aug/cnn_transformer_strong/`
