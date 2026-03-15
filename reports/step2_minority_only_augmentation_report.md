# Step 2 Report: Minority-Only Augmentation Experiment

**Project:** Cattle Pinkeye Detection - Step 2  
**Date:** March 13, 2026  
**Experiment:** Apply train-time augmentation only to minority class (`pink_eye`, label=1)

---

## 1. Objective

Evaluate whether restricting geometric augmentation to only the minority class (`pink_eye`) improves pinkeye detection performance under class imbalance.

---

## 2. Dataset and Protocol

### Dataset
- Root: `outputs/eye_crops_classification`
- Total images: **394**
  - `healthy` (label 0): **309**
  - `pink_eye` (label 1): **85**

### Training protocol (same as prior runs)
- Stratified 5-fold cross-validation
- 5 epochs
- Batch size: 8
- Weighted cross-entropy loss
- Early stopping enabled
- Pretrained backbones enabled

### Models
1. `resnet18`
2. `efficientnet_b0`
3. `cnn_transformer_strong`

---

## 3. Augmentation Policy Used

### Minority-only training augmentation
Applied **only when label = 1 (`pink_eye`)**:
- Horizontal flip (`p=0.5`)
- Rotation ±10 degrees (`p=0.3`)

Not used:
- Gaussian noise
- ColorJitter
- brightness/contrast augmentation

For `healthy` class in training:
- resize + normalization only

Validation/test:
- resize + normalization only

---

## 4. Experiment Run Info

- SLURM job: `45619485`
- Status: `COMPLETED`
- Elapsed: `00:03:19`
- Output root: `outputs/aug_flip_rotation_minority_only/`

---

## 5. Minority-Only Results (mean ± std over 5 folds)

| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1-score | ROC-AUC | Recall_pinkeye | F1_pinkeye |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `resnet18` | 0.9594 ± 0.0105 | 0.9443 ± 0.0342 | 0.8996 ± 0.0366 | 0.9176 ± 0.0789 | 0.9060 ± 0.0282 | 0.9820 ± 0.0218 | 0.9176 ± 0.0789 | **0.9060 ± 0.0282** |
| `efficientnet_b0` | 0.9366 ± 0.0236 | 0.9298 ± 0.0399 | 0.8249 ± 0.0903 | 0.9176 ± 0.0984 | 0.8624 ± 0.0484 | 0.9724 ± 0.0340 | 0.9176 ± 0.0984 | 0.8624 ± 0.0484 |
| `cnn_transformer_strong` | 0.9416 ± 0.0112 | 0.8988 ± 0.0277 | 0.8987 ± 0.0302 | 0.8235 ± 0.0588 | 0.8582 ± 0.0313 | 0.9702 ± 0.0175 | 0.8235 ± 0.0588 | 0.8582 ± 0.0313 |

Best model by primary metric (`F1_pinkeye`): **`resnet18`**

---

## 6. Comparison to Previous Settings (F1_pinkeye)

Reference settings:
- **No-color+noise:** flip + rotation + Gaussian noise
- **Flip+rotation (all classes):** geometric augmentation applied to both classes
- **Minority-only:** current experiment

| Model | No-color+noise | Flip+rotation (all classes) | Flip+rotation (minority-only) |
|---|---:|---:|---:|
| `resnet18` | **0.9190** | 0.8979 | 0.9060 |
| `efficientnet_b0` | **0.8721** | 0.8518 | 0.8624 |
| `cnn_transformer_strong` | 0.8672 | **0.8788** | 0.8582 |

---

## 7. Interpretation

1. `resnet18`: minority-only augmentation improved over all-class geometric augmentation, but did not surpass the no-color+noise baseline.
2. `efficientnet_b0`: same trend as ResNet18; minority-only outperformed all-class geometric, but remained below no-color+noise.
3. `cnn_transformer_strong`: minority-only was the weakest setting among its recent augmentation variants.
4. Overall for this experiment, minority-only augmentation did not produce the best global configuration across all models.

---

## 8. Output Paths

- `outputs/aug_flip_rotation_minority_only/resnet18/metrics/`
- `outputs/aug_flip_rotation_minority_only/efficientnet/metrics/`
- `outputs/aug_flip_rotation_minority_only/cnn_transformer_strong/metrics/`

Comparison references:
- `outputs/benchmark_gpu_nocolor_aug/...`
- `outputs/aug_flip_rotation/...`
