# Step 2 Comparison Report: with_aug vs no_color_aug

**Project:** Cattle Pinkeye Detection - Step 2  
**Date:** March 13, 2026  
**Dataset:** `outputs/eye_crops_classification` (394 images: healthy=309, pink_eye=85)  
**Protocol:** 5-fold stratified CV, 5 epochs, batch size 8, weighted CE, early stopping

---

## 1. Experiment Definitions

### A) with_aug (previous benchmark)

Output root: `outputs/benchmark_gpu/`

Training transforms included:
- random resized crop
- horizontal flip
- random rotation
- color jitter
- mild affine

### B) no_color_aug (new benchmark)

Output root: `outputs/benchmark_gpu_nocolor_aug/`

Training transforms included only:
- horizontal flip (`p=0.5`)
- rotation ±10 deg (`p=0.3`)
- small Gaussian noise (`p=0.1`)

No color/brightness/contrast jitter was used.

Validation/test preprocessing in this setup:
- resize
- normalize

---

## 2. Delta Summary (requested)

Delta is computed as:

`delta = no_color_aug - with_aug`

| Model | F1_pinkeye (with_aug) | F1_pinkeye (no_color_aug) | Delta F1_pinkeye | Accuracy (with_aug) | Accuracy (no_color_aug) | Delta Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| `resnet18` | 0.8992 | **0.9190** | **+0.0198** | 0.9569 | **0.9645** | **+0.0076** |
| `efficientnet_b0` | **0.9000** | 0.8721 | **-0.0279** | **0.9568** | 0.9416 | **-0.0152** |
| `cnn_transformer` | 0.8369 | 0.8272 | -0.0097 | 0.9239 | 0.9238 | -0.0001 |
| `cnn_transformer_strong` | **0.9051** | 0.8672 | **-0.0379** | **0.9594** | 0.9441 | **-0.0153** |

---

## 3. Full Metric Comparison (mean)

### 3.1 with_aug

| Model | Accuracy | Balanced Acc | Precision | Recall | F1 | ROC-AUC | Recall_pinkeye | F1_pinkeye |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `resnet18` | 0.9569 | 0.9342 | 0.9094 | 0.8941 | 0.8992 | 0.9867 | 0.8941 | 0.8992 |
| `efficientnet_b0` | 0.9568 | 0.9383 | 0.8980 | 0.9059 | 0.9000 | 0.9829 | 0.9059 | 0.9000 |
| `cnn_transformer` | 0.9239 | 0.9132 | 0.7934 | 0.8941 | 0.8369 | 0.9681 | 0.8941 | 0.8369 |
| `cnn_transformer_strong` | **0.9594** | 0.9358 | **0.9251** | 0.8941 | **0.9051** | 0.9850 | 0.8941 | **0.9051** |

### 3.2 no_color_aug

| Model | Accuracy | Balanced Acc | Precision | Recall | F1 | ROC-AUC | Recall_pinkeye | F1_pinkeye |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `resnet18` | **0.9645** | **0.9518** | **0.9146** | **0.9294** | **0.9190** | 0.9778 | **0.9294** | **0.9190** |
| `efficientnet_b0` | 0.9416 | 0.9372 | 0.8248 | 0.9294 | 0.8721 | **0.9827** | 0.9294 | 0.8721 |
| `cnn_transformer` | 0.9238 | 0.8918 | 0.8240 | 0.8353 | 0.8272 | 0.9663 | 0.8353 | 0.8272 |
| `cnn_transformer_strong` | 0.9441 | 0.9090 | 0.8914 | 0.8471 | 0.8672 | 0.9726 | 0.8471 | 0.8672 |

---

## 4. Interpretation

1. The no-color augmentation setting helped **ResNet18** the most and made it the top model in this experiment.
2. **EfficientNet-B0** and both hybrid transformer models dropped in F1/accuracy under no-color augmentation.
3. This suggests color-related variability/regularization in the original augmentation setup may have benefited non-ResNet architectures more.
4. For deployment candidates:
   - Best in `with_aug`: `cnn_transformer_strong`
   - Best in `no_color_aug`: `resnet18`

---

## 5. Output Paths

- with_aug:
  - `outputs/benchmark_gpu/resnet18/metrics/`
  - `outputs/benchmark_gpu/efficientnet/metrics/`
  - `outputs/benchmark_gpu/cnn_transformer/metrics/`
  - `outputs/benchmark_gpu/cnn_transformer_strong/metrics/`

- no_color_aug:
  - `outputs/benchmark_gpu_nocolor_aug/resnet18/metrics/`
  - `outputs/benchmark_gpu_nocolor_aug/efficientnet/metrics/`
  - `outputs/benchmark_gpu_nocolor_aug/cnn_transformer/metrics/`
  - `outputs/benchmark_gpu_nocolor_aug/cnn_transformer_strong/metrics/`
