# Pinkeye Detection Report: Eye Detection, Classification, and DDPM-Based Synthetic Augmentation

This report summarizes the end-to-end pinkeye pipeline, including eye detection (Step 1), classification benchmarking and augmentation analysis (Step 2), and the current DDPM-based synthetic image generation status for healthy and pink-eye classes.

## Table of Contents
- [1. Executive Summary](#1-executive-summary)
- [2. Step 1: Eye Localization (YOLOv8)](#2-step-1-eye-localization-yolov8)
  - [2.1. Data and setup](#21-data-and-setup)
  - [2.2. Detector performance](#22-detector-performance)
  - [2.3. Classification Dataset Detection Pass](#23-classification-dataset-detection-pass)
- [3. Step 2: Classification Benchmark (5-Fold Stratified CV)](#3-step-2-classification-benchmark-5-fold-stratified-cv)
  - [3.1. Dataset](#31-dataset)
  - [3.2. Protocol](#32-protocol)
  - [3.3. Baseline model comparison (mean +/- std)](#33-baseline-model-comparison-mean--std)
- [4. Augmentation Experiments and Insights](#4-augmentation-experiments-and-insights)
  - [4.1. A) with_aug vs no_color_aug](#41-a-with_aug-vs-no_color_aug)
  - [4.2. B) Gaussian-noise ablation (flip+rotation only)](#42-b-gaussian-noise-ablation-fliprotation-only)
  - [4.3. C) Minority-only augmentation (pink_eye only)](#43-c-minority-only-augmentation-pink_eye-only)
- [5. Overall Technical Conclusions](#5-overall-technical-conclusions)
- [6. DDPM-Based Synthetic Data Augmentation (Current Status)](#6-ddpm-based-synthetic-data-augmentation-current-status)
  - [6.1. Data sharing status](#61-data-sharing-status)
  - [6.2. DDPM model configurations](#62-ddpm-model-configurations)
  - [6.3. Original vs. Synthetic Images](#63-original-vs-synthetic-images)
  - [6.4. Current and planned usage](#64-current-and-planned-usage)
  - [6.5. Goal](#65-goal)
- [7. Support Needed: Synthetic Data Curation](#7-support-needed-synthetic-data-curation)
  - [7.1. Reason](#71-reason)
  - [7.2. Requested support](#72-requested-support)
  - [7.3. Expected impact](#73-expected-impact)

Project: Cattle Pinkeye Detection  
Prepared for: Team update  
Date: March 2026

## 1. Executive Summary

The project progressed from robust eye localization (Step 1) to multi-model disease classification on cropped eye regions (Step 2), followed by DDPM-based synthetic data generation.

Main outcomes:

- Step 1 eye detector achieved mAP50 = 0.995 on validation.
- On the Step 2 baseline benchmark (5-fold CV), Strong CNN+Transformer had the best overall F1/accuracy balance.
- Across augmentation ablations, ResNet18 became the most stable top performer under constrained augmentation settings.


<table>
  <tr>
    <td align="center">
      <img src="../runs/detect/predict3/1 (22).jpg" alt="Eye detected by the model" style="width: 420px; height: 320px; object-fit: cover;" />
    </td>
    <td align="center">
      <img src="../outputs/eye_crops_classification/pink_eye/1 (22).jpg" alt="Cropped by the model" style="width: 420px; height: 320px; object-fit: cover;" />
    </td>
  </tr>
  <tr>
    <td align="center">Eye detected by the model</td>
    <td align="center">Cropped by the model</td>
  </tr>
</table>

- The eye-only approach shows strong feasibility for pinkeye screening, with model choice and augmentation policy materially affecting minority-class performance.

## 2. Step 1: Eye Localization (YOLOv8)

### 2.1. Data and setup

- Eye dataset for annotation: 117 images, single class (eye).
- Annotation validation: 117/117 valid, no errors.
- Split: train/val/test = 93 / 11 / 13.

### 2.2. Detector performance

- Precision: 1.000
- Recall: 0.999
- mAP50: 0.995
- mAP50-95: 0.631

### 2.3. Classification Dataset Detection Pass

- Sample classification pool (`data/sample_test`): 408 images
- Images with eye detections: 394
- No detections: 14 (copied for manual review)

Conclusion: detector quality is strong enough to support the classification stage.

## 3. Step 2: Classification Benchmark (5-Fold Stratified CV)

### 3.1. Dataset

- Cropped-eye dataset: 394 images
- Healthy: 309
- Pink-eye: 85

### 3.2. Protocol

- 5-fold stratified CV
- 5 epochs per fold
- Weighted cross-entropy for imbalance handling
- Early stopping enabled

### 3.3. Baseline model comparison (mean +/- std)

| Model | Accuracy | Balanced Acc | F1-score | ROC-AUC | Recall_pinkeye | F1_pinkeye |
|---|---:|---:|---:|---:|---:|---:|
| ResNet18 | 0.9569 +/- 0.0144 | 0.9342 +/- 0.0289 | 0.8992 +/- 0.0342 | 0.9867 +/- 0.0117 | 0.8941 +/- 0.0644 | 0.8992 +/- 0.0342 |
| EfficientNet-B0 | 0.9568 +/- 0.0114 | 0.9383 +/- 0.0297 | 0.9000 +/- 0.0279 | 0.9829 +/- 0.0122 | 0.9059 +/- 0.0671 | 0.9000 +/- 0.0279 |
| CNN+Transformer | 0.9239 +/- 0.0235 | 0.9132 +/- 0.0215 | 0.8369 +/- 0.0429 | 0.9681 +/- 0.0327 | 0.8941 +/- 0.0492 | 0.8369 +/- 0.0429 |
| Strong CNN+Transformer | 0.9594 +/- 0.0106 | 0.9358 +/- 0.0229 | 0.9051 +/- 0.0214 | 0.9850 +/- 0.0024 | 0.8941 +/- 0.0644 | 0.9051 +/- 0.0214 |

Key readout:

- Best overall benchmark model: Strong CNN+Transformer
- Best pink-eye recall in this run: EfficientNet-B0

## 4. Augmentation Experiments and Insights

### 4.1. A) with_aug vs no_color_aug

No-color setup used only flip + rotation + small Gaussian noise (no color jitter/affine).

Most important deltas (`no_color_aug - with_aug`):

- ResNet18: improved
  - Delta F1_pinkeye: +0.0198
  - Delta Accuracy: +0.0076
- EfficientNet-B0: decreased
  - Delta F1_pinkeye: -0.0279
- Strong CNN+Transformer: decreased
  - Delta F1_pinkeye: -0.0379

Interpretation: color/affine augmentations appear beneficial for non-ResNet models in this dataset, while ResNet18 benefited from a simpler augmentation policy.

### 4.2. B) Gaussian-noise ablation (flip+rotation only)

Comparing flip+rotation to no_color_aug (which included Gaussian noise):

- ResNet18: F1_pinkeye decreased (-0.0210)
- EfficientNet-B0: F1_pinkeye decreased (-0.0202)
- Strong CNN+Transformer: F1_pinkeye increased (+0.0117)

Interpretation: Gaussian noise acted as useful regularization for CNN baselines but hurt the strong hybrid model.

### 4.3. C) Minority-only augmentation (pink_eye only)

Applying flip+rotation only to minority class:

- Best in this setting: ResNet18 (F1_pinkeye 0.9060 +/- 0.0282)
- Did not outperform each model's best previous setting overall.

Interpretation: minority-only augmentation alone was not a universal improvement.

## 5. Overall Technical Conclusions

- Eye-crop classification is viable: strong metrics across multiple architectures confirm informative signal exists in eye region.
- Model ranking depends on augmentation regime:
  - Baseline benchmark winner: Strong CNN+Transformer
  - Most robust under constrained/controlled augmentations: ResNet18
- Class imbalance handling is effective: weighted CE + stratified CV produced stable minority-class recall/F1.
- Augmentation policy is a first-order factor: small transform changes materially shift F1_pinkeye.

## 6. DDPM-Based Synthetic Data Augmentation (Current Status)

We have generated DDPM class-conditional synthetic eye images for both classes (`healthy` and `pink_eye`) for augmentation experiments.

### 6.1. Data sharing status

- Original cropped eye images and generated synthetic images have been uploaded to OneDrive.
- Upload includes both classes (`healthy` and `pink_eye`) for side-by-side review and curation.

### 6.2. DDPM model configurations

- Batch size: 16
- Number of timesteps: 10k
- Number of epochs: 2048

### 6.3. Original vs. Synthetic Images

- Healthy Images

<table>
  <tr>
    <td align="center">
      <img src="../outputs/eye_crops_classification/healthy/2 (24).jpg" alt="Original healthy image" style="width: 420px; height: 320px; object-fit: cover;" />
    </td>
    <td align="center">
      <img src="../outputs/ddpm/2. Generated_healthy_clean_42_43_44_45_46_47_48_49/healthy_synthetic (1).png" alt="Synthetic healthy image" style="width: 420px; height: 320px; object-fit: cover;" />
    </td>
  </tr>
  <tr>
    <td align="center">Original</td>
    <td align="center">Synthetic</td>
  </tr>
</table>

- Pink-Eye Images

<table>
  <tr>
    <td align="center">
      <img src="../outputs/eye_crops_classification/pink_eye/1 (22).jpg" alt="Original pink-eye image" style="width: 420px; height: 320px; object-fit: cover;" />
    </td>
    <td align="center">
      <img src="../outputs/ddpm/1. Generated_pink_clean_42_43_44_45_46_47_48_49_50/generated_pink (160).png" alt="Synthetic pink-eye image" style="width: 420px; height: 320px; object-fit: cover;" />
    </td>
  </tr>
  <tr>
    <td align="center">Original</td>
    <td align="center">Synthetic</td>
  </tr>
</table>

### 6.4. Current and planned usage

- Generate synthetic images under controlled quality checks.
- Add selected synthetic samples to training folds in a controlled manner.
- Re-run the Step 2 classifier benchmarks and compare against real-only training.
- Report impact on minority-class metrics, especially `Recall_pinkeye` and `F1_pinkeye`.

### 6.5. Goal

- Improve robustness and class balance by augmenting limited real pink-eye samples with high-quality synthetic data.

## 7. Support Needed: Synthetic Data Curation

After synthetic images are generated, support is needed for a data-curation pass before using them in model training.

### 7.1. Reason

- The generated samples are not expected to be 100% correct.
- Some images may contain visual artifacts, unrealistic anatomy, or incorrect class appearance.

### 7.2. Requested support

- Help review synthetic image batches and remove low-quality/incorrect samples.
- Help maintain a cleaned subset of synthetic data for training experiments.
- Optionally define a simple curation rubric to keep filtering consistent across reviewers.

### 7.3. Expected impact

- Higher quality augmented dataset.
- Lower risk of introducing noisy labels or harmful artifacts into classifier training.
- More reliable benchmark comparisons after augmentation.
