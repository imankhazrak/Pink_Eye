# Eye Detector Training Report

**Project:** Cattle Pinkeye Detection – Step 1 (Eye Localization)  
**Date:** March 12, 2026  
**Report:** Eye detection model training pipeline

---

## 1. Overview

This report documents the process of training a YOLOv8-based eye detector for cattle head images. The detector localizes the eye region in each image, which will later be cropped and used as input for the pink-eye classification stage.

---

## 2. Data Source

| Field | Value |
|-------|-------|
| **Source** | Roboflow (Eye Detection v3) |
| **Export date** | March 12, 2026 |
| **Total images** | 117 |
| **Format** | YOLOv8 |
| **Class** | `eye` (single class, ID 0) |
| **Annotation** | `0 x_center y_center width height` (normalized) |

**Reference:** [Roboflow Eye Detection Dataset](https://universe.roboflow.com/imans-workspace-rsc4n/eye-detection-fayd6)

---

## 3. Annotation Validation

**Script:** `scripts/check_annotations_roboflow.py`  
**Source:** `data/annotated_data/train/`

### Checks performed

- Image–label pairing (every image has a matching `.txt` file)
- Non-empty labels
- Format: 5 space-separated tokens per line
- Class ID = 0
- Coordinates in [0, 1], width > 0, height > 0
- Flagging of very small or large boxes (<1% or >90% of image)

### Results

| Metric | Count |
|--------|-------|
| Label files checked | 117 |
| Valid (no issues) | 117 |
| Warnings | 0 |
| Errors | 0 |

**Report:** `outputs/metadata/annotation_check_roboflow_report.csv`

### Visual QA

Sample previews with overlaid bounding boxes were generated at  
`outputs/annotation_preview/` to verify box placement.

---

## 4. Dataset Split

**Script:** `scripts/prepare_eye_dataset.py`  
**Output:** `data/dataset/`

### Structure

```
data/dataset/
├── images/
│   ├── train/   (93 images)
│   ├── val/     (11 images)
│   └── test/    (13 images)
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

### Split ratios

| Split | Ratio | Count |
|-------|-------|-------|
| Train | 80% | 93 |
| Val   | 10% | 11 |
| Test  | 10% | 13 |

**Random seed:** 42 (reproducible)

---

## 5. Training Configuration

| Parameter | Value |
|-----------|-------|
| **Model** | YOLOv8n (nano) pretrained |
| **Epochs** | 100 |
| **Image size** | 640 × 640 |
| **Batch size** | 16 |
| **Optimizer** | AdamW (auto-selected) |
| **GPU** | Tesla V100-PCIE-16GB |
| **Framework** | Ultralytics 8.4.21, PyTorch 2.8.0, CUDA 12.8 |

### data.yaml

```yaml
path: /users/PCS0229/imankhazrak/Pink_Eye/data/dataset
train: images/train
val: images/val
test: images/test

names:
  0: eye
```

---

## 6. Training Execution

**Method:** SLURM job submission  
**Script:** `train_eye_detector.slurm`  
**Job ID:** 45587341  
**Start:** Thu Mar 12 08:36:29 PM EDT 2026  
**End:** Thu Mar 12 08:39:09 PM EDT 2026  
**Duration:** ~2 min 41 s (0.025 hours)

### Command

```bash
yolo detect train \
    data=data/dataset/data.yaml \
    model=yolov8n.pt \
    epochs=100 \
    imgsz=640 \
    batch=16
```

---

## 7. Results

### Final validation metrics (best model)

| Metric | Value |
|--------|-------|
| **Precision** | 1.000 |
| **Recall** | 0.999 |
| **mAP50** | **0.995** |
| **mAP50-95** | 0.631 |
| **Images (val)** | 11 |
| **Instances** | 11 |

### Inference speed (validation)

- Preprocess: 2.6 ms/image  
- Inference: 0.8 ms/image  
- Postprocess: 0.7 ms/image  

### Model summary

- **Layers:** 73 (fused)
- **Parameters:** 3,005,843
- **GFLOPs:** 8.1
- **Best weights:** `runs/detect/train/weights/best.pt` (6.3 MB)

---

## 8. Output Artifacts

| Artifact | Path |
|----------|------|
| Best weights | `runs/detect/train/weights/best.pt` |
| Last weights | `runs/detect/train/weights/last.pt` |
| Results | `runs/detect/train/` |
| Training curves | `runs/detect/train/results.png` |
| Confusion matrix | `runs/detect/train/confusion_matrix.png` |
| SLURM log | `slurm-45587341.out` |

---

## 9. Pipeline Summary

```
Annotated images (Roboflow)
    ↓
Check annotations (117/117 valid)
    ↓
Split 80/10/10 → data/dataset/
    ↓
Train YOLOv8n (100 epochs, ~2.5 min)
    ↓
Best model: mAP50 = 0.995
    ↓
Predict on test set (13/13 eyes detected)
    ↓
Predict on classification dataset (sample_test: 408 images)
    ↓
394/408 with eye detections → ready for cropping & classification
14/408 no detections → outputs/no_detection_samples/ (manual review)
    ↓
Ready for pink-eye classifier dataset
```

---

## 10. Prediction on Unseen Images

**Command:**
```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=data/dataset/images/test save=True
```

**Results (13 held-out test images):**

| Metric | Value |
|--------|-------|
| Images processed | 13 |
| Eyes detected | 13 (1 per image) |
| Failures | 0 |
| Output | `runs/detect/predict/` |

All 13 test images had exactly one eye detected. **Manual inspection recommended:** open images in `runs/detect/predict/` to verify box placement (centered, tight around eye, robust to angles/lighting).

---

## 11. Eye Cropping

**Script:** `scripts/crop_eyes_roboflow.py`

**Command:**
```bash
python scripts/crop_eyes_roboflow.py --source data/annotated_data/train/images
```

**Results (117 images):**

| Metric | Value |
|--------|-------|
| Source | `data/annotated_data/train/images` |
| Success | 117 |
| Failed | 0 |
| Output | `outputs/eye_crops_pred/crops/` |
| Metadata | `outputs/metadata/eye_crops_roboflow.csv` |
| Padding | 10% |
| Conf threshold | 0.25 |

**Next:** Inspect cropped eyes in `outputs/eye_crops_pred/crops/` and use them to build the pink-eye classifier dataset. If you have disease labels (healthy/pinkeye) from another source, organize crops into `eye_crops/healthy/` and `eye_crops/pinkeye/` for classification training.

---

## 12. Classification Task Dataset (sample_test)

**Source:** `data/sample_test/`  
**Purpose:** Full set of images for the pink-eye classification task (Step 2).

**Command:**
```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=data/sample_test save=True project=runs/detect name=sample_test
```

### Results

| Metric | Value |
|--------|-------|
| **Images processed** | 408 |
| **Images with ≥1 eye detected** | 394 |
| **Images with no detections** | 14 |
| **Output** | `runs/detect/sample_test/` |
| **No-detection samples (copied)** | `outputs/no_detection_samples/` |

These 408 images represent the whole dataset available for the classification task. Images with detections can be cropped for classifier input; the 14 with no detections may need manual review (face/eye unclear, occluded, or odd angle).

**Images with no detections:**
- 2 (111).JPG, 2 (117).JPG, 2 (127).JPG, 2 (131).jpg, 2 (134).JPG, 2 (136).JPG, 2 (143).jpg, 2 (154).JPG, 2 (177).JPG, 2 (185).JPG, 2 (257).JPG, 2 (280).JPG, 2 (305).JPG, 2 (54).jpg

---

## 13. Next Steps

1. **Manual inspection:** Review prediction overlays in `runs/detect/sample_test/` and `outputs/no_detection_samples/`.
2. **Crop eyes for classification:** Run cropping on the 394 images with detections to prepare classifier input.
3. **Classifier dataset:** Organize cropped eyes by disease label and proceed to Step 2 (pink-eye classification).
4. **Optional:** Try `yolov8s.pt` for potential accuracy gains if needed.

---

## Appendix: Scripts Used

| Script | Purpose |
|--------|---------|
| `scripts/check_annotations_roboflow.py` | Validate YOLO annotations |
| `scripts/prepare_eye_dataset.py` | Split dataset and create data.yaml |
| `scripts/preview_annotations.py` | Draw boxes on sample images for QA |
| `scripts/crop_eyes_roboflow.py` | Crop eyes using trained detector |
| `train_eye_detector.slurm` | SLURM job for training |
