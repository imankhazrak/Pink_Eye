# Cattle Pinkeye Detection

A two-stage deep learning pipeline for detecting pinkeye (infectious bovine keratoconjunctivitis) in cattle from head images.

**Stage 1** localises the eye region with a YOLO detector and produces a cropped-eye dataset.
**Stage 2** (upcoming) trains classification models on the cropped eyes.

## Project Structure

```
Pink_Eye/
├── data/
│   ├── Labeled Cow Eyes Data/   # Source images (healthy/ and pinkeye/)
│   ├── annotations/             # YOLO-format eye bounding-box labels
│   └── animal_ids.csv           # Optional animal-ID metadata
├── detection_dataset/           # Generated YOLO dataset (images + labels)
├── outputs/
│   ├── detector/                # YOLO training runs + best weights
│   ├── eye_crops_pred/          # Crops from YOLO predictions
│   ├── eye_crops_gt/            # Crops from ground-truth annotations
│   └── metadata/                # CSVs (crop metadata, failures, reviews)
├── src/
│   └── step1/                   # Step 1 pipeline scripts
├── models/                      # Saved model checkpoints
├── notebooks/                   # Jupyter notebooks
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Step 1 – Eye Localisation and Cropping

Run each script as a Python module from the project root.
They should be executed in order; each script depends on the
output of the previous one.

### 1. Explore the dataset

```bash
python -m src.step1.explore_dataset
```

Prints per-class image counts, dimension statistics, and class-imbalance ratio.
Saves a sample grid to `outputs/sample_grid.png`.

### 2. Annotate eye bounding boxes

```bash
python -m src.step1.annotate_eyes
python -m src.step1.annotate_eyes --class_filter pinkeye
```

Opens an OpenCV window for each unannotated image.
Draw a box around the eye and press **s** to save.
Press **q** to quit; re-run to resume from where you left off.

**Annotation rule** (also displayed in the tool window):
draw one box around the visible eye region, including the
eyeball/cornea and a small amount of surrounding eyelid tissue.
Do not include excessive background. Keep the style consistent
across all images.

### 3. Validate annotations

```bash
python -m src.step1.check_annotations
```

Checks every `.txt` label file for format errors, out-of-range
coordinates, and suspicious box sizes. Saves a report to
`outputs/metadata/annotation_check_report.csv`.

### 4. Build the YOLO detection dataset

```bash
python -m src.step1.build_detection_dataset
```

Splits annotated images into train/val/test (70/15/15).
If `data/animal_ids.csv` exists with an `animal_id` column,
group-aware splitting is used to prevent data leakage; otherwise
stratified splitting by disease label is applied automatically.

### 5. Train the YOLO eye detector

```bash
python -m src.step1.train_detector
python -m src.step1.train_detector --model yolov8s.pt --epochs 80
```

The default model is **YOLOv8n** (nano).  To compare with a
higher-capacity model, pass `--model yolov8s.pt`.  The model
choice can also be changed permanently in `src/step1/config.py`
(`YOLO_MODEL`).

### 6. Validate the detector

```bash
python -m src.step1.validate_detector
python -m src.step1.validate_detector --split test --conf 0.30
```

Reports mAP, precision, recall and saves a visual grid of predicted
boxes to `outputs/detector/validation/`.

### 7. Crop eye regions

```bash
# Crop using YOLO predictions (default)
python -m src.step1.crop_eyes

# Crop using ground-truth annotations (for ablation studies)
python -m src.step1.crop_eyes --source ground_truth

# Sweep multiple confidence thresholds
python -m src.step1.crop_eyes --conf 0.15,0.25,0.40
```

**Predicted crops** go to `outputs/eye_crops_pred/{healthy,pinkeye}/`.
**Ground-truth crops** go to `outputs/eye_crops_gt/{healthy,pinkeye}/`.
Metadata and failure logs are saved to `outputs/metadata/`.

### 8. Review crop quality

```bash
# Interactive review
python -m src.step1.review_crops
python -m src.step1.review_crops --source ground_truth

# Quick visual grid (non-interactive)
python -m src.step1.review_crops --auto
```

Interactive mode: press **g** (good), **b** (bad), or **u** (uncertain).
All decisions are saved with timestamps to
`outputs/metadata/crop_review.csv`.

## Configuration

All tuneable parameters live in `src/step1/config.py`:

| Parameter | Default | Purpose |
|---|---|---|
| `YOLO_MODEL` | `yolov8n.pt` | Detector model (`yolov8s.pt` for comparison) |
| `CONFIDENCE_THRESHOLD` | `0.25` | Inference confidence cutoff |
| `CROP_PADDING_FRACTION` | `0.10` | Padding around predicted box |
| `TRAIN_EPOCHS` | `100` | Maximum training epochs |
| `TRAIN_PATIENCE` | `15` | Early-stopping patience |
| `TRAIN_RATIO / VAL_RATIO / TEST_RATIO` | `0.70 / 0.15 / 0.15` | Dataset split ratios |
| `RANDOM_SEED` | `42` | Reproducibility seed |

## Assumptions

- Source images are organised as `data/Labeled Cow Eyes Data/{healthy,pinkeye}/`.
- Each image belongs to exactly one class, determined by the folder name.
- If `data/animal_ids.csv` is absent, stratified splitting by class label is used.
- YOLO annotation files use class ID `0` (the single "eye" class).

## License

MIT
