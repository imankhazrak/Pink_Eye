"""
Shared configuration for the Step 1 eye-cropping pipeline.

All paths are relative to the project root (Pink_Eye/).
Edit the constants below to adjust the pipeline behaviour without
touching individual scripts.
"""

from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Source dataset ────────────────────────────────────────────────────────────
SOURCE_DATA_DIR = PROJECT_ROOT / "data" / "Labeled Cow Eyes Data"
CLASS_NAMES = ["healthy", "pinkeye"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Optional animal-ID metadata CSV.  If this file exists and contains an
# ``animal_id`` column, build_detection_dataset.py will use group-aware
# splitting so that images of the same animal stay in the same fold.
# Set to None or leave the file absent to fall back to stratified splitting.
ANIMAL_ID_CSV: Path | None = PROJECT_ROOT / "data" / "animal_ids.csv"

# ── Annotation ────────────────────────────────────────────────────────────────
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"

ANNOTATION_RULE = (
    "ANNOTATION RULE\n"
    "  - Draw ONE bounding box around the visible eye region.\n"
    "  - Include the eyeball / cornea and a small amount of surrounding\n"
    "    eyelid tissue.\n"
    "  - Do NOT include excessive background, ears, or hair.\n"
    "  - Keep the annotation style consistent across ALL images\n"
    "    (both healthy and pinkeye).\n"
    "  - When in doubt, prefer a slightly larger box over cutting off\n"
    "    disease signs."
)

# ── YOLO detection dataset ────────────────────────────────────────────────────
DETECTION_DATASET_DIR = PROJECT_ROOT / "detection_dataset"
YOLO_CONFIG_PATH = DETECTION_DATASET_DIR / "eye_detection.yaml"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ── YOLO model selection ──────────────────────────────────────────────────────
# Default baseline detector.  Change to "yolov8s.pt" for comparison.
#   yolov8n.pt  –  nano   (baseline, fast, lower capacity)
#   yolov8s.pt  –  small  (comparison, more capacity)
YOLO_MODEL = str(PROJECT_ROOT / "models" / "yolo" / "yolov8n.pt")

# ── Detection settings ────────────────────────────────────────────────────────
DETECTION_CLASS_ID = 0
DETECTION_CLASS_NAME = "eye"
TRAIN_IMGSZ = 640
TRAIN_EPOCHS = 100
TRAIN_PATIENCE = 15  # early-stopping patience

# ── Inference / cropping ──────────────────────────────────────────────────────
# Default confidence threshold.  Scripts also accept --conf to override.
CONFIDENCE_THRESHOLD = 0.25
CROP_PADDING_FRACTION = 0.10  # 10 % padding around predicted box

# ── Output directories ────────────────────────────────────────────────────────
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DETECTOR_DIR = OUTPUTS_DIR / "detector"
DETECTOR_WEIGHTS_DIR = DETECTOR_DIR / "weights"
DETECTOR_VALIDATION_DIR = DETECTOR_DIR / "validation"

# Two crop output trees: predicted boxes vs ground-truth boxes
EYE_CROPS_PRED_DIR = OUTPUTS_DIR / "eye_crops_pred"
EYE_CROPS_GT_DIR = OUTPUTS_DIR / "eye_crops_gt"

METADATA_DIR = OUTPUTS_DIR / "metadata"

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def ensure_dirs() -> None:
    """Create all output directories that may be needed."""
    for d in [
        ANNOTATIONS_DIR,
        DETECTION_DATASET_DIR,
        DETECTOR_WEIGHTS_DIR,
        DETECTOR_VALIDATION_DIR,
        EYE_CROPS_PRED_DIR / "healthy",
        EYE_CROPS_PRED_DIR / "pinkeye",
        EYE_CROPS_GT_DIR / "healthy",
        EYE_CROPS_GT_DIR / "pinkeye",
        METADATA_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def get_source_images() -> list[tuple[Path, str]]:
    """Return a list of (image_path, class_label) for every source image."""
    images = []
    for cls in CLASS_NAMES:
        cls_dir = SOURCE_DATA_DIR / cls
        if not cls_dir.exists():
            continue
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                images.append((p, cls))
    return images


def find_source_image(stem: str) -> tuple[Path, str] | None:
    """Locate a source image by filename stem across all class folders."""
    for cls in CLASS_NAMES:
        cls_dir = SOURCE_DATA_DIR / cls
        if not cls_dir.exists():
            continue
        for ext in IMAGE_EXTENSIONS:
            for suffix in (ext, ext.upper()):
                candidate = cls_dir / (stem + suffix)
                if candidate.exists():
                    return candidate, cls
    return None
