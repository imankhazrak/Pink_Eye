# Revision report: `paper/main.tex` (Pink_Eye)

Date: 2026-05-01

## Files inspected

- [`paper/main.tex`](main.tex) (rewritten structure and narrative)
- [`reports/step2_classification_benchmark_report.md`](../reports/step2_classification_benchmark_report.md)
- [`reports/step2_with_aug_vs_no_color_aug_report.md`](../reports/step2_with_aug_vs_no_color_aug_report.md)
- [`reports/step2_flip_rotation_control_report.md`](../reports/step2_flip_rotation_control_report.md)
- [`reports/step2_minority_only_augmentation_report.md`](../reports/step2_minority_only_augmentation_report.md)
- [`reports/eye_detector_training_report.md`](../reports/eye_detector_training_report.md)
- [`reports/advisor_summary_step1_step2.md`](../reports/advisor_summary_step1_step2.md)
- [`slurm/train_step2_benchmark.slurm`](../slurm/train_step2_benchmark.slurm) and companion Step 2 SLURM scripts
- [`src/training/train_step2.py`](../src/training/train_step2.py), [`src/dataset.py`](../src/dataset.py), [`src/step1/config.py`](../src/step1/config.py)
- [`.gitignore`](../.gitignore) (confirms `/outputs` excluded from version control)

## Experiment outputs used for numbers

All quantitative claims in **Results** trace to the Markdown reports above and/or the committed CSV summaries:

| Artifact | Purpose |
|----------|---------|
| [`paper/data/dataset_counts.csv`](data/dataset_counts.csv) | Healthy vs pinkeye counts (309 / 85) |
| [`paper/data/step2_benchmark.csv`](data/step2_benchmark.csv) | Mean±std metrics for Table / Figure (benchmark) |

Raw per-fold CSVs under `outputs/*/metrics/` were **not** available in the repository (gitignored); tables were already transcribed consistently in the Step 2 reports and carried forward into the paper.

## Figures generated or reused

| Output file | Source | Script |
|-------------|--------|--------|
| [`paper/figures/dataset_distribution.pdf`](figures/dataset_distribution.pdf) | `paper/data/dataset_counts.csv` | [`scripts/paper_plots.py`](../scripts/paper_plots.py) |
| [`paper/figures/benchmark_f1_pinkeye.pdf`](figures/benchmark_f1_pinkeye.pdf) | `paper/data/step2_benchmark.csv` | [`scripts/paper_plots.py`](../scripts/paper_plots.py) |
| Pipeline schematic | TikZ in `main.tex` | N/A (vector diagram, no numeric claims) |

## Figures removed or not shipped

- Fold-level confusion matrix, ROC, and learning-curve PNGs previously referenced under `outputs/benchmark_gpu/...` are **not** in the tracked tree (`/outputs` gitignored). The text states this explicitly.
- Grad-CAM example image previously under `outputs/benchmark_gpu/efficientnet/explainability/...` was removed; discussion cites the benchmark report and Grad-CAM literature instead.

## Tables updated

- Benchmark, per-fold, confusion matrix, no-colour delta, Gaussian-noise ablation, minority-only ablation, and full-configuration summary tables were validated against the corresponding `reports/step2_*.md` files.
- Fixed the previously broken LaTeX fragment (orphan `tabular` rows after the learning-curve figure) by restoring a complete `table` environment for the no-colour comparison (`tab:no_color_delta`).

## Sections modified in `main.tex`

- **Abstract**: Corrected ROC-AUC leadership (ResNet18 highest mean ROC-AUC in the benchmark table); avoided implying Strong CNN+Transformer wins every metric.
- **Methodology**: Split detector narrative (`prepare_eye_dataset.py`, Roboflow split) vs interactive `build_detection_dataset` (70/15/15) to prevent mixing two workflows; aligned Step 2 protocol with SLURM scripts (5 epochs, batch 8, patience 8); documented four augmentation policies with output-directory names; added reproducibility note for `src/dataset.py` vs March 2026 report snapshots; removed unverified global PyTorch/CUDA version strings for Stage 2 (cluster modules cited instead).
- **Results** (new `\section{Results}`): All empirical tables and benchmark plots.
- **Discussion** (new `\section{Discussion}`): Former discussion/limitations content; DDPM framed as code/config present without integrated classification evaluation.

## Bibliography

- Added `\bibitem{advisor_summary}` for the internal advisor markdown summary (408→394 crop narrative).
- Removed unused bibliography entries that were no longer cited (ViT, AutoAugment/RandAugment duplicates).

## LaTeX build

- **Class file**: `IEEEtran.cls` was copied into [`paper/IEEEtran.cls`](IEEEtran.cls) because the cluster TeX Live installation lacked it; compilation succeeds with `pdflatex main.tex` run from [`paper/`](.).
- **PDF output**: [`paper/main.pdf`](main.pdf)
- Remaining benign warnings may include float placement (`[h]`→`[ht]`) and caption package compatibility messages after removing `subcaption`.

## Assumptions and limitations

- Numeric tables match the March 2026 reports; if raw `outputs/` metrics are regenerated, numbers should be refreshed from those CSVs.
- Stage 1 framework versions (Ultralytics / PyTorch / CUDA) in the detector report are cited implicitly via the detector training narrative; Stage 2 uses module loads from `slurm/*.slurm` (e.g. CUDA 11.8) which differs from the detector log environment—both are documented separately.

## Recommendations

1. Optionally restore historical augmentation presets in [`src/dataset.py`](../src/dataset.py) to match `reports/step2_with_aug_vs_no_color_aug_report.md` for bitwise reproducibility.
2. Commit small exported summaries from `outputs/*/metrics/*_summary_metrics.csv` under `paper/data/` whenever benchmarks are re-run, to tighten the audit trail without committing large binary artifacts.
3. If `IEEEtran` is installed system-wide, delete the bundled [`paper/IEEEtran.cls`](IEEEtran.cls) to avoid duplication.
