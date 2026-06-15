# TeX revision summary

## Files created

| File | Description |
|------|-------------|
| [`paper/PinkEye_Draft-V01_revised.tex`](paper/PinkEye_Draft-V01_revised.tex) | Full revised manuscript (original [`paper/PinkEye_Draft-V01.tex`](paper/PinkEye_Draft-V01.tex) unchanged). |
| [`tex_revision_summary.md`](tex_revision_summary.md) | This summary. |

## Major concerns addressed (aligned with `revision_concern_validation_report.md`)

- **YOLO dataset size (100 vs 117):** All incorrect “100” detector-corpus references removed; Stage~1 Methods now state **117** images, **117** validated labels, random shuffle split seed **42**, and independence from the 408-image Stage~2 pool.
- **Stage 1 vs Stage 2:** New **Dataset roles** paragraph, **Table `tab:dataset_summary`**, and explicit inference sentence replacing the erroneous “labels passed to stage 2.”
- **Data section (323 / defective):** Replaced with folder-level counts, post-crop counts, and 14 no-detection images; removed **defective**.
- **Leakage / CV:** Crop/image-file-level stratified folds; **animal IDs not used**; limitation text added in new **Limitations** section.
- **YOLO performance wording:** Validation **n = 11** stated; softened claims; downstream paragraph no longer says “highly reliable” / “strong foundation.”
- **DDPM:** Sentence before DDPM figure, revised figure caption, consolidated Discussion DDPM paragraph; Conclusion updated to state synthetic samples were **not** in quantitative benchmarks.
- **Reproducibility:** New **`\subsection{Implementation and reproducibility}`** with values from `reports/eye_detector_training_report.md` and existing manuscript protocol.
- **Baseline CNN+Transformer:** Expanded itemize text + **Table `tab:baseline_cnn_transformer`** (matches `src/models/cnn_transformer.py`).
- **Citations:** Removed **yaseen2024yolov9** from the Stage~1 opening (YOLOv8n + Roboflow text only; Ultralytics citation remains on the next paragraph). **`\citep{selvaraju2017gradcam}`** and **`\citep{rw2019timm}`** where applicable.
- **Table `tab:loc`:** Caption updated to internal validation metrics.
- **Table `tab:full_comparison`:** Replaced `table*` with `table[t]`; mean ± std in single cells; configuration labels aligned with narrative.
- **Terminology / grammar:** Stage~1/2 capitalization, “object localization,” “The following four…,” benchmark table caption (“Results are reported…”), **Pinkeye** confusion-matrix headers, **pinkeye** for minority-only sentence.

## Manuscript sections touched

- Abstract  
- Data (`\ref{dataset}`)  
- Methods — Stage~1, Stage~2, new Implementation subsection, workflow figure (unchanged body; caption unchanged)  
- Results — detector narrative, Table `tab:loc`, 408-image paragraph, augmentation intro, DDPM figure lead-in + caption, Table `benchmark` caption, Table `tab:full_comparison`  
- Discussion — DDPM consolidation, Grad-CAM `\citep`  
- **New:** `\section{Limitations}` before Conclusion  
- Conclusion — DDPM sentence clarified  

## Citation keys that may still need manual verification

- **`cas-refs.bib`** is not in this repository tree; confirm **`ultralytics2023yolov8`**, **`rw2019timm`**, **`selvaraju2017gradcam`**, **`ho2020ddpm`**, and all other keys resolve after `bibtex` / `biber`.
- Removed in-text use of **`yaseen2024yolov9`**; if it remains only in the `.bib` file, you may delete the unused entry when you edit the bibliography.

## Limitations not resolved without new experiments

- **Magnitude of optimistic bias** from possible correlated left/right or same-animal images across folds (acknowledged; no grouped CV run).
- **External / multi-site validation** for both detector and classifier (stated as future work).
- **DDPM utility for accuracy** (only qualitative pilot; no mixed training benchmark in this work).

---

Revised manuscript saved as `paper/PinkEye_Draft-V01_revised.tex` and summary saved as `tex_revision_summary.md`.
