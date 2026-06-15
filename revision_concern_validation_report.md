# Revision Concern Validation Report

## 1. Executive Summary

This report evaluates ten reviewer-style concerns against the **Pink_Eye** repository and `paper/PinkEye_Draft-V01.tex` only (no new experiments, no retraining, no regeneration of results).

**Overall:** Most concerns are **valid or partially valid as manuscript clarity, consistency, citation, or limitation-framing issues**, not as evidence that the reported numbers contradict the archived project reports. The repository and internal reports consistently support **117** Stage~1 detector images (split **93 / 11 / 13**), a **408**-image Stage~2 head-image pool, **394** usable crops (**309** healthy, **85** pinkeye), and **14** no-detection images. Quantitative classifier tables align with `reports/step2_classification_benchmark_report.md`, `reports/step2_with_aug_vs_no_color_aug_report.md`, and `paper/data/step2_benchmark.csv`.

**Validity at a glance**

| Concern | Validity |
|--------|----------|
| 1. YOLO dataset size (100 vs 117) | **Valid** (internal manuscript inconsistency; repo supports 117) |
| 2. Stage 1 vs Stage 2 dataset separation | **Partially valid** (numbers correct in places; narrative needs one clear summary and removal of misleading sentences) |
| 3. Train–val leakage (animal / paired eyes) | **Partially valid** (evidence supports image/crop-level stratified folds only; animal IDs not present in repo) |
| 4. YOLO performance wording vs small val/test | **Partially valid** (metrics match reports; generalization language should be softened) |
| 5. DDPM role in quantitative benchmark | **Valid** as clarity issue (DDPM is pilot / qualitative; not the source of Tables benchmark–full comparison) |
| 6. Reproducibility | **Partially valid** (many details exist in reports/SLURM/code; manuscript omits some; current `src/dataset.py` vs March 2026 augmentation descriptions needs honest alignment) |
| 7. Baseline CNN+Transformer detail | **Partially valid** (code specifies more than the manuscript) |
| 8. Citations (YOLOv8 vs YOLOv9 key, etc.) | **Partially valid** (clear key mismatch in `.tex`; full `.bib` not in repo) |
| 9. Table / figure consistency | **Partially valid** (numeric consistency largely good; Table `tab:full_comparison` formatting and a few narrative tighten-ups) |
| 10. Writing / terminology | **Valid** (minor grammar and inconsistent terms) |

**No new experiments are required** to address these concerns for a revision round; the remaining gap for **animal-level leakage** is epistemic (cannot be quantified without identifiers or new grouped splits)—address with **transparent limitations** and **future work**, as requested.

---

## 2. Evidence Sources Inspected

| Source | Evidence provided |
|--------|-------------------|
| `paper/PinkEye_Draft-V01.tex` | Full manuscript text; inconsistencies (100 vs 117; “stratification” for detector; DDPM duplication; `\cite` vs `\citep`); tables and figure captions. |
| `paper/PinkEye_Draft-V01.pdf` | **Not found** in the workspace tree at report time; assessment relies on `.tex` and project reports. |
| `README.md` | Pipeline overview; optional `animal_ids.csv` for **Step 1** `build_detection_dataset` (70/15/15); Step~2 default paths and CLI. |
| `reports/eye_detector_training_report.md` | **117** images; **93/11/13** split; validation metrics (precision, recall, mAP50, mAP50-95); **11** val images; Ultralytics **8.4.21**, PyTorch **2.8.0**, CUDA **12.8**, **V100**; 408→394/14 on classification pool. |
| `reports/advisor_summary_step1_step2.md` | End-to-end summary; DDPM configs; **explicit** “planned usage” for synthetic mixing and re-benchmarking (not completed in that narrative). |
| `reports/step2_classification_benchmark_report.md` | 394 crops; 309/85; 5-fold CV; table matching manuscript benchmark. |
| `reports/step2_with_aug_vs_no_color_aug_report.md` | Definitions of with_aug vs no_color_aug; deltas matching manuscript Tables `tab:no_color_delta` / narrative. |
| `reports/step2_flip_rotation_control_report.md` | Flip/rotation vs noise ablation context. |
| `reports/step2_minority_only_augmentation_report.md` | Minority-only augmentation context. |
| `paper/main_revision_report.md` | Prior internal audit: **`src/dataset.py` vs March 2026 augmentation snapshots**; `outputs/` gitignored. |
| `paper/data/dataset_counts.csv` | 309 healthy, 85 `pink_eye`. |
| `paper/data/step2_benchmark.csv` | Mean/std for four models; matches benchmark table. |
| `scripts/prepare_eye_dataset.py` | **Random** 80/10/10 split, seed **42**, from `data/annotated_data/train/` → `data/dataset/`. |
| `runs/detect/train/args.yaml` | YOLOv8n, epochs 100, imgsz 640, batch 16; framework `seed: 0` (Ultralytics run card). |
| `slurm/train_step2_benchmark.slurm` | Step~2: 5 folds, 5 epochs, batch **8**, weighted CE, patience **8**, `outputs/eye_crops_classification`. |
| `src/dataset.py` | `StratifiedKFold` on **labels**; metadata keyed by **filename**; no animal column. |
| `src/training/cross_validation.py` | Orchestrates fold training; no grouped split. |
| `src/training/trainer.py` | Class weights `counts.sum() / (2.0 * counts)` for binary weighted CE. |
| `src/training/train_step2.py` | Defaults: image size 224, lr 1e-4, weight decay 1e-4, seed 42, patience 8, min-delta 1e-4. |
| `src/models/cnn_transformer.py` | Baseline hybrid: **ResNet18** backbone, `embed_dim=256`, **`num_heads=8`**, **`num_layers=2`**, pre-norm Transformer, **mean pool**, dropout **0.1**. |
| `src/models/cnn_transformer_strong.py` | EfficientNet-B0, CLS token, L=3, h=4, d=256, MLP head as implemented. |

---

## 3. Concern-by-Concern Validation and Handling Plan

### Concern 1: YOLO dataset size inconsistency (100 vs 117)

- **Reviewer concern:** The manuscript states YOLO was trained on **100** images in places while Table `yoloconfig` and Results give **93/11/13** (total **117**).
- **Repository/manuscript evidence:** In `PinkEye_Draft-V01.tex`, lines **167–169** say “100 cattle head images”, “All 100 annotation files”, and “All 100 labels are passed to the stage 2.” Results (**277**) correctly state “117 cow images” with split 93/11/13. `reports/eye_detector_training_report.md` (Sections 2–4) and `scripts/prepare_eye_dataset.py` support **117** paired image–label files and **93/11/13** counts.
- **Validity judgment:** **Valid** as an **internal manuscript inconsistency**; **not valid** as a claim that experiments used the wrong detector dataset size.
- **Why this judgment is appropriate:** Repository documentation and the Results section agree on **117**; the Stage~1 Methods paragraphs are factually wrong and the sentence about “labels passed to stage 2” is incorrect (Stage~2 uses **408** head images and separate disease labels).
- **How to address without new experiments:** Replace all “100” with **117** where it refers to the detector corpus; delete or rewrite the “labels passed to stage 2” sentence; optionally add one explicit sentence: “Stage~1 detector training used **117** Roboflow eye-annotation images, disjoint from the **408**-image Stage~2 classification pool.”
- **Exact recommended manuscript text (replace the opening paragraph of `\subsection{Stage~1: Eye Localization with YOLOv8}`):**

```latex
Since the goal is to predict whether the eye shows signs of IBK, the first step is to localize the eye in each cattle head image. You Only Look Once (YOLO; \citealp{redmon2016you}) is a widely used family of detectors for fast object localization. We fine-tuned a YOLOv8n detector on \textbf{117} cattle head images with Roboflow-exported eye bounding boxes (single class \texttt{eye}, YOLO-normalized coordinates). All \textbf{117} label files were validated for image--label pairing, coordinate ranges, and class ID${}=0$ prior to training. The dataset was split deterministically (random shuffle, seed~42) into training (80\,\%, 93 images), validation (10\,\%, 11 images), and test (10\,\%, 13 images), matching the logged dataset build in \texttt{scripts/prepare\_eye\_dataset.py} and \texttt{reports/eye\_detector\_training\_report.md}.
```

**Also replace** the sentence on line **169** (“All 100 labels are passed…”) **with:**

```latex
The trained detector was then applied to all 408 Stage~2 head images (Sec.~\ref{dataset}) for inference-time cropping; this inference pass is independent of the 117-image detector training corpus.
```

- **Risk if not addressed:** Reviewers will treat this as sloppiness or suspect undisclosed dataset changes.

---

### Concern 2: Stage 1 and Stage 2 dataset separation

- **Reviewer concern:** Unclear distinction between YOLO training data and pinkeye classification data.
- **Repository/manuscript evidence:** `eye_detector_training_report.md`: Stage~1 **117** annotated images (Roboflow); Stage~2 pool **408** images (`data/sample_test` in report; manuscript **150**, **218**, **293**). Cropping yields **394** usable crops, **14** no detection; **309** healthy, **85** pinkeye (`dataset_counts.csv`, reports). Manuscript **150** still says “defective (85)” vs folder naming `pinkeye` / `pink_eye` elsewhere.
- **Validity judgment:** **Partially valid** (counts largely correct; **structure** of the narrative should be tightened).
- **Why:** Evidence supports the user’s “known values”; the gap is **pedagogical** (one consolidated table + explicit “disjoint corpora” language).
- **How to address:** Add a **dataset summary table** early in **Data** or **Methods**; in Stage~1 subsection, never imply Stage~2 images were used to train the detector.
- **Exact recommended manuscript text (short paragraph after the first paragraph of `\section{Data}`):**

```latex
\textbf{Dataset roles.} Stage~1 and Stage~2 rely on different image pools. The eye detector was trained on 117 Roboflow-annotated cattle head images with only eye localisation labels. The IBK classification benchmark uses 408 field images collected at CFS with binary healthy/pinkeye labels; after detector cropping, 394 images yielded usable eye crops (309 healthy, 85 pinkeye) and 14 images had no passing detection (set aside for manual review).
```

- **Risk if not addressed:** Readers may assume leakage between detector training and classifier evaluation images.

---

### Concern 3: Potential train–validation leakage in Stage 2 (paired eyes / same animal)

- **Reviewer concern:** Stratified 5-fold CV at crop/image level may put left/right or repeated animals in both train and val, inflating metrics.
- **Repository/manuscript evidence:** `src/dataset.py` `make_fold_indices` uses `sklearn.model_selection.StratifiedKFold` on **binary labels** only (**102–117**). `cross_validation.py` splits `metadata` rows (**68–72**). No `animal_ids.csv` found under the repo root (glob **0** hits). README notes optional `animal_ids.csv` for **Step 1** `build_detection_dataset`, not for Step~2 code shown. Manuscript **150** states two lateral images per steer—so **paired eyes are plausible**, but **animal IDs are not used** in the inspected Step~2 splitting code.
- **Validity judgment:** **Partially valid** (structural risk is real; **magnitude** cannot be established from repository files alone).
- **Why:** Code proves **label-stratified image/crop-level** folds, **not** grouped-by-animal folds. Filenames (e.g. repeated prefixes) are **not** documented as animal IDs.
- **How to address without new experiments:** Methods: state explicitly that folds are **stratified by class at the crop (file) level**; Discussion/Limitations: state that **animal-level or session-level dependence was not modeled**; recommend **grouped cross-validation** as **future work** (do not rerun here).
- **Exact recommended manuscript text:**

**Methods (append to the CV paragraph, ca. line 252):**

```latex
Folds were constructed by stratifying on the binary label at the \emph{crop} (image file) level using scikit-learn's \texttt{StratifiedKFold} (shuffle enabled, random seed~42). Animal identifiers were not used to constrain folds; therefore, left/right views from the same handling session could appear in different folds.
```

**Limitations (new or extended bullet):**

```latex
Because each fold splits individual labeled crops rather than animals or capture sessions, estimated performance may be optimistically biased if correlated images of the same animal appear across train and validation folds. Animal-level grouped evaluation was outside the scope of this dataset release and is an important next step.
```

- **Risk if not addressed:** Reviewers may flag optimistic bias without acknowledging it.

---

### Concern 4: YOLO detector performance may be overclaimed

- **Reviewer concern:** Very high metrics with small val/test sets may overstate field generalization.
- **Repository/manuscript evidence:** `reports/eye_detector_training_report.md`: precision **1.000**, recall **0.999**, mAP50 **0.995**, mAP50-95 **0.631**; **11** validation images, **11** instances; test pass **13/13**. Manuscript **277–287** says “near-perfect” and “highly reliable” for downstream use.
- **Validity judgment:** **Partially valid** (metrics faithfully reported; **wording** may exceed evidence).
- **Why:** Metrics match the report; small **n** for val/test is a **statistical limitation**, not a numerical error.
- **How to address:** Tie numbers explicitly to the **validation split** (and optional **13-image test** pass from the report); qualify language as **internal** localization on this annotated corpus.
- **Exact recommended manuscript text (replace sentences in Results paragraph ca. 277–278):**

```latex
On the held-out \textbf{validation} split ($n=11$ images), the fine-tuned YOLOv8n detector achieved precision $1.000$, recall $0.999$, mAP@0.5 $0.995$, and mAP@0.5:0.95 $0.631$ (Table~\ref{tab:loc}), indicating strong agreement with the Roboflow eye boxes under this evaluation protocol. A separate 13-image test split showed correct detections in all images in the logged run, but both splits are small; metrics should be interpreted as \emph{internal} localization performance on this annotation distribution rather than field-wide generalization.
```

**Discussion softening (one sentence to add):**

```latex
Detector metrics were computed on small validation/test subsets relative to deployment variability; prospective evaluation on larger, independently collected head-image sets is needed before claiming robust chute-side or pasture-level localization.
```

- **Risk if not addressed:** Claims of “near-perfect” localization may be read as externally validated performance.

---

### Concern 5: DDPM synthetic augmentation ambiguity

- **Reviewer concern:** Unclear if DDPM images entered Tables **benchmark** through **tab:full_comparison**.
- **Repository/manuscript evidence:** `reports/advisor_summary_step1_step2.md` Section **6.4** lists synthetic mixing and re-benchmarking as **planned**, not completed in that document. Step~2 benchmark reports and CSVs (`paper/data/step2_benchmark.csv`) describe **394 real crops** only. Manuscript **305–322** shows DDPM in a **qualitative** figure; **527** repeats DDPM motivation twice.
- **Validity judgment:** **Valid** as a **presentation/clarity** issue; **not valid** as an accusation that tables silently used synthetic images (no evidence in reports/CSVs).
- **How to address:** State once clearly: **all quantitative tables used real crops only**; DDPM is **pilot / illustrative**; move duplicated DDPM prose to one paragraph.
- **Exact recommended manuscript text:**

**Results (one sentence before or in the DDPM figure paragraph, ca. 305):**

```latex
All quantitative results in Tables~\ref{benchmark}--\ref{tab:full_comparison} were obtained from real Stage~2 crops only; class-conditional DDPM samples were not mixed into training for those benchmarks.
```

**Figure `ddpm_qualitative1` caption replacement:**

```latex
\caption{Qualitative pilot examples only (not used in the quantitative benchmark). For each class, one real Stage~2 crop is shown beside a representative sample from a class-conditional DDPM trained on cropped-eye data. DDPM integration into Stage~2 training and evaluation under the benchmark protocol is left to future work.}
```

**Discussion (replace the redundant second DDPM paragraph on line 527 with one consolidated paragraph ending):**

```latex
We trained class-conditional DDPMs as a \emph{pilot} exploration of synthetic appearance diversity (configuration details are recorded in project documentation). Synthetic images were not used in the reported 5-fold classifier benchmarks; systematic curation, mixed real/synthetic training, and re-evaluation under the same protocol remain future work.
```

**Conclusion:** Keep “future work” framing; avoid implying DDPM already improved Table metrics.

- **Risk if not addressed:** Suspected **train set contamination** or **misleading figure**.

---

### Concern 6: Reproducibility details

- **Reviewer concern:** Insufficient detail to reproduce.
- **Repository/manuscript evidence:** Manuscript already states much of Step~2 (224 input, AdamW, lr/wd, weighted CE formula narrative, 5-fold seed 42, patience, batch 8, 5 epochs) **252–256**. YOLO table **192–211** matches `eye_detector_training_report.md`. **Gaps:** Ultralytics/PyTorch/CUDA **for Step~2** vs detector (SLURM `cuda/11.8.0` vs detector report CUDA **12.8**); `runs/detect/train/args.yaml` shows Ultralytics `seed: 0` while split uses **42**; **`src/dataset.py`** current transforms are **lighter** than `reports/step2_with_aug_vs_no_color_aug_report.md` “with_aug” bullet list—`paper/main_revision_report.md` already flags this (**[not found in repository]:** exact `torch`/`torchvision` versions for the Step~2 cluster job unless user commits `pip freeze` from that environment).
- **Validity judgment:** **Partially valid**.
- **How to address:** Add a subsection consolidating **verified** numbers from reports + SLURM + code; document **code–report alignment** for augmentations honestly (either “matches March 2026 cluster snapshot documented in reports” or update code and re-run—**the latter is out of scope here**; for this revision, **describe what the benchmark reports define** and cite report paths).
- **Risk:** “Not reproducible” if versions and augmentation definitions diverge.

*(LaTeX for subsection 6 is provided in Section 6 below, omitting any value not verified.)*

---

### Concern 7: Baseline CNN+Transformer architecture detail

- **Reviewer concern:** Manuscript description insufficient vs “Strong” variant.
- **Repository/manuscript evidence:** Manuscript **224** gives a high-level path. Code `src/models/cnn_transformer.py`: backbone **`resnet18`** (default in `build_cnn_transformer`), **`out_indices=(4,)`**, `1×1` conv projection to **`embed_dim=256`**, **`LayerNorm` on tokens**, **`TransformerEncoder` with `norm_first=True` (pre-norm)**, **`num_layers=2`**, **`num_heads=8`**, **`dim_feedforward=embed_dim*4`**, dropout **0.1**, **mean pooling**, head **`LayerNorm` + `Linear(256, num_classes)`**.
- **Validity judgment:** **Partially valid**.
- **How to address:** Add a small table or 2–3 precise sentences mirroring the code; align “final-stage CNN” with **ResNet18 last feature map**.
- **Risk:** “Vague hybrid” criticism.

---

### Concern 8: Citation concerns (YOLOv8 vs YOLOv9 key, timm, bibliography)

- **Reviewer concern:** `\citep{yaseen2024yolov9}` cited for YOLOv8; general cleanup.
- **Repository/manuscript evidence:** Line **167** pairs `YOLOv8` with `yaseen2024yolov9` (key name suggests YOLOv9). Line **190** already cites `ultralytics2023yolov8`. No `cas-refs.bib` in repository; cannot verify full bibliographic entries or years/venues from files alone.
- **Validity judgment:** **Partially valid** (clear **key/topic mismatch risk** on line 167).
- **How to address:** Replace `yaseen2024yolov9` on the first YOLOv8 mention with **`ultralytics2023yolov8`** (or a dedicated YOLOv8 paper if the bibliography is updated accordingly—**manual verification outside repo** for the optimal canonical citation). Keep `rw2019timm` if the `.bib` entry follows the publisher’s Harvard style; fix line **525** `\cite{selvaraju2017gradcam}` → `\citep{...}` for consistency with `natbib` usage elsewhere.
- **Risk:** Obvious citation–implementation mismatch.

---

### Concern 9: Table and figure consistency

- **Reviewer concern:** Tables/figures/captions inconsistent or unclear.
- **Repository/manuscript evidence:** **Table `benchmark` vs `tab:per_fold`:** means and std for ResNet18 and Strong CNN+Transformer match arithmetic from per-fold rows (**381–388** vs **338–348**). **Confusion matrix `tab:confusion`:** $302+7+9+76=394$ (**404–405**). **Table `tab:no_color_delta`:** consistent with `reports/step2_with_aug_vs_no_color_aug_report.md`. **Table `tab:full_comparison` (490–515):** numeric values align with reports; **layout** places std on a second row under an empty model column—easy to misread as a second model. **Figure workflow (265):** already mentions DDPM as pilot/future—good. **DDPM figure (320):** should explicitly say “not in benchmark” (Concern 5). **Table `tab:loc`:** caption “Eye Detection Accuracy” is imprecise (detection metrics, not classification accuracy).
- **Validity judgment:** **Partially valid** (mostly numeric consistency; **formatting/clarity** fixes needed).
- **How to address:** Rename Table `tab:loc` caption; reformat `tab:full_comparison` using `\multicolumn` or “mean ± std” in one row per configuration.
- **Risk:** Presentation looks like hiding std or mis-labeling metrics.

---

### Concern 10: Writing, grammar, and terminology

- **Reviewer concern:** Grammar, inconsistent disease terminology, awkward phrasing.
- **Repository/manuscript evidence:** Examples: **150** “pink eye” vs usual “pinkeye”; **150** “healthy (323 images) and defective (85)” vs **309** healthy **crops**—**323 vs 309** is a major inconsistency (323 = head images labeled healthy before crop failures). **162** “objective detection” → “object detection”. **167** “accurate objective detection approach” → grammar. Mixed “Pinkeye” capital P in body text (**305**). “Pink-eye” in confusion matrix header (**402**) vs “pinkeye” elsewhere. Abstract “diseased” vs “IBK-affected” optional consistency.
- **Validity judgment:** **Valid** (editorial).
- **How to address:** Adopt **IBK** first mention + **pinkeye** colloquial; use **healthy vs pinkeye** for classes; replace **defective** with **pinkeye-positive** or **pinkeye-labeled**. Clarify head-image vs crop counts in one sentence, for example: “The classification pool contains 408 lateral head images with folder labels (323 healthy, 85 pinkeye); after Stage~1 cropping, 394 images yielded usable eye crops (309 healthy, 85 pinkeye) and 14 images had no passing detection.” The counts are mutually consistent because $323-309=14$ (all reported failures are consistent with coming from the healthy-labeled head-image subset under this arithmetic), while the pinkeye-labeled count is unchanged at 85.
- **Risk:** **323 vs 309** looks like a hard error unless the head-image vs post-detection crop distinction is stated explicitly.

---

## 4. Recommended Manuscript Changes by Section

- **Abstract:** Optional: replace “diseased” with “IBK-affected” or “pinkeye-positive” for precision; no numeric change needed if abstract stays aligned with 394 crops.
- **Data (`\ref{dataset}`):** Fix **323 healthy / defective** narrative: distinguish **head-image folder counts** from **post-cropping usable crop counts**; add the dataset summary table (Section 5); align “pink eye” → “pinkeye” per house style.
- **Methods – Stage 1:** Replace **100** → **117**; fix “stratification” → **random split (seed 42)** unless you truly used stratified splitting **[not found in `prepare_eye_dataset.py`]**; remove erroneous “labels passed to stage 2”; optionally note Ultralytics run `seed: 0` separately from dataset split seed (**from `runs/detect/train/args.yaml`**).
- **Methods – Stage 2:** Add explicit **crop-level stratified K-fold** and **no animal grouping**; clarify weighted CE implementation matches code $N/(2n_c)$ for $K=2$.
- **Results – Detector:** Soften claims; specify **validation n**; optional footnote on test **n=13**.
- **Results – DDPM figure:** Caption clarification (Section 3, Concern 5).
- **Discussion / Conclusion:** Consolidate DDPM text; add detector **small-n** caveat; add **possible correlated folds** limitation.
- **Limitations:** New subsection or bullets (Section 7).
- **References:** Fix YOLOv8 primary cite; harmonize `\citep`; full `.bib` cleanup **requires manual pass** (file not in repo).
- **Tables:** Fix `tab:loc` caption; reformat `tab:full_comparison`; ensure Table 10 std rows read clearly.
- **Figures:** Workflow caption is already strong; align DDPM caption with “pilot only.”

---

## 5. Proposed Dataset Summary Table

```latex
\begin{table}[t]
  \caption{Summary of image corpora used in Stage~1 (eye detection) and Stage~2 (IBK classification on crops). Stage~1 training data are separate from the Stage~2 field-image pool.}
  \label{tab:dataset_summary}
  \centering
  \begin{tabular}{llr}
    \toprule
    \textbf{Component} & \textbf{Description} & \textbf{Count} \\
    \midrule
    Stage~1 detector training pool & Roboflow-annotated cattle head images (single-class eye boxes) & 117 \\
    Stage~1 split & Train / val / test & 93 / 11 / 13 \\
    Stage~2 classification pool & CFS head images with healthy/pinkeye folder labels & 408 \\
    Stage~2 usable crops & After YOLO inference at conf.\ 0.25 (usable for classifiers) & 394 \\
    Stage~2 no detection & No box above threshold (set aside for review) & 14 \\
    Stage~2 healthy crops & Usable healthy-labeled crops & 309 \\
    Stage~2 pinkeye crops & Usable pinkeye-labeled crops & 85 \\
    \bottomrule
  \end{tabular}
\end{table}
```

---

## 6. Proposed Implementation and Reproducibility Subsection

Use only **verified** items below. (Do **not** paste bracketed notes into the manuscript.)

```latex
\subsection{Implementation and reproducibility}

\textbf{Software and hardware (Stage~1).}
The eye detector was trained with the Ultralytics YOLO CLI using a YOLOv8n checkpoint pretrained on MS-COCO, $640\times 640$ inputs, batch size 16, and 100 epochs, as recorded in \texttt{runs/detect/train/args.yaml} and summarized in \texttt{reports/eye\_detector\_training\_report.md}.
That report documents one executed environment as Ultralytics 8.4.21, PyTorch 2.8.0, and CUDA 12.8 on a Tesla V100-PCIE-16GB GPU.

\textbf{Dataset construction (Stage~1).}
The 117 annotated images were split 80/10/10 (train/val/test) with random seed 42 using \texttt{scripts/prepare\_eye\_dataset.py}, yielding 93/11/13 images.

\textbf{Software and launch (Stage~2).}
Classifier benchmarks were executed with \texttt{python -m src.training.train\_step2} as submitted through the SLURM template \texttt{slurm/train\_step2\_benchmark.slurm}, which loads \texttt{python/3.10} and \texttt{cuda/11.8.0} on the cluster module stack.
Each run wrote a JSON run card under the chosen output root (e.g., \texttt{outputs/benchmark\_gpu/<model>/logs/<model>\_run\_config.json}) alongside per-fold metrics CSVs.

\textbf{Stage~2 training protocol (shared across reported benchmarks).}
Input resolution $224\times 224$; AdamW optimizer with learning rate $10^{-4}$ and weight decay $10^{-4}$; 5-fold stratified cross-validation (scikit-learn, shuffle enabled) with random seed 42; batch size 8; up to five epochs per fold with early stopping on validation F1 for the pinkeye class (patience 8, minimum improvement $10^{-4}$); weighted cross-entropy with class weights computed from the training-fold label counts using the implementation in \texttt{src/training/trainer.py} (binary inverse-frequency scaling).
Grad-CAM visualizations were enabled for CNN backbones in the training utility defaults.

\textbf{Augmentation reporting alignment.}
Augmentation policies referenced in the ablation reports (e.g., \texttt{reports/step2\_with\_aug\_vs\_no\_color\_aug\_report.md}) describe the transforms used in the March~2026 benchmark jobs; readers should treat those report definitions as the authoritative specification of each policy label (\emph{full}, \emph{no-color}, \emph{flip+rotation}, \emph{minority-only}) when reproducing tables.
```

---

## 7. Proposed Limitations Section

```latex
\section{Limitations}

This study has several limitations relevant to interpretation and deployment.

\textbf{Dataset size and imbalance.}
The usable pinkeye-positive crop set contains 85 examples; while weighted loss and stratified folds stabilize training, estimates of rare-class performance remain high-variance and should be updated as more positives are collected.

\textbf{Evaluation dependence and possible correlation across folds.}
Cross-validation splits were performed at the labeled crop (image file) level with class-stratification only. Animal identifiers were not available in the released splitting code; consequently, correlated views (e.g., left/right images from the same animal or repeated sessions) could appear in both training and validation folds within a split, which may inflate performance relative to strictly animal-held-out evaluation.

\textbf{Eye localization sample size.}
Detector metrics were computed on small validation and test subsets (11 and 13 images in the logged configuration). These results support internal consistency with the annotation distribution but do not substitute for large-scale, multi-farm validation.

\textbf{External validity.}
All field images were collected under a single facility protocol with defined handling and camera settings; generalization to other farms, breeds, lighting, and camera hardware is unknown without external test sets.

\textbf{Diffusion-based augmentation.}
Class-conditional DDPM samples were explored qualitatively to assess plausibility of synthetic appearances; they were not used in the quantitative classifier benchmarks reported here. Any future integration requires curation and independent evaluation under the same reporting protocol.
```

---

## 8. Proposed Response-to-Reviewer Language

1. **Dataset size inconsistency:** “We thank the reviewer. The Stage~1 detector corpus contains **117** annotated images (93/11/13). Occurrences of ‘100’ in the submitted Methods were typographical inconsistencies; we corrected them and aligned all sections with the logged dataset build.”

2. **Stage 1 vs 2 separation:** “We clarified that detector training images (117, Roboflow eye boxes) differ from the classification pool (408 CFS images) and added Table~X summarizing counts and flows (408→394 crops; 14 no detection).”

3. **Leakage:** “Splits were stratified by disease label at the crop level using standard scikit-learn routines, without animal-level grouping because animal IDs were not incorporated in the Step~2 splitting code we release. We now state this explicitly and discuss potential optimistic bias; animal-held-out evaluation is important future work.”

4. **YOLO claims:** “We retained the logged validation metrics but revised wording to emphasize small validation/test sizes and internal localization performance on this annotation distribution.”

5. **DDPM:** “All tables in the quantitative benchmark used **real** crops only. DDPM panels are illustrative pilot outputs; we revised captions and discussion to prevent any implication that synthetic images contributed to the reported metrics.”

6. **Reproducibility:** “We expanded Methods with cluster launch details, hyperparameters, and pointers to logged run cards and internal reports that document augmentation policy definitions.”

7. **CNN+Transformer baseline:** “We expanded the baseline hybrid description to match the implementation (ResNet18 feature map, token projection, transformer depth/width/heads, pooling, and head).”

8. **Citations:** “We corrected the YOLOv8 citation linkage and harmonized citation commands; the bibliography was checked for consistency.” *(Bibliographic completeness may still require manual verification against your reference manager.)*

9. **Tables/figures:** “We relabeled the detection metric table, clarified std presentation in the augmentation summary table, and aligned DDPM captions with their non-quantitative role.”

10. **Writing:** “We standardized terminology (pinkeye vs IBK) and corrected grammar.”

---

## 9. Local Edits and Terminology Cleanup

**Terminology policy (recommended)**

- Use **infectious bovine keratoconjunctivitis (IBK)** on first mention in major sections; use **pinkeye** thereafter for readability.
- For **classes**, prefer **healthy** vs **pinkeye** (or **pinkeye-positive**) in tables; avoid **defective**.
- Use **pinkeye** (one word, lowercase) in prose; reserve **Pink-eye** only if matching a table header style—prefer **Pinkeye** column headers without hyphen for consistency.
- Use **object detection**, not “objective detection.”

**High-priority numeric/clarity fixes**

- Reconcile **323 healthy images** at head-image level with **309 healthy crops** after detection (explain **14** failed detections if that matches your failure analysis **[verify against `outputs/metadata/` if present locally; not committed in this tree]**).
- **Line 169:** delete incorrect “100 labels passed to stage 2.”

**Grammar / LaTeX micro-edits (non-exhaustive)**

- **162:** “stage 1 detect” → “Stage~1 detects”; “stage 2 classify” → “Stage~2 classifies.”
- **167:** “fast and accurate objective detection approach” → “fast and accurate object detection.”
- **298:** “Four following training augmentations” → “The following four training augmentation policies.”
- **525:** `\cite{selvaraju2017gradcam}` → `\citep{selvaraju2017gradcam}` (match `natbib` style).
- **150:** “SilencerTM” → use `\textsuperscript{TM}` or publisher trademark rule.

---

## 10. Final Priority Checklist

- [ ] Replace **every** incorrect “100” with **117** in Stage~1 Methods; fix annotation validation count.
- [ ] Remove/replace the erroneous sentence about “labels passed to stage 2.”
- [ ] Add **Table `tab:dataset_summary`** (Section 5) and a short “dataset roles” paragraph in **Data**.
- [ ] Fix **323 vs 309** narrative in **Data** by separating **head-image** counts from **crop** counts.
- [ ] Correct Stage~1 split description: **random 80/10/10 (seed 42)**, not “stratification” **[per `prepare_eye_dataset.py`]**.
- [ ] Soften YOLO generalization language; state **val n=11** (and test n=13 if reported).
- [ ] Add **crop-level stratified K-fold** + **no animal grouping** + limitation text.
- [ ] Add explicit sentence: **DDPM not used in quantitative tables**; revise DDPM caption(s); deduplicate Discussion DDPM paragraph.
- [ ] Expand **baseline CNN+Transformer** description (ResNet18; 2 layers; 8 heads; mean pool; dropout 0.1) and optional small architecture table.
- [ ] Fix **YOLOv8** citation on line 167 (`yaseen2024yolov9` → appropriate YOLOv8/Ultralytics key); harmonize `\cite`/`\citep`.
- [ ] Rename Table `tab:loc` caption (“Detection metrics” rather than “Accuracy”).
- [ ] Reformat Table `tab:full_comparison` so std is unambiguous.
- [ ] Pass: **object detection** grammar; consistent **pinkeye** terminology.

---

**Note:** Raw per-fold CSVs and many `outputs/` artifacts are **gitignored** per `paper/main_revision_report.md`; the manuscript should continue to cite **committed** `reports/*.md` and `paper/data/*.csv` as the auditable numerical sources unless you choose to commit small exported summaries later (that would be outside this no-new-experiments task).
