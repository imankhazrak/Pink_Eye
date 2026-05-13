#!/usr/bin/env python3
"""Generate publication-style figures for paper/ from committed CSV summaries.

Data sources: reports/step2_classification_benchmark_report.md (cross-check),
paper/data/*.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Academic style defaults
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)


def plot_dataset_distribution(csv_path: Path, out_pdf: Path) -> None:
    """Bar chart: Stage 2 folder counts (healthy vs pink_eye), total n=394.

    CSV columns:
      - ``class_name`` (``healthy`` / ``pink_eye``)
      - ``n_images``: **number of crops** in that folder (not the class label).

    Do not use a column named ``n`` for counts---older CSVs used ``n`` for
    label encoding (0/1), which incorrectly drew bars of height 0 and 1.
    """
    df = pd.read_csv(csv_path, comment="#")
    df.columns = [c.strip() for c in df.columns]

    count_col = next(
        (c for c in ("n_images", "count", "n_samples") if c in df.columns),
        None,
    )
    if count_col is None:
        raise ValueError(
            f"{csv_path}: need column n_images, count, or n_samples. "
            f"Got {list(df.columns)}"
        )

    name_col = next(
        (c for c in ("class_name", "class") if c in df.columns),
        None,
    )
    if name_col is None:
        raise ValueError(f"{csv_path}: need class_name or class column.")

    lookup = df.set_index(name_col)[count_col].astype(int).to_dict()

    # Fixed order: minority second bar matches manuscript wording.
    order = [
        ("healthy", "Healthy\n(class 0)", "#3C5488"),
        ("pink_eye", "Pink-eye\n(class 1)", "#E64B35"),
    ]
    counts: list[int] = []
    labels: list[str] = []
    colors: list[str] = []
    for key, lab, col in order:
        if key not in lookup:
            raise ValueError(f"Missing class_name={key!r} in {csv_path}")
        counts.append(int(lookup[key]))
        labels.append(lab)
        colors.append(col)

    total = sum(counts)
    if total != 394:
        raise ValueError(
            f"{csv_path}: expected 394 usable crops (309+85), got total={total}"
        )

    percents = [100.0 * c / total for c in counts]

    fig_w, fig_h = 4.2, 2.85
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    x = np.arange(len(labels))
    bars = ax.bar(x, counts, color=colors, edgecolor="black", linewidth=0.65, width=0.62)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Number of eye-crop images")
    ymax = max(counts) * 1.22
    ax.set_ylim(0, ymax)
    ax.set_title(
        "Stage 2 eye-crop classification dataset",
        fontsize=10,
        pad=8,
    )

    # Count and share of the 394 usable crops (309 vs 85; 78.4% vs 21.6%).
    for rect, c, p in zip(bars, counts, percents):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height() + ymax * 0.015,
            f"{int(c)}\n({p:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.text(
        0.5,
        -0.06,
        rf"Total usable crops $n={total}$",
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
        color="#333333",
    )

    fig.subplots_adjust(top=0.88, bottom=0.18)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_benchmark_f1_pinkeye(csv_path: Path, out_pdf: Path) -> None:
    df = pd.read_csv(csv_path)
    df = df.sort_values("f1_pinkeye_mean")
    models = df["model"].tolist()
    means = df["f1_pinkeye_mean"].values
    stds = df["f1_pinkeye_std"].values
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    y = np.arange(len(models))
    ax.barh(y, means, xerr=stds, color="#8172B2", edgecolor="black", linewidth=0.5, capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_xlabel(r"F1$_{\mathrm{PE}}$ (mean $\pm$ std, 5-fold CV)")
    ax.set_title("Pinkeye class F1 under baseline augmentation")
    ax.set_xlim(0.75, 1.0)
    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paper-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "paper",
    )
    args = parser.parse_args()
    data_dir = args.paper_root / "data"
    fig_dir = args.paper_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_dataset_distribution(data_dir / "dataset_counts.csv", fig_dir / "dataset_distribution.pdf")
    plot_benchmark_f1_pinkeye(data_dir / "step2_benchmark.csv", fig_dir / "benchmark_f1_pinkeye.pdf")
    print(f"Wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
