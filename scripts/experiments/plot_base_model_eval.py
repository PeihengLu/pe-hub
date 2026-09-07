#!/usr/bin/env python3
"""Publication figures from ``paper_comparison.csv``.

Usage:
  python scripts/experiments/plot_base_model_eval.py \\
    results/base_model_eval/<RUN_ID>/paper_comparison.csv
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle


# Display order for the heatmap (rows / columns).
MODEL_ORDER = [
    ("deepprime", "", "DeepPrime"),
    ("oped", "", "OPED"),
    ("optiprime", "", "OptiPrime"),
    ("pridict2", "HEK", "PRIDICT2 HEK"),
    ("pridict2", "K562", "PRIDICT2 K562"),
]

BENCH_ORDER = [
    ("deeppe-pooled__hek293t", "DeepPE HEK"),
    ("deeppe-pooled__hct116", "DeepPE HCT"),
    ("deeppe-pooled__mda_mb_231", "DeepPE MDA"),
    ("deepprime-clinvar", "ClinVar"),
    ("pridict1-library1", "Library 1"),
    ("pridict2-library-diverse__hek293t", "Diverse HEK"),
    ("pridict2-library-diverse__k562", "Diverse K562"),
    ("pridict2-library-diverse__k562mlh1dn", "Diverse MLH1dn"),
    ("optiprime-lib-mmr__hek293t__pe2", "Lib-MMR HEK PE2"),
    ("optiprime-lib-mmr__hek293t__pe4", "Lib-MMR HEK PE4"),
    ("optiprime-lib-mmr__hela__pe2", "Lib-MMR HeLa PE2"),
    ("optiprime-lib-mmr__hela__pe4", "Lib-MMR HeLa PE4"),
    ("optiprime-lib-cv__hek293t__pe2", "Lib-CV HEK PE2"),
    ("optiprime-lib-cv__hek293t__pe4", "Lib-CV HEK PE4"),
    ("optiprime-lib-cv__hela__pe2", "Lib-CV HeLa PE2"),
    ("optiprime-lib-cv__hela__pe4", "Lib-CV HeLa PE4"),
    ("minsepie-insert-pooled__hek293t", "MinSePIE HEK"),
    ("minsepie-insert-pooled__rc", "MinSePIE RC"),
]

FILL_MEASURED = "measured"
FILL_AUTHOR = "author_fill"
FILL_MISSING = "missing"

# Wong-inspired, colorblind-safe. One hue per model; fill pattern encodes source.
MODEL_COLORS = {
    "DeepPrime": "#0072B2",
    "OPED": "#E69F00",
    "OptiPrime": "#009E73",
    "PRIDICT2 HEK": "#CC79A7",
    "PRIDICT2 K562": "#56B4E9",
}

# Extra x-gap after these labels so library families stay readable.
FAMILY_GAP_AFTER = {
    "DeepPE MDA": 0.45,
    "Library 1": 0.45,
    "Diverse MLH1dn": 0.45,
    "Lib-CV HeLa PE4": 0.45,
}


def cell_fill_kind(row: Optional[dict[str, Any]]) -> str:
    """How to draw one model × benchmark cell."""
    if row is None:
        return FILL_MISSING
    if row.get("plot_marker") == "author_fill" or row.get("value_source") == "author_fill":
        return FILL_AUTHOR
    if row.get("value_source") == "leak_unfilled":
        return FILL_MISSING
    if _f(row.get("pearson_plot")) is None:
        return FILL_MISSING
    return FILL_MEASURED


CLOSE_MATCH_ORDER = [
    ("deepprime", "", "deepprime-clinvar", "DeepPrime\nClinVar"),
    ("pridict2", "HEK", "pridict2-library-diverse__hek293t", "PRIDICT2 HEK\nDiverse HEK"),
    ("pridict2", "K562", "pridict2-library-diverse__k562", "PRIDICT2 K562\nDiverse K562"),
]


def _f(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def load_comparison(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("model") or ""),
        str(row.get("pridict2_head") or ""),
        str(row.get("benchmark_name") or ""),
    )


def index_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {_row_key(row): row for row in rows}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#222222",
            "axes.titlecolor": "#111111",
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.dpi": 400,
            "hatch.linewidth": 0.7,
        }
    )


def plot_benchmark_bars(rows: list[dict[str, Any]], out_path: Path) -> Path:
    """Double-column grouped bars: model color + fill pattern for data source."""
    by_key = index_rows(rows)
    n_models = len(MODEL_ORDER)
    bar_width = 0.14
    bar_pad = 0.018
    cluster = n_models * bar_width + (n_models - 1) * bar_pad

    centers: list[float] = []
    labels: list[str] = []
    cursor = 0.0
    for bench_key, bench_label in BENCH_ORDER:
        centers.append(cursor)
        labels.append(bench_label)
        cursor += 1.0 + FAMILY_GAP_AFTER.get(bench_label, 0.0)

    fig, ax = plt.subplots(figsize=(9.6, 5.05))
    missing_stub_half = 0.045

    for model_index, (model, head, model_label) in enumerate(MODEL_ORDER):
        color = MODEL_COLORS[model_label]
        offset = (model_index - (n_models - 1) / 2) * (bar_width + bar_pad)
        xs: list[float] = []
        heights: list[float] = []
        bottoms: list[float] = []
        kinds: list[str] = []
        for center, (bench_key, _label) in zip(centers, BENCH_ORDER):
            row = by_key.get((model, head, bench_key))
            kind = cell_fill_kind(row)
            xs.append(center + offset)
            kinds.append(kind)
            if kind == FILL_MISSING:
                heights.append(2 * missing_stub_half)
                bottoms.append(-missing_stub_half)
            else:
                value = _f(row.get("pearson_plot")) if row else None
                heights.append(float(value) if value is not None else 0.0)
                bottoms.append(0.0)

        measured_x, measured_h = [], []
        author_x, author_h = [], []
        missing_x, missing_h, missing_b = [], [], []
        for x, height, bottom, kind in zip(xs, heights, bottoms, kinds):
            if kind == FILL_MEASURED:
                measured_x.append(x)
                measured_h.append(height)
            elif kind == FILL_AUTHOR:
                author_x.append(x)
                author_h.append(height)
            else:
                missing_x.append(x)
                missing_h.append(height)
                missing_b.append(bottom)

        if measured_x:
            ax.bar(
                measured_x,
                measured_h,
                width=bar_width,
                color=color,
                edgecolor=color,
                linewidth=0.4,
                zorder=3,
            )
        if author_x:
            ax.bar(
                author_x,
                author_h,
                width=bar_width,
                facecolor="white",
                edgecolor=color,
                linewidth=0.9,
                hatch="///",
                zorder=3,
            )
        if missing_x:
            ax.bar(
                missing_x,
                missing_h,
                width=bar_width,
                bottom=missing_b,
                facecolor="white",
                edgecolor="#8A8A8A",
                linewidth=0.7,
                hatch="xxx",
                linestyle="dotted",
                zorder=2,
            )

    ax.axhline(0.0, color="#666666", linewidth=0.7, zorder=1)
    ax.set_ylabel("Pearson r")
    ax.set_ylim(-0.65, 1.02)
    ax.set_xlim(centers[0] - cluster / 2 - 0.15, centers[-1] + cluster / 2 + 0.15)
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.yaxis.grid(True, color="#E8E8E8", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_title(
        "Base-model Pearson r across PE-hub benchmarks",
        loc="left",
        fontsize=11,
        pad=8,
    )
    ax.tick_params(length=3)

    model_handles = [
        Patch(facecolor=MODEL_COLORS[label], edgecolor=MODEL_COLORS[label], label=label)
        for _model, _head, label in MODEL_ORDER
    ]
    fill_handles = [
        Patch(facecolor="#555555", edgecolor="#555555", label="PE-hub measured"),
        Patch(
            facecolor="white",
            edgecolor="#555555",
            hatch="///",
            label="Author-reported fill",
        ),
        Patch(
            facecolor="white",
            edgecolor="#8A8A8A",
            hatch="xxx",
            linestyle="dotted",
            label="Not scored (leak)",
        ),
    ]
    legend_models = ax.legend(
        handles=model_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=5,
        frameon=False,
        fontsize=8,
        handlelength=1.1,
        columnspacing=1.1,
    )
    ax.add_artist(legend_models)
    ax.legend(
        handles=fill_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.40),
        ncol=3,
        frameon=False,
        fontsize=8,
        handlelength=1.6,
        columnspacing=1.4,
    )
    fig.subplots_adjust(bottom=0.32, left=0.08, right=0.995, top=0.92)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)
    return out_path


def plot_heatmap(rows: list[dict[str, Any]], out_path: Path) -> Path:
    by_key = index_rows(rows)
    n_models = len(MODEL_ORDER)
    n_benches = len(BENCH_ORDER)
    values = np.full((n_models, n_benches), np.nan)
    hatch = np.zeros((n_models, n_benches), dtype=bool)

    for i, (model, head, _label) in enumerate(MODEL_ORDER):
        for j, (bench, _blabel) in enumerate(BENCH_ORDER):
            row = by_key.get((model, head, bench))
            if row is None:
                continue
            plot_r = _f(row.get("pearson_plot"))
            if plot_r is None:
                continue
            values[i, j] = plot_r
            hatch[i, j] = row.get("plot_marker") == "author_fill"

    fig, ax = plt.subplots(figsize=(15.4, 4.6))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#F4F4F4")
    finite = values[np.isfinite(values)]
    vmax = max(0.95, float(np.nanmax(finite)) if finite.size else 1.0)
    vmin = min(-0.2, float(np.nanmin(finite)) if finite.size else -0.2)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    im = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(n_benches))
    ax.set_xticklabels([label for _key, label in BENCH_ORDER], rotation=55, ha="right")
    ax.set_yticks(range(n_models))
    ax.set_yticklabels([label for _m, _h, label in MODEL_ORDER])
    ax.set_title("Pearson r on PE-hub base-model evaluation", loc="left", fontsize=13, pad=10)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, n_benches, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_models, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(n_models):
        for j in range(n_benches):
            value = values[i, j]
            if not np.isfinite(value):
                continue
            text_color = "white" if abs(value) >= 0.55 else "#111111"
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
                zorder=3,
            )
            if hatch[i, j]:
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor="none",
                        hatch="///",
                        edgecolor="#222222",
                        linewidth=0.6,
                        zorder=2,
                    )
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Pearson r", fontsize=10)
    cbar.outline.set_visible(False)
    ax.legend(
        handles=[
            Patch(facecolor="#F4F4F4", edgecolor="#CCCCCC", label="Not scored / leak unfilled"),
            Patch(facecolor="white", edgecolor="#888888", hatch="///", label="Author-reported fill"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.42),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)
    return out_path


def plot_vs_paper(rows: list[dict[str, Any]], out_path: Path) -> Path:
    by_key = index_rows(rows)
    categories: list[str] = []
    measured: list[float] = []
    paper: list[float] = []
    for model, head, bench, label in CLOSE_MATCH_ORDER:
        row = by_key.get((model, head, bench))
        if row is None:
            continue
        meas = _f(row.get("pearson_measured"))
        pub = _f(row.get("paper_pearson"))
        if meas is None or pub is None:
            continue
        categories.append(label)
        measured.append(meas)
        paper.append(pub)

    x = np.arange(len(categories))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.bar(
        x - width / 2,
        measured,
        width,
        label="PE-hub",
        color="#2F6FED",
        edgecolor="none",
    )
    ax.bar(
        x + width / 2,
        paper,
        width,
        label="Paper",
        color="#B8B8B8",
        edgecolor="none",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Pearson r")
    ax.set_ylim(0.6, 1.0)
    ax.set_title("Close-match protocol cells vs paper-reported Pearson", loc="left", fontsize=12)
    ax.yaxis.grid(True, color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    for index, (left, right) in enumerate(zip(measured, paper)):
        ax.text(index - width / 2, left + 0.008, f"{left:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(index + width / 2, right + 0.008, f"{right:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)
    return out_path


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "comparison_csv",
        type=Path,
        help="Path to paper_comparison.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Figure directory (default: <csv-dir>/figures)",
    )
    args = parser.parse_args(argv)
    csv_path = args.comparison_csv.resolve()
    if not csv_path.is_file():
        raise SystemExit(f"Error: {csv_path} not found")
    out_dir = args.out_dir or (csv_path.parent / "figures")
    apply_style()
    rows = load_comparison(csv_path)
    bars = plot_benchmark_bars(rows, out_dir / "eval_benchmark_bars.pdf")
    heat = plot_heatmap(rows, out_dir / "eval_pearson_heatmap.pdf")
    vs_paper = plot_vs_paper(rows, out_dir / "eval_vs_paper.pdf")
    print(f"Wrote {bars}")
    print(f"Wrote {heat}")
    print(f"Wrote {vs_paper}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
