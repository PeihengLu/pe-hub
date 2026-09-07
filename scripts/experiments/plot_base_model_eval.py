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
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch, Rectangle


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIAGRAM_DIR = REPO_ROOT / "txt" / "diagrams"

# Same Tableau-style study palette as ``txt/diagrams/generate_data_summary.py``.
STUDY_COLORS = [
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
]
SUMMARY_BLUE = STUDY_COLORS[0]
SUMMARY_ORANGE = STUDY_COLORS[1]
SUMMARY_RED = STUDY_COLORS[2]
SUMMARY_TEAL = STUDY_COLORS[3]
SUMMARY_GREEN = STUDY_COLORS[4]
SUMMARY_PURPLE = STUDY_COLORS[5]
MISSING_CELL = "#F4F4F4"


def _hex_to_rgb(color: str) -> tuple[float, float, float]:
    raw = color.lstrip("#")
    return tuple(int(raw[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(max(0, min(255, int(round(c * 255)))) for c in rgb))


def _lerp_hex(start: str, end: str, weight: float) -> str:
    t = min(max(weight, 0.0), 1.0)
    rgb = tuple(a + (b - a) * t for a, b in zip(_hex_to_rgb(start), _hex_to_rgb(end)))
    return _rgb_to_hex(rgb)


# Full-strength poles at |r|=1 so a unit of Pearson r has the same color change
# on both sides. Displayed range is typically −0.2…1.0.
PEARSON_NEG_POLE = SUMMARY_RED
PEARSON_POS_POLE = "#1F4E79"


def _color_at_pearson(value: float) -> str:
    if value >= 0:
        return _lerp_hex("#FFFFFF", PEARSON_POS_POLE, value)
    return _lerp_hex("#FFFFFF", PEARSON_NEG_POLE, -value)


def _pearson_cmap(vmin: float, vmax: float) -> LinearSegmentedColormap:
    """White at r=0; red/blue poles at ±1 so the positive ramp is not washed out."""
    span = vmax - vmin
    if span <= 0:
        cmap = LinearSegmentedColormap.from_list("pehub_pearson", ["#FFFFFF", "#FFFFFF"])
        cmap.set_bad(MISSING_CELL)
        return cmap
    stops = np.linspace(0.0, 1.0, 21)
    colors = [(float(stop), _color_at_pearson(vmin + stop * span)) for stop in stops]
    cmap = LinearSegmentedColormap.from_list("pehub_pearson", colors)
    cmap.set_bad(MISSING_CELL)
    return cmap


# Display order for the heatmap (rows / columns).
MODEL_ORDER = [
    ("deepprime", "", "DeepPrime"),
    ("oped", "", "OPED"),
    ("optiprime", "", "OptiPrime"),
    ("pridict2", "HEK", "PRIDICT2 HEK"),
    ("pridict2", "K562", "PRIDICT2 K562"),
]

# (benchmark_key, column label, study group label)
BENCH_META = [
    ("deeppe-pooled__hek293t", "HEK", "DeepPE"),
    ("deeppe-pooled__hct116", "HCT", "DeepPE"),
    ("deeppe-pooled__mda_mb_231", "MDA", "DeepPE"),
    ("deepprime-clinvar", "ClinVar", "DeepPrime"),
    ("pridict1-library1", "Library 1", "PRIDICT1"),
    ("pridict2-library-diverse__hek293t", "HEK", "PRIDICT2 Library-Diverse"),
    ("pridict2-library-diverse__k562", "K562", "PRIDICT2 Library-Diverse"),
    ("pridict2-library-diverse__k562mlh1dn", "MLH1dn", "PRIDICT2 Library-Diverse"),
    ("optiprime-lib-mmr__hek293t__pe2", "HEK PE2", "OptiPrime Lib-MMR"),
    ("optiprime-lib-mmr__hek293t__pe4", "HEK PE4", "OptiPrime Lib-MMR"),
    ("optiprime-lib-mmr__hela__pe2", "HeLa PE2", "OptiPrime Lib-MMR"),
    ("optiprime-lib-mmr__hela__pe4", "HeLa PE4", "OptiPrime Lib-MMR"),
    ("optiprime-lib-cv__hek293t__pe2", "HEK PE2", "OptiPrime Lib-CV"),
    ("optiprime-lib-cv__hek293t__pe4", "HEK PE4", "OptiPrime Lib-CV"),
    ("optiprime-lib-cv__hela__pe2", "HeLa PE2", "OptiPrime Lib-CV"),
    ("optiprime-lib-cv__hela__pe4", "HeLa PE4", "OptiPrime Lib-CV"),
    ("minsepie-insert-pooled__hek293t__pe2", "HEK PE2", "MinSePIE"),
]
BENCH_ORDER = [(key, f"{study} {label}" if study not in label else label) for key, label, study in BENCH_META]

# Two heatmap rows; MinSePIE RC (n=57) is dropped.
HEATMAP_PANELS = [
    BENCH_META[:8],
    BENCH_META[8:],
]

STUDY_GROUP_COLORS = {
    "DeepPE": STUDY_COLORS[0],
    "DeepPrime": STUDY_COLORS[1],
    "PRIDICT1": STUDY_COLORS[2],
    "PRIDICT2 Library-Diverse": STUDY_COLORS[3],
    "OptiPrime Lib-MMR": STUDY_COLORS[4],
    "OptiPrime Lib-CV": STUDY_COLORS[5],
    "MinSePIE": STUDY_COLORS[6],
}

FILL_MEASURED = "measured"
FILL_AUTHOR = "author_fill"
FILL_MISSING = "missing"

# One hue per model, taken from the data-summary study palette.
MODEL_COLORS = {
    "DeepPrime": SUMMARY_ORANGE,
    "OPED": SUMMARY_BLUE,
    "OptiPrime": SUMMARY_TEAL,
    "PRIDICT2 HEK": SUMMARY_PURPLE,
    "PRIDICT2 K562": SUMMARY_GREEN,
}

# Extra x-gap after these labels so library families stay readable.
FAMILY_GAP_AFTER = {
    "DeepPE MDA": 0.45,
    "PRIDICT1 Library 1": 0.45,
    "PRIDICT2 Library-Diverse MLH1dn": 0.45,
    "OptiPrime Lib-CV HeLa PE4": 0.45,
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
    sns.set_theme(style="white", context="notebook", font_scale=1.05)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#444444",
            "axes.titlecolor": "#222222",
            "axes.titleweight": "600",
            "xtick.color": "#555555",
            "ytick.color": "#555555",
            "grid.color": "#E6E6E6",
            "grid.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.dpi": 600,
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


def _label_on_color(hex_color: str) -> str:
    raw = hex_color.lstrip("#")
    red, green, blue = int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)
    luminance = (0.299 * red + 0.587 * green + 0.114 * blue) / 255.0
    return "#222222" if luminance > 0.62 else "#FFFFFF"


def _study_spans(panel: list[tuple[str, str, str]]) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    start = 0
    while start < len(panel):
        study = panel[start][2]
        end = start
        while end + 1 < len(panel) and panel[end + 1][2] == study:
            end += 1
        spans.append((start, end, study))
        start = end + 1
    return spans


def _heatmap_values(
    by_key: dict[tuple[str, str, str], dict[str, Any]],
    panel: list[tuple[str, str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    n_models = len(MODEL_ORDER)
    n_benches = len(panel)
    values = np.full((n_models, n_benches), np.nan)
    hatch = np.zeros((n_models, n_benches), dtype=bool)
    for i, (model, head, _label) in enumerate(MODEL_ORDER):
        for j, (bench, _blabel, _study) in enumerate(panel):
            row = by_key.get((model, head, bench))
            if row is None:
                continue
            plot_r = _f(row.get("pearson_plot"))
            if plot_r is None:
                continue
            values[i, j] = plot_r
            hatch[i, j] = row.get("plot_marker") == "author_fill"
    return values, hatch


def _points_to_data(ax: Any, points: float, *, horizontal: bool) -> float:
    """Convert a length in typographic points to axis data units."""
    bbox = ax.get_position()
    fig_w, fig_h = ax.figure.get_size_inches()
    if horizontal:
        axis_in = bbox.width * fig_w
        data_range = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
    else:
        axis_in = bbox.height * fig_h
        data_range = abs(ax.get_ylim()[1] - ax.get_ylim()[0])
    if axis_in <= 0:
        return 0.0
    return (points / 72.0) * (data_range / axis_in)


def _draw_heatmap_panel(
    ax: Any,
    values: np.ndarray,
    hatch: np.ndarray,
    panel: list[tuple[str, str, str]],
    *,
    cmap: Any,
    norm: Any,
    max_cols: int,
    show_ylabel: bool,
) -> Any:
    n_models, n_benches = values.shape
    im = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(n_benches))
    ax.set_xticklabels([label for _key, label, _study in panel], rotation=32, ha="right")
    ax.set_yticks(range(n_models))
    ax.set_yticklabels([label for _m, _h, label in MODEL_ORDER] if show_ylabel else [])
    ax.tick_params(length=0, labelsize=10, colors="#555555")
    ax.set_xticks(np.arange(-0.5, n_benches, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_models, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(n_models):
        for j in range(n_benches):
            value = values[i, j]
            if not np.isfinite(value):
                continue
            text_color = "white" if abs(value) >= 0.55 else "#444444"
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

    header_h = 0.46
    ax.set_xlim(-0.5, max_cols - 0.5)
    ax.set_ylim(n_models - 0.5, -0.5 - header_h - 0.04)
    body_lw = 2.2
    pad_x = 0.5 * _points_to_data(ax, body_lw, horizontal=True)
    pad_y = 0.5 * _points_to_data(ax, body_lw, horizontal=False)
    for start, end, study in _study_spans(panel):
        color = STUDY_GROUP_COLORS[study]
        width = end - start + 1
        ax.add_patch(
            Rectangle(
                (start - 0.5 + pad_x, -0.5 + pad_y),
                width - 2.0 * pad_x,
                n_models - 2.0 * pad_y,
                fill=False,
                edgecolor=color,
                linewidth=body_lw,
                zorder=5,
                clip_on=False,
                joinstyle="miter",
            )
        )
        ax.add_patch(
            Rectangle(
                (start - 0.5, -0.5 - header_h),
                width,
                header_h,
                facecolor=color,
                edgecolor=color,
                linewidth=0.0,
                zorder=6,
                clip_on=False,
            )
        )
        ax.text(
            (start + end) / 2.0,
            -0.5 - header_h / 2.0,
            study,
            ha="center",
            va="center",
            fontsize=7.5 if len(study) > 16 else 9,
            fontweight="600",
            color=_label_on_color(color),
            zorder=7,
            clip_on=False,
        )

    return im


def plot_heatmap(rows: list[dict[str, Any]], out_path: Path) -> Path:
    by_key = index_rows(rows)
    matrices = [_heatmap_values(by_key, panel) for panel in HEATMAP_PANELS]
    finite = np.concatenate([values[np.isfinite(values)] for values, _hatch in matrices if np.isfinite(values).any()])
    data_max = float(np.nanmax(finite)) if finite.size else 1.0
    data_min = float(np.nanmin(finite)) if finite.size else -0.2
    vmin = min(-0.2, np.floor(data_min * 10.0) / 10.0)
    vmax = max(1.0, np.ceil(data_max * 10.0) / 10.0)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = _pearson_cmap(vmin, vmax)
    max_cols = max(len(panel) for panel in HEATMAP_PANELS)

    fig = plt.figure(figsize=(14.4, 6.6))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 1.0],
        hspace=0.24,
        left=0.10,
        right=0.82,
        top=0.96,
        bottom=0.18,
    )
    axes = [fig.add_subplot(grid[0]), fig.add_subplot(grid[1])]
    images = []
    for ax, panel, (values, hatch) in zip(axes, HEATMAP_PANELS, matrices):
        images.append(
            _draw_heatmap_panel(
                ax,
                values,
                hatch,
                panel,
                cmap=cmap,
                norm=norm,
                max_cols=max_cols,
                show_ylabel=True,
            )
        )
    fig.canvas.draw()
    top_pos = axes[0].get_position()
    bot_pos = axes[1].get_position()
    colorbar_ax = fig.add_axes(
        [top_pos.x1 + 0.005, bot_pos.y0, 0.016, top_pos.y1 - bot_pos.y0]
    )
    cbar = fig.colorbar(images[0], cax=colorbar_ax)
    cbar.set_label("Pearson r", fontsize=12, color="#444444")
    cbar.set_ticks(np.arange(vmin, vmax + 1e-9, 0.2))
    cbar.ax.tick_params(labelsize=11, colors="#555555")
    cbar.outline.set_visible(False)
    fig.legend(
        handles=[
            Patch(facecolor=MISSING_CELL, edgecolor="#D0D0D0", label="Not scored / leak unfilled"),
            Patch(facecolor="white", edgecolor="#888888", hatch="///", label="Author-reported fill"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.46, 0.10),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#D0D0D0",
        framealpha=1.0,
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.18)
    fig.savefig(
        out_path.with_suffix(".png"),
        dpi=600,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.18,
    )
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
        help=(
            "Figure directory. Default: txt/diagrams for the heatmap "
            "(same folder as data_composition.png)."
        ),
    )
    parser.add_argument(
        "--all-figures",
        action="store_true",
        help="Also write grouped-bar and close-match vs-paper figures.",
    )
    args = parser.parse_args(argv)
    csv_path = args.comparison_csv.resolve()
    if not csv_path.is_file():
        raise SystemExit(f"Error: {csv_path} not found")
    out_dir = args.out_dir or DEFAULT_DIAGRAM_DIR
    apply_style()
    rows = load_comparison(csv_path)
    heat = plot_heatmap(rows, out_dir / "eval_pearson_heatmap.pdf")
    print(f"Wrote {heat}")
    print(f"Wrote {heat.with_suffix('.png')}")
    if args.all_figures:
        bars = plot_benchmark_bars(rows, out_dir / "eval_benchmark_bars.pdf")
        vs_paper = plot_vs_paper(rows, out_dir / "eval_vs_paper.pdf")
        print(f"Wrote {bars}")
        print(f"Wrote {vs_paper}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
