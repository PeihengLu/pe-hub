#!/usr/bin/env python3
"""Publication figures from ``paper_comparison.csv`` (vendor eval or scratch-benchmark).

Usage:
  python scripts/experiments/plot_base_model_eval.py \\
    scripts/experiments/base-model-eval/results/<RUN_ID>/paper_comparison.csv
  python scripts/experiments/plot_base_model_eval.py \\
    scripts/experiments/scratch-benchmark/results/<RUN_ID>
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass, replace
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
# Used for heatmap panel borders; close-match bars use the Pearson navy ramp.
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
PEARSON_NEG_POLE = STUDY_COLORS[2]
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
# DeepPE HCT/MDA (15-site endo, no author fold) and MinSePIE RC (n=57) are omitted.
BENCH_META = [
    ("deeppe-pooled__hek293t", "HEK", "DeepPE"),
    ("deepprime-clinvar", "ClinVar", "DeepPrime"),
    ("pridict1-library1", "Library 1", "PRIDICT1"),
    ("pridict2-library-diverse__hek293t", "HEK", "PRIDICT2 Library-Diverse"),
    ("pridict2-library-diverse__k562", "K562", "PRIDICT2 Library-Diverse"),
    ("pridict2-library-diverse__k562mlh1dn", "MLH1dn", "PRIDICT2 Library-Diverse"),
    ("minsepie-insert-pooled__hek293t__pe2", "HEK PE2", "MinSePIE"),
    ("optiprime-lib-mmr__hek293t__pe2", "HEK PE2", "OptiPrime Lib-MMR"),
    ("optiprime-lib-mmr__hek293t__pe4", "HEK PE4", "OptiPrime Lib-MMR"),
    ("optiprime-lib-mmr__hela__pe2", "HeLa PE2", "OptiPrime Lib-MMR"),
    ("optiprime-lib-mmr__hela__pe4", "HeLa PE4", "OptiPrime Lib-MMR"),
    ("optiprime-lib-cv__hek293t__pe2", "HEK PE2", "OptiPrime Lib-CV"),
    ("optiprime-lib-cv__hek293t__pe4", "HEK PE4", "OptiPrime Lib-CV"),
    ("optiprime-lib-cv__hela__pe2", "HeLa PE2", "OptiPrime Lib-CV"),
    ("optiprime-lib-cv__hela__pe4", "HeLa PE4", "OptiPrime Lib-CV"),
]
BENCH_ORDER = [(key, f"{study} {label}" if study not in label else label) for key, label, study in BENCH_META]

HEATMAP_PANELS = [
    BENCH_META[:7],
    BENCH_META[7:],
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

SCRATCH_MODEL_ORDER = [
    ("deepprime", "", "DeepPrime"),
    ("oped", "", "OPED"),
    ("pridict2", "", "PRIDICT2"),
]
SCRATCH_BENCH_META = [
    ("pridict1-library1", "Library 1", "PRIDICT1"),
    ("pridict2-library-diverse", "Diverse", "PRIDICT2 Library-Diverse"),
    ("deepprime-clinvar", "ClinVar", "DeepPrime"),
    ("deeppe-pooled", "Pooled", "DeepPE"),
    ("minsepie-insert-pooled", "Insert", "MinSePIE"),
    ("optiprime-lib-mmr", "Lib-MMR", "OptiPrime Lib-MMR"),
    ("optiprime-lib-cv", "Lib-CV", "OptiPrime Lib-CV"),
]


@dataclass(frozen=True)
class PlotLayout:
    name: str
    model_order: list[tuple[str, str, str]]
    bench_meta: list[tuple[str, str, str]]
    heatmap_panels: list[list[tuple[str, str, str]]]
    value_column: str
    cbar_label: str

    @property
    def bench_order(self) -> list[tuple[str, str]]:
        return [
            (key, f"{study} {label}" if study not in label else label)
            for key, label, study in self.bench_meta
        ]


BASE_LAYOUT = PlotLayout(
    name="base",
    model_order=MODEL_ORDER,
    bench_meta=BENCH_META,
    heatmap_panels=HEATMAP_PANELS,
    value_column="pearson_plot",
    cbar_label="Pearson r",
)
SCRATCH_LAYOUT = PlotLayout(
    name="scratch",
    model_order=SCRATCH_MODEL_ORDER,
    bench_meta=SCRATCH_BENCH_META,
    heatmap_panels=[SCRATCH_BENCH_META],
    value_column="spearman_plot",
    cbar_label="Spearman R",
)


def cell_fill_kind(
    row: Optional[dict[str, Any]],
    value_column: str = "pearson_plot",
) -> str:
    """How to draw one model × benchmark cell."""
    if row is None:
        return FILL_MISSING
    if row.get("plot_marker") == "author_fill" or row.get("value_source") == "author_fill":
        return FILL_AUTHOR
    if row.get("value_source") == "leak_unfilled":
        return FILL_MISSING
    if _f(row.get(value_column)) is None:
        return FILL_MISSING
    return FILL_MEASURED


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


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _scratch_bench(row: dict[str, Any]) -> str:
    name = str(row.get("dataset_name") or row.get("benchmark_name") or "")
    parts = name.split("__")
    return parts[-1] if parts else name


def load_scratch_cell_rows(run_dir: Path) -> list[dict[str, Any]]:
    """Load per-seed JSONL under a scratch-benchmark run directory."""
    by_key: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for path in sorted(run_dir.glob("*/results.jsonl")):
        if path.parent == run_dir:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row.get("model"), row.get("dataset_name"), row.get("repeat_id"))
            by_key[key] = row
    return list(by_key.values())


def scratch_to_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean metrics per model × pooled benchmark in paper_comparison.csv shape."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("model") or ""), _scratch_bench(row))].append(row)
    out: list[dict[str, Any]] = []
    for (model, bench), items in sorted(groups.items()):
        spears = [v for row in items if (v := _f(row.get("test_spearman"))) is not None]
        pears = [v for row in items if (v := _f(row.get("test_pearson"))) is not None]
        n_samples = [v for row in items if (v := _f(row.get("n_samples"))) is not None]
        spearman_mean = _mean(spears)
        pearson_mean = _mean(pears)
        plottable = spearman_mean is not None or pearson_mean is not None
        out.append(
            {
                "model": model,
                "pridict2_head": "",
                "benchmark_name": bench,
                "cell_line": "",
                "status": "ok" if any(row.get("status") == "ok" for row in items) else "missing",
                "value_source": "measured" if plottable else "leak_unfilled",
                "plot_marker": "measured" if plottable else "",
                "plot_hatch": "",
                "pearson_plot": pearson_mean if pearson_mean is not None else "",
                "spearman_plot": spearman_mean if spearman_mean is not None else "",
                "pearson_measured": pearson_mean if pearson_mean is not None else "",
                "spearman_measured": spearman_mean if spearman_mean is not None else "",
                "paper_pearson": "",
                "paper_spearman": "",
                "pearson_delta": "",
                "spearman_delta": "",
                "paper_citation": "",
                "paper_protocol_match": "",
                "n_samples": _mean(n_samples) if n_samples else "",
                "n_ok": len(items),
                "n_spearman": len(spears),
                "leak_reason": "" if plottable else "undefined_correlation",
            }
        )
    return out


def write_comparison(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise SystemExit(f"Error: no comparison rows to write to {path}")
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def infer_layout(rows: list[dict[str, Any]]) -> PlotLayout:
    benches = {str(row.get("benchmark_name") or "") for row in rows}
    if any(
        marker in bench
        for bench in benches
        for marker in ("__hek293t", "__hela", "__k562", "__mda")
    ):
        return BASE_LAYOUT
    scratch_keys = {meta[0] for meta in SCRATCH_BENCH_META}
    if benches & scratch_keys:
        return SCRATCH_LAYOUT
    heads = {str(row.get("pridict2_head") or "") for row in rows}
    if "HEK" in heads or "K562" in heads:
        return BASE_LAYOUT
    return BASE_LAYOUT


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
    layout: PlotLayout = BASE_LAYOUT,
) -> tuple[np.ndarray, np.ndarray]:
    n_models = len(layout.model_order)
    n_benches = len(panel)
    values = np.full((n_models, n_benches), np.nan)
    hatch = np.zeros((n_models, n_benches), dtype=bool)
    for i, (model, head, _label) in enumerate(layout.model_order):
        for j, (bench, _blabel, _study) in enumerate(panel):
            row = by_key.get((model, head, bench))
            if row is None:
                continue
            plot_r = _f(row.get(layout.value_column))
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
    model_order: list[tuple[str, str, str]] = MODEL_ORDER,
) -> Any:
    n_models, n_benches = values.shape
    im = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(n_benches))
    ax.set_xticklabels([label for _key, label, _study in panel], rotation=32, ha="right")
    ax.set_yticks(range(n_models))
    ax.set_yticklabels([label for _m, _h, label in model_order] if show_ylabel else [])
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


def plot_heatmap(
    rows: list[dict[str, Any]],
    out_path: Path,
    layout: PlotLayout = BASE_LAYOUT,
) -> Path:
    by_key = index_rows(rows)
    panels = layout.heatmap_panels
    matrices = [_heatmap_values(by_key, panel, layout) for panel in panels]
    finite = np.concatenate(
        [values[np.isfinite(values)] for values, _hatch in matrices if np.isfinite(values).any()]
    )
    data_max = float(np.nanmax(finite)) if finite.size else 1.0
    data_min = float(np.nanmin(finite)) if finite.size else -0.2
    vmin = min(-0.2, np.floor(data_min * 10.0) / 10.0)
    vmax = max(1.0, np.ceil(data_max * 10.0) / 10.0)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = _pearson_cmap(vmin, vmax)
    max_cols = max(len(panel) for panel in panels)
    n_panels = len(panels)

    if n_panels == 1:
        fig = plt.figure(figsize=(11.4, 3.8))
        grid = fig.add_gridspec(
            1, 1, left=0.12, right=0.82, top=0.90, bottom=0.28
        )
        axes = [fig.add_subplot(grid[0])]
        legend_anchor = (0.46, 0.08)
    else:
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
        legend_anchor = (0.46, 0.10)

    images = []
    for ax, panel, (values, hatch) in zip(axes, panels, matrices):
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
                model_order=layout.model_order,
            )
        )
    fig.canvas.draw()
    top_pos = axes[0].get_position()
    bot_pos = axes[-1].get_position()
    colorbar_ax = fig.add_axes(
        [top_pos.x1 + 0.005, bot_pos.y0, 0.016, top_pos.y1 - bot_pos.y0]
    )
    cbar = fig.colorbar(images[0], cax=colorbar_ax)
    cbar.set_label(layout.cbar_label, fontsize=12, color="#444444")
    cbar.set_ticks(np.arange(vmin, vmax + 1e-9, 0.2))
    cbar.ax.tick_params(labelsize=11, colors="#555555")
    cbar.outline.set_visible(False)
    legend_handles = [
        Patch(facecolor=MISSING_CELL, edgecolor="#D0D0D0", label="Not scored"),
    ]
    if layout.name == "base":
        legend_handles.append(
            Patch(facecolor="white", edgecolor="#888888", hatch="///", label="Author-reported fill")
        )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=legend_anchor,
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


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "comparison_csv",
        type=Path,
        help=(
            "paper_comparison.csv, or a scratch-benchmark results/<RUN_ID> "
            "directory (pools */results.jsonl)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Figure directory. Default: txt/diagrams for vendor eval; "
            "<run>/figures for a scratch-benchmark run directory."
        ),
    )
    parser.add_argument(
        "--layout",
        choices=("auto", "base", "scratch"),
        default="auto",
        help="Heatmap layout. auto infers from benchmark_name keys.",
    )
    parser.add_argument(
        "--metric",
        choices=("auto", "pearson", "spearman"),
        default="auto",
        help="Which comparison column to plot (default: pearson for base, spearman for scratch).",
    )
    args = parser.parse_args(argv)
    input_path = args.comparison_csv.resolve()
    if input_path.is_dir():
        cell_rows = load_scratch_cell_rows(input_path)
        if not cell_rows:
            raise SystemExit(f"Error: no cell results.jsonl under {input_path}")
        rows = scratch_to_comparison(cell_rows)
        csv_path = input_path / "paper_comparison.csv"
        write_comparison(csv_path, rows)
        print(f"Wrote {csv_path} ({len(rows)} rows)")
        default_out = input_path / "figures"
        inferred = SCRATCH_LAYOUT
    elif input_path.is_file():
        rows = load_comparison(input_path)
        default_out = DEFAULT_DIAGRAM_DIR
        inferred = infer_layout(rows)
    else:
        raise SystemExit(f"Error: {input_path} not found")

    if args.layout == "base":
        layout = BASE_LAYOUT
    elif args.layout == "scratch":
        layout = SCRATCH_LAYOUT
    else:
        layout = inferred

    if args.metric == "pearson":
        layout = replace(layout, value_column="pearson_plot", cbar_label="Pearson r")
    elif args.metric == "spearman":
        layout = replace(layout, value_column="spearman_plot", cbar_label="Spearman R")

    out_dir = args.out_dir or default_out
    apply_style()
    heatmap_name = (
        "eval_spearman_heatmap.pdf" if layout.value_column == "spearman_plot" else "eval_pearson_heatmap.pdf"
    )
    heat = plot_heatmap(rows, out_dir / heatmap_name, layout=layout)
    print(f"Wrote {heat}")
    print(f"Wrote {heat.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
