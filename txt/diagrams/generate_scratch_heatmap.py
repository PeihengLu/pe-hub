#!/usr/bin/env python3
"""Generate from-scratch base-weight heatmaps for the term paper (Results).

Layout matches the vendor base-model eval: two panels, columns grouped by
study with coloured headers. Metrics are mean ± s.d. over holdout_3 seeds.

Data sources (post-fix OPED; split Library-Diverse / Hsu):
  - OPED:            results/20260917T222613Z only (do not fall back to pooled)
  - DeepPrime/PRIDICT2 split cells: 20260917T153347Z
  - Single-condition Library 1 / ClinVar / DeepPE / MinSePIE: 20260911T065340Z
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
RESULTS = REPO / "scripts/experiments/scratch-benchmark/results"

OPED_RUN = "20260917T222613Z"
SPLIT_RUN = "20260917T153347Z"
POOLED_RUN = "20260911T065340Z"

# Font sizes (pt) — edit these to retune the figure.
FONT_CELL = 10.0          # mean ± s.d. inside each cell
FONT_DATASET = 10.0       # study / dataset group headers
FONT_XTICK = 10.0         # column labels (HEK, ClinVar, …)
FONT_YTICK = 10.0         # model names (DeepPrime, OPED, …)
FONT_LEGEND_LABEL = 12.0  # colorbar title (Pearson r / Spearman ρ)
FONT_LEGEND_TICK = 11.0   # colorbar tick numbers
FONT_TITLE = 12.0         # figure title

# Same palette poles as plot_base_model_eval.py
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
PEARSON_POS_POLE = "#1F4E79"
MISSING_CELL = "#F4F4F4"

STUDY_GROUP_COLORS = {
    "DeepPE": STUDY_COLORS[0],
    "DeepPrime": STUDY_COLORS[1],
    "PRIDICT1": STUDY_COLORS[2],
    "PRIDICT2 Library-Diverse": STUDY_COLORS[3],
    "MinSePIE": STUDY_COLORS[6],
    "OptiPrime Lib-MMR": STUDY_COLORS[4],
    "OptiPrime Lib-CV": STUDY_COLORS[5],
}

MODELS = ["deepprime", "oped", "pridict2"]
MODEL_LABELS = {
    "deepprime": "DeepPrime",
    "oped": "OPED",
    "pridict2": "PRIDICT2",
}

# (bench_key, column label, study group) — same study order as vendor heatmap.
BENCH_META: list[tuple[str, str, str]] = [
    ("deeppe-pooled", "HEK", "DeepPE"),
    ("deepprime-clinvar", "ClinVar", "DeepPrime"),
    ("pridict1-library1", "Library 1", "PRIDICT1"),
    ("pridict2-library-diverse__hek293t", "HEK", "PRIDICT2 Library-Diverse"),
    ("pridict2-library-diverse__k562", "K562", "PRIDICT2 Library-Diverse"),
    ("pridict2-library-diverse__k562mlh1dn", "MLH1dn", "PRIDICT2 Library-Diverse"),
    ("minsepie-insert-pooled", "HEK PE2", "MinSePIE"),
    ("optiprime-lib-mmr__hek293t__pe2", "HEK PE2", "OptiPrime Lib-MMR"),
    ("optiprime-lib-mmr__hek293t__pe4", "HEK PE4", "OptiPrime Lib-MMR"),
    ("optiprime-lib-mmr__hela__pe2", "HeLa PE2", "OptiPrime Lib-MMR"),
    ("optiprime-lib-mmr__hela__pe4", "HeLa PE4", "OptiPrime Lib-MMR"),
    ("optiprime-lib-cv__hek293t__pe2", "HEK PE2", "OptiPrime Lib-CV"),
    ("optiprime-lib-cv__hek293t__pe4", "HEK PE4", "OptiPrime Lib-CV"),
    ("optiprime-lib-cv__hela__pe2", "HeLa PE2", "OptiPrime Lib-CV"),
    ("optiprime-lib-cv__hela__pe4", "HeLa PE4", "OptiPrime Lib-CV"),
]
HEATMAP_PANELS = [BENCH_META[:7], BENCH_META[7:]]

SINGLE_CONDITION = {
    "pridict1-library1",
    "deepprime-clinvar",
    "deeppe-pooled",
    "minsepie-insert-pooled",
}


def _hex_to_rgb(color: str) -> tuple[float, float, float]:
    raw = color.lstrip("#")
    return tuple(int(raw[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    return "#{:02X}{:02X}{:02X}".format(
        *(max(0, min(255, int(round(c * 255)))) for c in rgb)
    )


def _lerp_hex(start: str, end: str, weight: float) -> str:
    t = min(max(weight, 0.0), 1.0)
    rgb = tuple(a + (b - a) * t for a, b in zip(_hex_to_rgb(start), _hex_to_rgb(end)))
    return _rgb_to_hex(rgb)


def _label_on_color(bg: str) -> str:
    r, g, b = _hex_to_rgb(bg)
    # Perceived luminance
    return "#FFFFFF" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else "#222222"


def _correlation_cmap(vmin: float, vmax: float) -> LinearSegmentedColormap:
    span = vmax - vmin
    if span <= 0:
        cmap = LinearSegmentedColormap.from_list("pehub_scratch", ["#FFFFFF", "#FFFFFF"])
        cmap.set_bad(MISSING_CELL)
        return cmap
    stops = np.linspace(0.0, 1.0, 21)
    colors = [
        (float(stop), _lerp_hex("#FFFFFF", PEARSON_POS_POLE, max(0.0, vmin + stop * span)))
        for stop in stops
    ]
    cmap = LinearSegmentedColormap.from_list("pehub_scratch", colors)
    cmap.set_bad(MISSING_CELL)
    return cmap


def _seed_values(cell_dir: Path, metric: str) -> list[float]:
    state = cell_dir / "state"
    if not state.is_dir():
        return []
    vals: list[float] = []
    for path in sorted(state.glob("seed_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "ok":
            continue
        value = payload.get(metric)
        if value is None:
            continue
        vals.append(float(value))
    return vals


def _cell_dir(model: str, bench: str) -> Path | None:
    """Resolve the preferred results cell for this model × bench."""
    if model == "oped":
        path = RESULTS / OPED_RUN / f"oped__{bench}"
        return path if path.is_dir() else None
    # Prefer split re-run; fall back to pooled single-condition sheets.
    for run in (SPLIT_RUN, POOLED_RUN):
        if bench not in SINGLE_CONDITION and run == POOLED_RUN:
            continue
        path = RESULTS / run / f"{model}__{bench}"
        if (path / "state").is_dir() and any((path / "state").glob("seed_*.json")):
            return path
    # Last resort: pooled run even for split-named benches (should not hit)
    path = RESULTS / POOLED_RUN / f"{model}__{bench}"
    return path if path.is_dir() else None


def load_metric(metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return means, stds, and n_seeds matrices shaped (n_models, n_benches)."""
    n_m, n_b = len(MODELS), len(BENCH_META)
    means = np.full((n_m, n_b), np.nan)
    stds = np.full((n_m, n_b), np.nan)
    counts = np.zeros((n_m, n_b), dtype=int)
    for i, model in enumerate(MODELS):
        for j, (bench, _label, _study) in enumerate(BENCH_META):
            cell = _cell_dir(model, bench)
            if cell is None:
                continue
            vals = _seed_values(cell, metric)
            if not vals:
                continue
            means[i, j] = sum(vals) / len(vals)
            stds[i, j] = statistics.stdev(vals) if len(vals) > 1 else 0.0
            counts[i, j] = len(vals)
    return means, stds, counts


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


def _points_to_data(ax: Any, points: float, *, horizontal: bool) -> float:
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


def _draw_panel(
    ax: Any,
    means: np.ndarray,
    stds: np.ndarray,
    counts: np.ndarray,
    panel: list[tuple[str, str, str]],
    col_offset: int,
    *,
    cmap: Any,
    norm: Any,
    max_cols: int,
    show_ylabel: bool,
) -> Any:
    n_models = len(MODELS)
    n_benches = len(panel)
    values = means[:, col_offset : col_offset + n_benches]
    sds = stds[:, col_offset : col_offset + n_benches]
    ns = counts[:, col_offset : col_offset + n_benches]

    # Map this panel's columns onto [0, max_cols) so both rows share the same
    # outer width (top has 7 studies, bottom 8 — stretch equally).
    scale = max_cols / n_benches
    im = ax.imshow(
        values,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        extent=(-0.5, max_cols - 0.5, n_models - 0.5, -0.5),
        interpolation="nearest",
    )

    centers = [(j + 0.5) * scale - 0.5 for j in range(n_benches)]
    ax.set_xticks(centers)
    ax.set_xticklabels([label for _k, label, _s in panel], rotation=32, ha="right")
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(
        [MODEL_LABELS[m] for m in MODELS] if show_ylabel else []
    )
    ax.tick_params(axis="x", length=0, labelsize=FONT_XTICK, colors="#555555")
    ax.tick_params(axis="y", length=0, labelsize=FONT_YTICK, colors="#555555")
    # Grid at stretched column boundaries
    ax.set_xticks([j * scale - 0.5 for j in range(n_benches + 1)], minor=True)
    ax.set_yticks(np.arange(-0.5, n_models, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(n_models):
        for j in range(n_benches):
            value = values[i, j]
            cx = centers[j]
            if not np.isfinite(value):
                x0 = j * scale - 0.5
                ax.add_patch(
                    Rectangle(
                        (x0, i - 0.5),
                        scale,
                        1,
                        facecolor=MISSING_CELL,
                        edgecolor="none",
                        zorder=1,
                    )
                )
                ax.plot(
                    [cx - 0.3 * scale, cx + 0.3 * scale],
                    [i - 0.35, i + 0.35],
                    color="#6E6E6E",
                    lw=1.15,
                    zorder=4,
                )
                ax.plot(
                    [cx - 0.3 * scale, cx + 0.3 * scale],
                    [i + 0.35, i - 0.35],
                    color="#6E6E6E",
                    lw=1.15,
                    zorder=4,
                )
                continue
            text_color = "white" if value >= 0.55 else "#444444"
            n = int(ns[i, j])
            sd = sds[i, j]
            label = f"{value:.2f}\n±{sd:.2f}"
            if n < 3:
                label += f"\n(n={n})"
            ax.text(
                cx,
                i,
                label,
                ha="center",
                va="center",
                fontsize=FONT_CELL,
                color=text_color,
                zorder=3,
                linespacing=0.95,
            )

    header_h = 0.46
    ax.set_xlim(-0.5, max_cols - 0.5)
    ax.set_ylim(n_models - 0.5, -0.5 - header_h - 0.04)
    body_lw = 2.2
    pad_x = 0.5 * _points_to_data(ax, body_lw, horizontal=True)
    pad_y = 0.5 * _points_to_data(ax, body_lw, horizontal=False)
    for start, end, study in _study_spans(panel):
        color = STUDY_GROUP_COLORS[study]
        x0 = start * scale - 0.5
        width = (end - start + 1) * scale
        ax.add_patch(
            Rectangle(
                (x0 + pad_x, -0.5 + pad_y),
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
                (x0, -0.5 - header_h),
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
            x0 + width / 2.0,
            -0.5 - header_h / 2.0,
            study,
            ha="center",
            va="center",
            fontsize=FONT_DATASET,
            fontweight="600",
            color=_label_on_color(color),
            zorder=7,
            clip_on=False,
        )
    return im


def plot_split_heatmap(
    means: np.ndarray,
    stds: np.ndarray,
    counts: np.ndarray,
    *,
    metric_label: str,
    out_stem: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = _correlation_cmap(vmin, vmax)
    panels = HEATMAP_PANELS
    max_cols = max(len(p) for p in panels)

    fig = plt.figure(figsize=(12.8, 5.8))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 1.0],
        hspace=0.32,
        left=0.10,
        right=0.84,
        top=0.94,
        bottom=0.12,
    )
    axes = [fig.add_subplot(grid[0]), fig.add_subplot(grid[1])]
    images = []
    offsets = [0, len(panels[0])]
    for ax, panel, offset in zip(axes, panels, offsets):
        images.append(
            _draw_panel(
                ax,
                means,
                stds,
                counts,
                panel,
                offset,
                cmap=cmap,
                norm=norm,
                max_cols=max_cols,
                show_ylabel=True,
            )
        )

    fig.canvas.draw()
    top_pos = axes[0].get_position()
    bot_pos = axes[-1].get_position()
    cax = fig.add_axes([top_pos.x1 + 0.01, bot_pos.y0, 0.018, top_pos.y1 - bot_pos.y0])
    cbar = fig.colorbar(images[0], cax=cax)
    cbar.set_label(metric_label, fontsize=FONT_LEGEND_LABEL, color="#444444")
    cbar.set_ticks(np.arange(vmin, vmax + 1e-9, 0.2))
    cbar.ax.tick_params(labelsize=FONT_LEGEND_TICK, colors="#555555")
    cbar.outline.set_visible(False)

    fig.suptitle(
        "From-scratch holdout$_3$ (mean ± s.d. over seeds)",
        fontsize=FONT_TITLE,
        color="#333333",
        y=0.98,
    )

    pdf = OUT_DIR / f"{out_stem}.pdf"
    png = OUT_DIR / f"{out_stem}.png"
    # Do not use bbox_inches='tight' — it crops the top panel's pad column and
    # makes the two rows look different widths.
    fig.savefig(pdf, facecolor="white", pad_inches=0.18)
    fig.savefig(png, dpi=300, facecolor="white", pad_inches=0.18)
    plt.close(fig)
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")


def main(_argv: list[str] | None = None) -> None:
    for required in (OPED_RUN, SPLIT_RUN, POOLED_RUN):
        if not (RESULTS / required).is_dir():
            raise SystemExit(f"Missing results run: {RESULTS / required}")

    pearson, pearson_std, pearson_n = load_metric("test_pearson")
    spearman, spearman_std, spearman_n = load_metric("test_spearman")

    # Report incomplete cells
    for i, model in enumerate(MODELS):
        for j, (bench, _l, _s) in enumerate(BENCH_META):
            n = int(pearson_n[i, j])
            if n < 3:
                print(f"note: {model} @ {bench}: n={n}/3 seeds")

    plot_split_heatmap(
        pearson,
        pearson_std,
        pearson_n,
        metric_label="Pearson $r$",
        out_stem="scratch_pearson_heatmap",
    )
    plot_split_heatmap(
        spearman,
        spearman_std,
        spearman_n,
        metric_label=r"Spearman $\rho$",
        out_stem="scratch_spearman_heatmap",
    )


if __name__ == "__main__":
    main(sys.argv[1:])
