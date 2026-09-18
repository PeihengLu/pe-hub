#!/usr/bin/env python3
"""Generate from-scratch base-weight heatmap for the term paper (Results).

Supports the corrected per-condition matrix (Library-Diverse × 3 cell lines;
Lib-MMR / Lib-CV × HEK/HeLa × PE2/PE4). If a run only has the older pooled
bench names, those columns are plotted instead.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent

# Match scripts/experiments/plot_base_model_eval.py vendor Pearson palette:
# white at 0, navy at +1 (same positive pole as the base-model heatmaps).
PEARSON_POS_POLE = "#1F4E79"
MISSING_CELL = "#F4F4F4"


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


def _vendor_correlation_cmap(vmin: float, vmax: float) -> LinearSegmentedColormap:
    """Same white→navy ramp as plot_base_model_eval._pearson_cmap for r≥0."""
    span = vmax - vmin
    if span <= 0:
        cmap = LinearSegmentedColormap.from_list("pehub_scratch", ["#FFFFFF", "#FFFFFF"])
        cmap.set_bad(MISSING_CELL)
        return cmap
    stops = np.linspace(0.0, 1.0, 21)
    # Pole strength tracks absolute correlation (vendor: white at 0, #1F4E79 at 1).
    colors = [
        (float(stop), _lerp_hex("#FFFFFF", PEARSON_POS_POLE, vmin + stop * span))
        for stop in stops
    ]
    cmap = LinearSegmentedColormap.from_list("pehub_scratch", colors)
    cmap.set_bad(MISSING_CELL)
    return cmap

MODELS = ["deepprime", "oped", "pridict2"]
MODEL_LABELS = {
    "deepprime": "DeepPrime",
    "oped": "OPED",
    "pridict2": "PRIDICT2",
}

# Preferred (unpooled) layout — matches scratch-benchmark/_common.sh MATRIX_ALL.
BENCHES_SPLIT = [
    "pridict1-library1",
    "pridict2-library-diverse__hek293t",
    "pridict2-library-diverse__k562",
    "pridict2-library-diverse__k562mlh1dn",
    "deepprime-clinvar",
    "deeppe-pooled",
    "minsepie-insert-pooled",
    "optiprime-lib-mmr__hek293t__pe2",
    "optiprime-lib-mmr__hek293t__pe4",
    "optiprime-lib-mmr__hela__pe2",
    "optiprime-lib-mmr__hela__pe4",
    "optiprime-lib-cv__hek293t__pe2",
    "optiprime-lib-cv__hek293t__pe4",
    "optiprime-lib-cv__hela__pe2",
    "optiprime-lib-cv__hela__pe4",
]
BENCH_LABELS_SPLIT = {
    "pridict1-library1": "Library 1",
    "pridict2-library-diverse__hek293t": "Lib-Diverse HEK",
    "pridict2-library-diverse__k562": "Lib-Diverse K562",
    "pridict2-library-diverse__k562mlh1dn": "Lib-Diverse MLH1dn",
    "deepprime-clinvar": "ClinVar",
    "deeppe-pooled": "DeepPE HEK",
    "minsepie-insert-pooled": "MinSePIE HEK PE2",
    "optiprime-lib-mmr__hek293t__pe2": "Lib-MMR HEK PE2",
    "optiprime-lib-mmr__hek293t__pe4": "Lib-MMR HEK PE4",
    "optiprime-lib-mmr__hela__pe2": "Lib-MMR HeLa PE2",
    "optiprime-lib-mmr__hela__pe4": "Lib-MMR HeLa PE4",
    "optiprime-lib-cv__hek293t__pe2": "Lib-CV HEK PE2",
    "optiprime-lib-cv__hek293t__pe4": "Lib-CV HEK PE4",
    "optiprime-lib-cv__hela__pe2": "Lib-CV HeLa PE2",
    "optiprime-lib-cv__hela__pe4": "Lib-CV HeLa PE4",
}

# Legacy pooled layout (20260911T065340Z).
BENCHES_POOLED = [
    "pridict1-library1",
    "pridict2-library-diverse",
    "deepprime-clinvar",
    "deeppe-pooled",
    "minsepie-insert-pooled",
    "optiprime-lib-mmr",
    "optiprime-lib-cv",
]
BENCH_LABELS_POOLED = {
    "pridict1-library1": "Library 1",
    "pridict2-library-diverse": "Library-Diverse (pooled)",
    "deepprime-clinvar": "ClinVar",
    "deeppe-pooled": "DeepPE HEK",
    "minsepie-insert-pooled": "MinSePIE inserts",
    "optiprime-lib-mmr": "Lib-MMR (pooled)",
    "optiprime-lib-cv": "Lib-CV (pooled)",
}


def resolve_run(run_id: str | None) -> Path:
    results = REPO / "scripts/experiments/scratch-benchmark/results"
    if run_id:
        path = results / run_id
    else:
        latest = results / "LATEST_RUN_ID"
        if latest.is_file():
            path = results / latest.read_text(encoding="utf-8").strip()
        else:
            path = results / "20260911T065340Z"
    if not path.is_dir():
        raise SystemExit(f"Run directory not found: {path}")
    return path


def detect_layout(run: Path) -> tuple[list[str], dict[str, str]]:
    """Prefer split layout when any unpooled Library-Diverse / Hsu cell exists."""
    for model in MODELS:
        for bench in (
            "pridict2-library-diverse__hek293t",
            "optiprime-lib-mmr__hek293t__pe2",
            "optiprime-lib-cv__hek293t__pe2",
        ):
            if (run / f"{model}__{bench}" / "state").is_dir():
                return BENCHES_SPLIT, BENCH_LABELS_SPLIT
    return BENCHES_POOLED, BENCH_LABELS_POOLED


def load_metric(
    run: Path,
    benches: list[str],
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    means = np.full((len(MODELS), len(benches)), np.nan)
    stds = np.full_like(means, np.nan)
    for i, model in enumerate(MODELS):
        for j, bench in enumerate(benches):
            state = run / f"{model}__{bench}" / "state"
            vals: list[float] = []
            for path in sorted(state.glob("seed_*.json")):
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("status") != "ok":
                    continue
                value = payload.get(metric)
                if value is None:
                    continue
                vals.append(float(value))
            if not vals:
                continue
            means[i, j] = sum(vals) / len(vals)
            stds[i, j] = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return means, stds


def plot_heatmap(
    means: np.ndarray,
    stds: np.ndarray,
    *,
    benches: list[str],
    labels: dict[str, str],
    title: str,
    out_stem: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    sns.set_theme(style="white", context="notebook", font_scale=1.0)
    width = max(10.5, 0.85 * len(benches) + 3.5)
    fig, ax = plt.subplots(figsize=(width, 3.8))
    annot = np.empty(means.shape, dtype=object)
    for i in range(means.shape[0]):
        for j in range(means.shape[1]):
            if np.isnan(means[i, j]):
                annot[i, j] = ""
            else:
                annot[i, j] = f"{means[i, j]:.2f}\n±{stds[i, j]:.2f}"

    sns.heatmap(
        means,
        ax=ax,
        annot=annot,
        fmt="",
        cmap=_vendor_correlation_cmap(vmin, vmax),
        vmin=vmin,
        vmax=vmax,
        linewidths=0.6,
        linecolor="#f0f0f0",
        cbar_kws={"label": title, "shrink": 0.85},
        xticklabels=[labels[b] for b in benches],
        yticklabels=[MODEL_LABELS[m] for m in MODELS],
        square=False,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=35, labelsize=9)
    ax.tick_params(axis="y", labelrotation=0, labelsize=11)
    ax.set_title(
        "From-scratch holdout$_3$ (mean ± s.d. over 3 seeds)",
        fontsize=12,
        pad=10,
    )
    fig.tight_layout()
    pdf = OUT_DIR / f"{out_stem}.pdf"
    png = OUT_DIR / f"{out_stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")


def main(argv: list[str] | None = None) -> None:
    run_id = argv[0] if argv else None
    run = resolve_run(run_id)
    benches, labels = detect_layout(run)
    print(f"Run: {run.name} ({'split' if benches is BENCHES_SPLIT else 'pooled'} layout)")

    pearson, pearson_std = load_metric(run, benches, "test_pearson")
    spearman, spearman_std = load_metric(run, benches, "test_spearman")
    plot_heatmap(
        pearson,
        pearson_std,
        benches=benches,
        labels=labels,
        title="Pearson $r$",
        out_stem="scratch_pearson_heatmap",
    )
    plot_heatmap(
        spearman,
        spearman_std,
        benches=benches,
        labels=labels,
        title=r"Spearman $\rho$",
        out_stem="scratch_spearman_heatmap",
    )


if __name__ == "__main__":
    main(sys.argv[1:])
