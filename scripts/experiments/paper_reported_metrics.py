"""Author-reported Pearson/Spearman for base-model eval comparison and leak fills.

``r`` is Pearson, ``R``/``ρ`` is Spearman (Mathis / Hsu notation).
Only ``fill_on_leak`` rows are copied into leaked eval cells for plotting.
"""
from __future__ import annotations

from typing import Any, Optional

# Keys used to join an eval row to a paper number.
# benchmark: prefix match on benchmark_name (e.g. "pridict2-library-diverse__hek293t")
# cell_line / pe_system / pridict2_head: exact match when set.

PAPER_METRICS: list[dict[str, Any]] = [
    {
        "id": "deepprime-clinvar-test",
        "model": "deepprime",
        "benchmark": "deepprime-clinvar",
        "cell_line": "hek293t",
        "pearson": 0.84,
        "spearman": 0.86,
        "citation": "Yu et al. Cell 2023 Fig. 3D",
        "protocol": "ClinVar_Test author holdout (original_fold=-1)",
        "protocol_match": "close",
        "fill_on_leak": False,
    },
    {
        "id": "pridict2-diverse-hek",
        "model": "pridict2",
        "benchmark": "pridict2-library-diverse__hek293t",
        "cell_line": "hek293t",
        "pridict2_head": "HEK",
        "pearson": 0.90,
        "spearman": 0.91,
        "citation": "Mathis et al. Nat. Biotechnol. 2024 Fig. 1o",
        "protocol": "PRIDICT2.0 ensemble, HEK head, grouped 5-fold CV on Library-Diverse",
        "protocol_match": "close",
        "fill_on_leak": False,
    },
    {
        "id": "pridict2-diverse-k562",
        "model": "pridict2",
        "benchmark": "pridict2-library-diverse__k562",
        "cell_line": "k562",
        "pridict2_head": "K562",
        "pearson": 0.70,
        "spearman": 0.81,
        "citation": "Mathis et al. Nat. Biotechnol. 2024 Fig. 1p",
        "protocol": "PRIDICT2.0 ensemble, K562 head, grouped 5-fold CV on Library-Diverse",
        "protocol_match": "close",
        "fill_on_leak": False,
    },
    {
        "id": "pridict2-diverse-k562mlh1dn-hek",
        "model": "pridict2",
        "benchmark": "pridict2-library-diverse__k562mlh1dn",
        "cell_line": "k562mlh1dn",
        "pridict2_head": "HEK",
        "pearson": None,
        "spearman": 0.88,
        "citation": "Mathis et al. Nat. Biotechnol. 2024 (text; HEK vs K562 head on MLH1dn)",
        "protocol": "PRIDICT2.0 HEK head on K562-MLH1dn Library-Diverse (paper reports Spearman only)",
        "protocol_match": "close",
        "fill_on_leak": False,
    },
    {
        "id": "pridict1-library1-cv",
        "model": "pridict1",
        "benchmark": "pridict1-library1",
        "cell_line": "hek293t",
        "pearson": 0.86,
        "spearman": 0.85,
        "citation": "Mathis et al. Nat. Biotechnol. 2023 Fig. 2b–e",
        "protocol": "PRIDICT (not PRIDICT2) grouped 5-fold CV on Library 1",
        "protocol_match": "different_model",
        "fill_on_leak": False,
        "notes": "Do not fill PRIDICT2 library1 leak cells with this; different model, and PRIDICT2 trained on all of library1.",
    },
    {
        "id": "oped-deeppe-ht-test",
        "model": "oped",
        "benchmark": "deeppe-pooled__hek293t",
        "cell_line": "hek293t",
        "pearson": 0.769,
        "spearman": 0.798,
        "citation": "Liu et al. Nat. Mach. Intell. 2023 Fig. 2a",
        "protocol": (
            "Liu Fig. 2a is DeepPE HT-test only (n=4457). PE-hub DeepPE HEK "
            "uses the author original_fold=-1 rows from HT+type+position "
            "(~5060), not a random holdout and not HT-test alone"
        ),
        "protocol_match": "loose",
        "fill_on_leak": False,
    },
    {
        "id": "optiprime-lib-mmr-hek",
        "model": "optiprime",
        "benchmark": "optiprime-lib-mmr__hek293t",
        "cell_line": "hek293t",
        "pe_system": "pe2",
        "pearson": 0.723,
        "spearman": 0.775,
        "citation": "Hsu et al. Nat. Biotechnol. 2026 Fig. 4b,c",
        "protocol": "Mean over four held-out PE2 conditions (Lib-MMR/Lib-CV × HEK293T/HeLa); PE4 is not filled",
        "protocol_match": "approximate",
        "fill_on_leak": True,
    },
    {
        "id": "optiprime-lib-mmr-hela",
        "model": "optiprime",
        "benchmark": "optiprime-lib-mmr__hela",
        "cell_line": "hela",
        "pe_system": "pe2",
        "pearson": 0.723,
        "spearman": 0.775,
        "citation": "Hsu et al. Nat. Biotechnol. 2026 Fig. 4b,c",
        "protocol": "Mean over four held-out PE2 conditions (Lib-MMR/Lib-CV × HEK293T/HeLa); PE4 is not filled",
        "protocol_match": "approximate",
        "fill_on_leak": True,
    },
    {
        "id": "optiprime-lib-cv-hek",
        "model": "optiprime",
        "benchmark": "optiprime-lib-cv__hek293t",
        "cell_line": "hek293t",
        "pe_system": "pe2",
        "pearson": 0.723,
        "spearman": 0.775,
        "citation": "Hsu et al. Nat. Biotechnol. 2026 Fig. 4b,c",
        "protocol": "Mean over four held-out PE2 conditions (Lib-MMR/Lib-CV × HEK293T/HeLa); PE4 is not filled",
        "protocol_match": "approximate",
        "fill_on_leak": True,
    },
    {
        "id": "optiprime-lib-cv-hela",
        "model": "optiprime",
        "benchmark": "optiprime-lib-cv__hela",
        "cell_line": "hela",
        "pe_system": "pe2",
        "pearson": 0.723,
        "spearman": 0.775,
        "citation": "Hsu et al. Nat. Biotechnol. 2026 Fig. 4b,c",
        "protocol": "Mean over four held-out PE2 conditions (Lib-MMR/Lib-CV × HEK293T/HeLa); PE4 is not filled",
        "protocol_match": "approximate",
        "fill_on_leak": True,
    },
    {
        "id": "deepprime-diverse-hek-le3bp",
        "model": "deepprime",
        "benchmark": "pridict2-library-diverse__hek293t",
        "cell_line": "hek293t",
        "pearson": None,
        "spearman": 0.74,
        "citation": "Mathis et al. Nat. Biotechnol. 2024 Fig. 2g",
        "protocol": "DeepPrime on Library-Diverse filtered to ≤3 bp edits (47% of library)",
        "protocol_match": "loose",
        "fill_on_leak": False,
    },
    {
        "id": "deepprime-diverse-k562-le3bp",
        "model": "deepprime",
        "benchmark": "pridict2-library-diverse__k562",
        "cell_line": "k562",
        "pearson": None,
        "spearman": 0.65,
        "citation": "Mathis et al. Nat. Biotechnol. 2024 Fig. 2g",
        "protocol": "DeepPrime on Library-Diverse filtered to ≤3 bp edits",
        "protocol_match": "loose",
        "fill_on_leak": False,
    },
    {
        "id": "deepprime-diverse-mlh1dn-le3bp",
        "model": "deepprime",
        "benchmark": "pridict2-library-diverse__k562mlh1dn",
        "cell_line": "k562mlh1dn",
        "pearson": None,
        "spearman": 0.72,
        "citation": "Mathis et al. Nat. Biotechnol. 2024 Fig. 2g",
        "protocol": "DeepPrime on Library-Diverse filtered to ≤3 bp edits",
        "protocol_match": "loose",
        "fill_on_leak": False,
    },
]


VALUE_SOURCE_MEASURED = "measured"
VALUE_SOURCE_AUTHOR_FILL = "author_fill"
VALUE_SOURCE_LEAK_UNFILLED = "leak_unfilled"

PLOT_MARKER_MEASURED = "measured"
PLOT_MARKER_AUTHOR_FILL = "author_fill"
PLOT_HATCH_AUTHOR_FILL = "///"


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def match_paper_metric(row: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Return the best paper metric for an eval/summary row, or None."""
    model = _norm(row.get("model"))
    bench = str(row.get("benchmark_name") or "")
    cell = _norm(row.get("cell_line"))
    head = row.get("pridict2_head")
    head_key = str(head).strip() if head not in (None, "") else None

    matches: list[dict[str, Any]] = []
    for metric in PAPER_METRICS:
        if _norm(metric["model"]) != model:
            continue
        prefix = str(metric["benchmark"])
        if bench != prefix and not bench.startswith(prefix):
            continue
        want_cell = metric.get("cell_line")
        if want_cell and _norm(want_cell) != cell:
            continue
        want_pe = metric.get("pe_system")
        if want_pe and _norm(want_pe) != _norm(row.get("pe_system")):
            continue
        want_head = metric.get("pridict2_head")
        if want_head:
            if head_key != str(want_head):
                continue
        matches.append(metric)

    if not matches:
        return None
    # Prefer an exact benchmark_name match over a prefix match.
    exact = [m for m in matches if bench == str(m["benchmark"])]
    chosen = exact[0] if exact else matches[0]
    return chosen


def is_leak_row(row: dict[str, Any]) -> bool:
    if str(row.get("error_type") or "") == "data_leak":
        return True
    if str(row.get("leak_reason") or ""):
        return True
    return False


def annotate_row_with_paper(row: dict[str, Any]) -> dict[str, Any]:
    """Add paper columns and plot values. Does not drop measured metrics."""
    out = dict(row)
    leak = is_leak_row(out)
    if "pearson_measured" in out:
        measured_pearson = out.get("pearson_measured")
        measured_spearman = out.get("spearman_measured")
    else:
        measured_pearson = None if leak else out.get("pearson")
        measured_spearman = None if leak else out.get("spearman")
    out["pearson_measured"] = measured_pearson
    out["spearman_measured"] = measured_spearman

    paper = match_paper_metric(out)
    leak = is_leak_row(out)
    fill = bool(paper and paper.get("fill_on_leak") and leak)

    if paper:
        out["paper_id"] = paper["id"]
        out["paper_pearson"] = paper.get("pearson")
        out["paper_spearman"] = paper.get("spearman")
        out["paper_citation"] = paper.get("citation")
        out["paper_protocol"] = paper.get("protocol")
        out["paper_protocol_match"] = paper.get("protocol_match")
    else:
        out["paper_id"] = None
        out["paper_pearson"] = None
        out["paper_spearman"] = None
        out["paper_citation"] = None
        out["paper_protocol"] = None
        out["paper_protocol_match"] = None

    if fill:
        out["value_source"] = VALUE_SOURCE_AUTHOR_FILL
        out["plot_marker"] = PLOT_MARKER_AUTHOR_FILL
        out["plot_hatch"] = PLOT_HATCH_AUTHOR_FILL
        out["pearson_plot"] = paper.get("pearson")
        out["spearman_plot"] = paper.get("spearman")
        # Fill the plotting columns used by existing figure scripts.
        out["pearson"] = paper.get("pearson")
        out["spearman"] = paper.get("spearman")
    elif leak:
        out["value_source"] = VALUE_SOURCE_LEAK_UNFILLED
        out["plot_marker"] = None
        out["plot_hatch"] = None
        out["pearson_plot"] = None
        out["spearman_plot"] = None
    else:
        out["value_source"] = VALUE_SOURCE_MEASURED
        out["plot_marker"] = PLOT_MARKER_MEASURED
        out["plot_hatch"] = ""
        out["pearson_plot"] = measured_pearson
        out["spearman_plot"] = measured_spearman
    return out
