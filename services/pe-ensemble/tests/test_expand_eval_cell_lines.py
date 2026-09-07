"""Tests for per-cell-line eval benchmark expansion."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "packages" / "pe-common"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments"))

from expand_eval_cell_lines import expand_benchmark_spec  # noqa: E402


def test_expands_library_diverse_style_cells(tmp_path: Path):
    study = tmp_path / "standardized" / "pridict2" / "library_diverse"
    study.mkdir(parents=True)
    for stem in ("hek-pe2", "k562-pe2", "k562mlh1dn-pe2"):
        (study / f"{stem}.parquet").write_bytes(b"")
    out = expand_benchmark_spec(
        "pridict2-library-diverse|pridict2|library-diverse",
        tmp_path,
    )
    assert out == [
        "pridict2-library-diverse__hek293t|pridict2|library-diverse|hek293t|pe2",
        "pridict2-library-diverse__k562|pridict2|library-diverse|k562|pe2",
        "pridict2-library-diverse__k562mlh1dn|pridict2|library-diverse|k562mlh1dn|pe2",
    ]


def test_hek_and_hek293t_collapse_to_one_cell(tmp_path: Path):
    folder = tmp_path / "standardized" / "pridict2" / "library_diverse"
    folder.mkdir(parents=True)
    (folder / "hek-pe2.parquet").write_bytes(b"")
    (folder / "hek293t-pe2.parquet").write_bytes(b"")
    (folder / "k562-pe2.parquet").write_bytes(b"")
    out = expand_benchmark_spec(
        "pridict2-library-diverse|pridict2|library-diverse",
        tmp_path,
    )
    assert out == [
        "pridict2-library-diverse__hek293t|pridict2|library-diverse|hek293t|pe2",
        "pridict2-library-diverse__k562|pridict2|library-diverse|k562|pe2",
    ]


def test_single_cell_keeps_benchmark_name(tmp_path: Path):
    folder = tmp_path / "standardized" / "pridict1" / "library1"
    folder.mkdir(parents=True)
    (folder / "hek293t-pe2.parquet").write_bytes(b"")
    out = expand_benchmark_spec("pridict1-library1|pridict1|library1", tmp_path)
    assert out == ["pridict1-library1|pridict1|library1|hek293t|pe2"]


def test_pooled_datasets_union_cell_lines(tmp_path: Path):
    deeppe = tmp_path / "standardized" / "deeppe"
    (deeppe / "deeppe_ht").mkdir(parents=True)
    (deeppe / "deeppe_endo").mkdir(parents=True)
    (deeppe / "deeppe_ht" / "hek293t-pe2.parquet").write_bytes(b"")
    (deeppe / "deeppe_endo" / "hek293t-pe2.parquet").write_bytes(b"")
    (deeppe / "deeppe_endo" / "hct116-pe2.parquet").write_bytes(b"")
    out = expand_benchmark_spec(
        "deeppe-pooled|deeppe|deeppe-ht,deeppe-endo",
        tmp_path,
    )
    assert out == [
        "deeppe-pooled__hct116|deeppe|deeppe-ht,deeppe-endo|hct116|pe2",
        "deeppe-pooled__hek293t|deeppe|deeppe-ht,deeppe-endo|hek293t|pe2",
    ]


def test_expands_pe2_and_pe4_without_pooling(tmp_path: Path):
    folder = tmp_path / "standardized" / "optiprime" / "lib_mmr"
    folder.mkdir(parents=True)
    for stem in ("hek293t-pe2", "hek293t-pe4", "hela-pe2", "hela-pe4"):
        (folder / f"{stem}.parquet").write_bytes(b"")
    out = expand_benchmark_spec("optiprime-lib-mmr|optiprime|lib-mmr", tmp_path)
    assert out == [
        "optiprime-lib-mmr__hek293t__pe2|optiprime|lib-mmr|hek293t|pe2",
        "optiprime-lib-mmr__hek293t__pe4|optiprime|lib-mmr|hek293t|pe4",
        "optiprime-lib-mmr__hela__pe2|optiprime|lib-mmr|hela|pe2",
        "optiprime-lib-mmr__hela__pe4|optiprime|lib-mmr|hela|pe4",
    ]
