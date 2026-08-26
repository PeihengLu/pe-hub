#!/usr/bin/env python3
"""Build a tiny DATA_ROOT for peen preflight (subsample real standardized sheets).

Writes under ``--work-dir/datasets``:
  - catalog/pe_database.db  (copied from the real datasets tree)
  - standardized/...        (small parquet sheets used by peen filter/merge)
  - formatted/              (empty; peen rebuilds quickly on the mini sheets)

Also seeds intentional ``target_uid`` overlap between library1 and ClinVar so
``--merge --use-original-fold`` is exercised.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = REPO / "datasets"

SHEETS = (
    ("pridict1", "library1", "hek293t", "pe2"),
    ("deepprime", "deepprime_clinvar", "hek293t", "pe2"),
    ("pridict2", "library_diverse", "hek", "pe2"),
    ("pridict2", "library_diverse", "k562", "pe2"),
)


def _std_path(root: Path, study: str, dataset: str, cell: str, pe: str) -> Path:
    return root / "standardized" / study / dataset / f"{cell}-{pe}.parquet"


def _add_target_uid(df: pd.DataFrame) -> pd.DataFrame:
    from pe_common.data_utils import add_target_uid

    return add_target_uid(df)


def _sample_with_overlap(
    library1: pd.DataFrame,
    clinvar: pd.DataFrame,
    *,
    n_each: int,
    n_overlap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clin = clinvar.copy()
    if "original_fold" in clin.columns:
        clin = clin[clin["original_fold"].notna()]
    clin = _add_target_uid(clin)
    l1 = _add_target_uid(library1.copy())

    clin_uids = set(clin["target_uid"].dropna().astype(str))
    l1_overlap = l1[l1["target_uid"].astype(str).isin(clin_uids)]
    n_overlap = min(n_overlap, len(l1_overlap), n_each // 2)
    overlap = (
        l1_overlap.sample(n=n_overlap, random_state=seed)
        if n_overlap
        else l1_overlap.iloc[0:0]
    )
    overlap_uids = set(overlap["target_uid"].astype(str))

    clin_hit = clin[clin["target_uid"].astype(str).isin(overlap_uids)]
    # one clin row per overlapped uid when possible
    clin_overlap = (
        clin_hit.groupby("target_uid", sort=False).head(1)
        if len(clin_hit)
        else clin_hit
    )
    clin_rest_n = max(0, n_each - len(clin_overlap))
    clin_rest_pool = clin[~clin.index.isin(clin_overlap.index)]
    clin_rest = (
        clin_rest_pool.sample(n=min(clin_rest_n, len(clin_rest_pool)), random_state=seed)
        if clin_rest_n
        else clin_rest_pool.iloc[0:0]
    )
    clin_out = pd.concat([clin_overlap, clin_rest], ignore_index=True)

    l1_rest_n = max(0, n_each - len(overlap))
    l1_rest_pool = l1[~l1.index.isin(overlap.index)]
    l1_rest = (
        l1_rest_pool.sample(n=min(l1_rest_n, len(l1_rest_pool)), random_state=seed + 1)
        if l1_rest_n
        else l1_rest_pool.iloc[0:0]
    )
    l1_out = pd.concat([overlap, l1_rest], ignore_index=True)

    # Drop helper column — not part of standardized on-disk schema.
    for frame in (l1_out, clin_out):
        if "target_uid" in frame.columns:
            frame.drop(columns=["target_uid"], inplace=True)
    return l1_out, clin_out


def _sample_simple(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def build(work_dir: Path, source: Path, n_rows: int, seed: int) -> dict[str, int]:
    work_dir = work_dir.resolve()
    source = source.resolve()
    data_root = work_dir / "datasets"
    if data_root.exists():
        shutil.rmtree(data_root)
    data_root.mkdir(parents=True)

    catalog_src = source / "catalog" / "pe_database.db"
    if not catalog_src.is_file():
        raise FileNotFoundError(
            f"Missing catalog DB at {catalog_src}. Run pedb init on the main datasets first."
        )
    (data_root / "catalog").mkdir(parents=True)
    shutil.copy2(catalog_src, data_root / "catalog" / "pe_database.db")
    (data_root / "formatted").mkdir(parents=True)
    (data_root / "exported").mkdir(parents=True)
    (data_root / "raw").mkdir(parents=True)

    counts: dict[str, int] = {}
    l1_path = _std_path(source, "pridict1", "library1", "hek293t", "pe2")
    clin_path = _std_path(source, "deepprime", "deepprime_clinvar", "hek293t", "pe2")
    if not l1_path.is_file() or not clin_path.is_file():
        raise FileNotFoundError(
            f"Need standardized sheets:\n  {l1_path}\n  {clin_path}"
        )

    print(f"Sampling library1 + ClinVar (n≈{n_rows}, overlap)…")
    l1_full = pd.read_parquet(l1_path)
    clin_full = pd.read_parquet(clin_path)
    l1_s, clin_s = _sample_with_overlap(
        l1_full,
        clin_full,
        n_each=n_rows,
        n_overlap=max(8, n_rows // 4),
        seed=seed,
    )
    out_l1 = _std_path(data_root, "pridict1", "library1", "hek293t", "pe2")
    out_clin = _std_path(data_root, "deepprime", "deepprime_clinvar", "hek293t", "pe2")
    out_l1.parent.mkdir(parents=True, exist_ok=True)
    out_clin.parent.mkdir(parents=True, exist_ok=True)
    l1_s.to_parquet(out_l1, index=False)
    clin_s.to_parquet(out_clin, index=False)
    counts["pridict1/library1"] = len(l1_s)
    counts["deepprime/deepprime_clinvar"] = len(clin_s)

    for study, dataset, cell, pe in SHEETS:
        if (study, dataset) in {("pridict1", "library1"), ("deepprime", "deepprime_clinvar")}:
            continue
        src = _std_path(source, study, dataset, cell, pe)
        if not src.is_file():
            print(f"WARNING: skip missing {src}", file=sys.stderr)
            continue
        print(f"Sampling {study}/{dataset}/{cell}-{pe}…")
        sampled = _sample_simple(pd.read_parquet(src), n_rows, seed + hash((study, dataset, cell)) % 10_000)
        dst = _std_path(data_root, study, dataset, cell, pe)
        dst.parent.mkdir(parents=True, exist_ok=True)
        sampled.to_parquet(dst, index=False)
        counts[f"{study}/{dataset}/{cell}"] = len(sampled)

    print(f"Wrote mini DATA_ROOT → {data_root}")
    for key, n in counts.items():
        print(f"  {key}: {n} rows")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="Scratch directory (datasets/ will be created underneath)",
    )
    parser.add_argument(
        "--source-datasets",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Real datasets/ tree to sample from",
    )
    parser.add_argument("--n-rows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(args.work_dir, args.source_datasets, args.n_rows, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
