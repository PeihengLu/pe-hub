"""PRIDICT2 vendor provenance: library1 has no author test split."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pe_common.data_utils import target_uid_series

from app.models.pridict2_vendor_provenance import (
    parse_pridict2_vendor_run,
    sheet_target_uids,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
LIBRARY1 = (
    REPO_ROOT / "datasets" / "standardized" / "pridict1" / "library1" / "hek293t-pe2.parquet"
)
DIVERSE = (
    REPO_ROOT
    / "datasets"
    / "standardized"
    / "pridict2"
    / "library_diverse"
    / "hek-pe2.parquet"
)


def test_parse_model_a_and_b_run_ids():
    assert parse_pridict2_vendor_run(
        "pridict1_1__exp_2023-08-25_20-55-53__run_3"
    ) == ("A", 3)
    assert parse_pridict2_vendor_run(
        "pridict1_2__exp_2023-08-28_22-22-26__run_0__HEK"
    ) == ("B", 0)


def test_parse_rejects_non_vendor_ids():
    with pytest.raises(ValueError):
        parse_pridict2_vendor_run("pridict2__custom__20260901__70ecaf")


def test_library1_has_no_author_fold_and_all_rows_are_train():
    if not LIBRARY1.is_file():
        pytest.skip("library1 parquet not in checkout")
    frame = pd.read_parquet(LIBRARY1, columns=["original_fold"])
    assert frame["original_fold"].isna().all()
    loci = sheet_target_uids(LIBRARY1)
    assert len(loci) == len(sheet_target_uids(LIBRARY1, drop_author_test=True))
    assert len(loci) > 1000


def test_library_diverse_exclude_fold_drops_only_that_fold():
    if not DIVERSE.is_file():
        pytest.skip("library-diverse parquet not in checkout")
    all_loci = sheet_target_uids(DIVERSE)
    held = sheet_target_uids(DIVERSE, exclude_original_fold=2.0)
    full = pd.read_parquet(DIVERSE)
    uids = target_uid_series(full)
    fold2_uids = {
        str(uid)
        for uid in uids.loc[full["original_fold"] == 2.0].tolist()
        if uid and str(uid) not in {"nan", "<NA>"}
    }
    assert len(held) < len(all_loci)
    assert fold2_uids
    assert fold2_uids.isdisjoint(held)
