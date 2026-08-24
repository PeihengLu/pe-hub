"""Data splitting utilities shared across PE models."""

from __future__ import annotations

import hashlib
from typing import Any, Optional

import numpy as np
import pandas as pd

TARGET_UID_COLUMN = "target_uid"


def _stable_group_sort_key(value: Any) -> str:
    """Build a stable sort key across mixed Python scalar types."""
    return f"{type(value).__name__}:{value}"


def _hash_token(prefix: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def compute_target_uid(
    protospacer: Optional[str],
    wt_sequence: Optional[str] = None,
) -> str:
    """Return a deterministic, dataset-independent identifier for a target locus.

    The identifier is derived from the protospacer sequence (``ps:<sha1>``) so
    that the same genomic target site resolves to the same ID in every dataset.
    When no protospacer can be extracted, the padded WT sequence is used as a
    fallback (``wt:<sha1>``). Returns an empty string when neither is available.
    """
    protospacer = ("" if protospacer is None else str(protospacer)).strip().upper()
    if protospacer and protospacer != "NAN":
        return _hash_token("ps", protospacer)
    wt = ("" if wt_sequence is None else str(wt_sequence)).strip().upper()
    if wt and wt != "NAN":
        return _hash_token("wt", wt)
    return ""


def target_uid_series(
    df: pd.DataFrame,
    *,
    wt_col: str = "wt_sequence",
    protospacer_l_col: str = "protospacer_location_l",
    protospacer_r_col: str = "protospacer_location_r",
) -> pd.Series:
    """Compute a universal ``target_uid`` for every row of a standardized frame.

    Rows sharing a protospacer (i.e. targeting the same locus) receive an
    identical ID regardless of which dataset they originate from. This is the
    cross-dataset key used to record training provenance and to detect data
    leakage between a model's training set and an evaluation benchmark.
    """
    protospacers = extract_protospacer_series(
        df,
        wt_col=wt_col,
        protospacer_l_col=protospacer_l_col,
        protospacer_r_col=protospacer_r_col,
    )
    if wt_col in df.columns:
        wt = df[wt_col].astype("string")
    else:
        wt = pd.Series([pd.NA] * len(df), index=df.index, dtype="string")

    uids = [
        compute_target_uid(
            None if ps is pd.NA else ps,
            None if wt_value is pd.NA else wt_value,
        )
        for ps, wt_value in zip(protospacers, wt)
    ]
    return pd.Series(uids, index=df.index, dtype="string")


def add_target_uid(
    df: pd.DataFrame,
    *,
    column: str = TARGET_UID_COLUMN,
    wt_col: str = "wt_sequence",
    protospacer_l_col: str = "protospacer_location_l",
    protospacer_r_col: str = "protospacer_location_r",
) -> pd.DataFrame:
    """Return a copy of ``df`` with a universal ``target_uid`` column added."""
    output = df.copy()
    output[column] = target_uid_series(
        output,
        wt_col=wt_col,
        protospacer_l_col=protospacer_l_col,
        protospacer_r_col=protospacer_r_col,
    )
    return output


def build_test_mask_from_group_id(
    group_series: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.Series:
    """
    Deterministically assign test groups from a group-id series.

    The selection depends only on:
      1) the set of non-null group IDs,
      2) test_size, and
      3) random_state.

    This means the same group-id set yields the same test groups even if row
    order differs between dataframes.
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")

    groups = pd.Series(group_series, copy=False)
    unique_groups = groups.dropna().unique().tolist()
    if not unique_groups:
        return pd.Series(False, index=groups.index, dtype=bool)

    # Sort before shuffling so sampling is stable for the same set of groups.
    ordered_groups = sorted(unique_groups, key=_stable_group_sort_key)
    rng = np.random.default_rng(random_state)
    rng.shuffle(ordered_groups)

    n_test_groups = max(1, int(np.ceil(len(ordered_groups) * test_size)))
    test_groups = set(ordered_groups[:n_test_groups])
    return groups.isin(test_groups)


def extract_protospacer_series(
    df: pd.DataFrame,
    *,
    wt_col: str = "wt_sequence",
    protospacer_l_col: str = "protospacer_location_l",
    protospacer_r_col: str = "protospacer_location_r",
) -> pd.Series:
    """Extract protospacer sequences from standardized rows."""
    if wt_col not in df.columns:
        raise ValueError(f"Missing {wt_col!r} for protospacer extraction")
    wt = df[wt_col].astype(str)
    left = pd.to_numeric(
        df.get(protospacer_l_col, pd.Series(0, index=df.index)),
        errors="coerce",
    ).fillna(0).astype(int)
    right = pd.to_numeric(
        df.get(protospacer_r_col, pd.Series(0, index=df.index)),
        errors="coerce",
    ).fillna(0).astype(int)

    def _slice_protospacer(seq: str, l: int, r: int) -> str:
        if not isinstance(seq, str) or l < 0 or r <= l or r > len(seq):
            return ""
        return seq[l:r].upper()

    return pd.Series(
        (_slice_protospacer(seq, int(l), int(r)) for seq, l, r in zip(wt, left, right)),
        index=df.index,
        dtype="string",
    )


def target_location_group_series(
    df: pd.DataFrame,
    *,
    wt_col: str = "wt_sequence",
    protospacer_l_col: str = "protospacer_location_l",
    protospacer_r_col: str = "protospacer_location_r",
) -> pd.Series:
    """Assign integer group ids so rows sharing a protospacer share a group."""
    protospacers = extract_protospacer_series(
        df,
        wt_col=wt_col,
        protospacer_l_col=protospacer_l_col,
        protospacer_r_col=protospacer_r_col,
    )
    codes, _ = pd.factorize(protospacers, sort=False)
    return pd.Series(codes, index=df.index, dtype=int)


def reassign_group_ids_by_target_location(
    df: pd.DataFrame,
    *,
    group_col: str = "group_id",
    wt_col: str = "wt_sequence",
    protospacer_l_col: str = "protospacer_location_l",
    protospacer_r_col: str = "protospacer_location_r",
) -> pd.DataFrame:
    """Reassign ``group_col`` from protospacer so merged datasheets share groups by target."""
    output = df.copy()
    output[group_col] = target_location_group_series(
        output,
        wt_col=wt_col,
        protospacer_l_col=protospacer_l_col,
        protospacer_r_col=protospacer_r_col,
    )
    return output


def propagate_original_fold_by_target_uid(
    df: pd.DataFrame,
    *,
    fold_col: str = "original_fold",
    wt_col: str = "wt_sequence",
    protospacer_l_col: str = "protospacer_location_l",
    protospacer_r_col: str = "protospacer_location_r",
) -> pd.DataFrame:
    """Fill missing ``original_fold`` from rows that share the same ``target_uid``.

    Used when merging DeepPrime (author folds) with datasheets that lack folds
    (e.g. PRIDICT library1). Overlapping loci inherit DeepPrime's fold so the
    merged table can use ``use_original_fold=True`` consistently. Rows whose
    target never appears with a known fold stay NaN and fall back to
    target-location random splits in ``assign_splits``.
    """
    if fold_col not in df.columns or df.empty:
        return df.copy()

    output = df.copy()
    folds = pd.to_numeric(output[fold_col], errors="coerce")
    uids = target_uid_series(
        output,
        wt_col=wt_col,
        protospacer_l_col=protospacer_l_col,
        protospacer_r_col=protospacer_r_col,
    )
    known = folds.notna() & uids.astype(str).str.len().gt(0)
    if not known.any():
        output[fold_col] = folds
        return output

    # One fold per target_uid: mode among known values (stable via sorted unique).
    fold_by_uid: dict[str, float] = {}
    known_df = pd.DataFrame({"uid": uids[known].astype(str), "fold": folds[known]})
    for uid, group in known_df.groupby("uid", sort=True):
        values = sorted({float(v) for v in group["fold"].tolist()})
        if len(values) == 1:
            fold_by_uid[str(uid)] = values[0]
        else:
            # Conflicting author folds for one locus — prefer DeepPrime test (-1)
            # when present, else the lowest fold id for determinism.
            fold_by_uid[str(uid)] = -1.0 if -1.0 in values else values[0]

    missing = folds.isna() & uids.astype(str).str.len().gt(0)
    if missing.any():
        inherited = uids[missing].astype(str).map(fold_by_uid)
        folds.loc[missing] = inherited.to_numpy()

    output[fold_col] = folds
    return output
