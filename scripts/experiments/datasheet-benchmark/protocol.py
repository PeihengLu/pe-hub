"""Protocol helpers for the datasheet benchmark (no pedb/peen I/O).

Size rule (override with ``--protocol``):
  * ``n_rows < size_threshold`` → N-fold CV, independent Optuna per outer fold
  * ``n_rows >= size_threshold`` → holdout_3 repeated N times (split + init seeds)
"""
from __future__ import annotations

from typing import Any, Literal, Optional

import pandas as pd

from pe_common.splits import (
    SPLIT_COLUMN,
    SplitConfig,
    assign_splits,
    list_assigned_folds,
)

ProtocolName = Literal["cv", "holdout_3"]
ProtocolChoice = Literal["auto", "cv", "holdout_3"]

DEFAULT_SIZE_THRESHOLD = 50_000
DEFAULT_N = 5
DEFAULT_N_TRIALS = 20
DEFAULT_BASE_SEED = 42
DEFAULT_HOLDOUT_TRAIN_PCT = 0.7
DEFAULT_HOLDOUT_VAL_PCT = 0.15
DEFAULT_HOLDOUT_TEST_PCT = 0.15
DEFAULT_INNER_TRAIN_PCT = 0.8
DEFAULT_INNER_VAL_PCT = 0.2
GROUP_COL_CANDIDATES = ("target_uid", "group_id")


class ProtocolError(ValueError):
    """Invalid split remapping or protocol selection."""


def choose_protocol(
    n_rows: int,
    *,
    size_threshold: int = DEFAULT_SIZE_THRESHOLD,
    override: ProtocolChoice = "auto",
) -> tuple[ProtocolName, str]:
    """Return ``(protocol, reason)`` from row count or an explicit override."""
    if n_rows < 0:
        raise ProtocolError(f"n_rows must be >= 0, got {n_rows}")
    if size_threshold < 1:
        raise ProtocolError(f"size_threshold must be >= 1, got {size_threshold}")
    if override == "cv":
        return "cv", "user override (--protocol cv)"
    if override == "holdout_3":
        return "holdout_3", "user override (--protocol holdout_3)"
    if override != "auto":
        raise ProtocolError(f"Unknown protocol override: {override!r}")
    if n_rows < size_threshold:
        return (
            "cv",
            f"{n_rows} rows < {size_threshold} (small) → N-fold CV with per-fold Optuna",
        )
    return (
        "holdout_3",
        f"{n_rows} rows >= {size_threshold} (large) → holdout_3 × N split/init seeds",
    )


def repeat_seeds(base_seed: int, n: int) -> list[int]:
    if n < 1:
        raise ProtocolError(f"n must be >= 1, got {n}")
    return [int(base_seed) + i for i in range(int(n))]


def resolve_group_col(df: pd.DataFrame, preferred: Optional[str] = None) -> str:
    """Pick a locus-group column for ``assign_splits``."""
    if preferred and preferred in df.columns:
        return preferred
    for name in GROUP_COL_CANDIDATES:
        if name in df.columns:
            return name
    raise ProtocolError(
        "Cannot assign splits: model-format frame has neither target_uid nor group_id. "
        "Fetch via pedb/peen (PE-DB attaches target_uid on export)."
    )


def drop_split_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    for column in (SPLIT_COLUMN, "split_source"):
        if column in output.columns:
            output = output.drop(columns=[column])
    return output


def _n_groups(df: pd.DataFrame, group_col: str) -> int:
    series = df[group_col]
    return int(series.dropna().astype(str).nunique())


def assign_holdout_3(
    df: pd.DataFrame,
    *,
    random_state: int,
    train_pct: float = DEFAULT_HOLDOUT_TRAIN_PCT,
    val_pct: float = DEFAULT_HOLDOUT_VAL_PCT,
    test_pct: float = DEFAULT_HOLDOUT_TEST_PCT,
    use_original_fold: bool = False,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """Group-aware 70/15/15 (or custom) holdout on ``df``."""
    group_col = resolve_group_col(df, group_col)
    n_groups = _n_groups(df, group_col)
    if n_groups < 3:
        raise ProtocolError(
            f"holdout_3 needs at least 3 locus groups, found {n_groups} in {group_col!r}"
        )
    assigned, _summary = assign_splits(
        drop_split_columns(df),
        SplitConfig(
            strategy="holdout_3",
            train_pct=float(train_pct),
            val_pct=float(val_pct),
            test_pct=float(test_pct),
            random_state=int(random_state),
            use_original_fold=bool(use_original_fold),
            group_col=group_col,
        ),
    )
    return assigned


def assign_cv_folds(
    df: pd.DataFrame,
    *,
    cv_folds: int,
    random_state: int,
    use_original_fold: bool = False,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """Assign ``fold_0`` … ``fold_{n-1}`` with no outer test holdout."""
    if cv_folds < 2:
        raise ProtocolError(f"cv_folds must be >= 2, got {cv_folds}")
    group_col = resolve_group_col(df, group_col)
    n_groups = _n_groups(df, group_col)
    if n_groups < cv_folds:
        raise ProtocolError(
            f"N-fold CV needs at least {cv_folds} locus groups, found {n_groups} in {group_col!r}"
        )
    assigned, _summary = assign_splits(
        drop_split_columns(df),
        SplitConfig(
            strategy="cv",
            cv_folds=int(cv_folds),
            random_state=int(random_state),
            use_original_fold=bool(use_original_fold),
            group_col=group_col,
        ),
    )
    return assigned


def assign_inner_train_val(
    df: pd.DataFrame,
    *,
    random_state: int,
    train_pct: float = DEFAULT_INNER_TRAIN_PCT,
    val_pct: float = DEFAULT_INNER_VAL_PCT,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """Split remaining (non-test) rows into ``train`` / ``val`` only.

    Implemented as pe-common ``holdout_2`` (train/test) with ``test`` relabelled
    to ``val``, so Optuna trials train a single model per outer fold.
    """
    if abs(float(train_pct) + float(val_pct) - 1.0) > 1e-6:
        raise ProtocolError(
            f"inner train_pct + val_pct must sum to 1, got {train_pct} + {val_pct}"
        )
    group_col = resolve_group_col(df, group_col)
    n_groups = _n_groups(df, group_col)
    if n_groups < 2:
        raise ProtocolError(
            f"inner train/val split needs at least 2 locus groups, found {n_groups}"
        )
    assigned, _summary = assign_splits(
        drop_split_columns(df),
        SplitConfig(
            strategy="holdout_2",
            train_pct=float(train_pct),
            test_pct=float(val_pct),
            random_state=int(random_state),
            group_col=group_col,
        ),
    )
    assigned = assigned.copy()
    assigned.loc[assigned[SPLIT_COLUMN].astype(str) == "test", SPLIT_COLUMN] = "val"
    return assigned


def remap_cv_fold_to_nested_holdout(
    df: pd.DataFrame,
    fold_label: str,
    *,
    inner_random_state: int,
    inner_train_pct: float = DEFAULT_INNER_TRAIN_PCT,
    inner_val_pct: float = DEFAULT_INNER_VAL_PCT,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """Hold ``fold_label`` out as ``test``; split the rest into train/val for HPO."""
    if SPLIT_COLUMN not in df.columns:
        raise ProtocolError("CV remapping requires an assigned split column")
    labels = set(list_assigned_folds(df))
    if fold_label not in labels:
        raise ProtocolError(
            f"Fold {fold_label!r} not in assigned folds {sorted(labels)}"
        )
    split_series = df[SPLIT_COLUMN].astype(str)
    outer_test = df.loc[split_series == fold_label].copy()
    remaining = df.loc[split_series != fold_label].copy()
    if outer_test.empty or remaining.empty:
        raise ProtocolError(f"Fold {fold_label!r} produced an empty train or test partition")
    group_col = resolve_group_col(df, group_col)
    inner = assign_inner_train_val(
        remaining,
        random_state=inner_random_state,
        train_pct=inner_train_pct,
        val_pct=inner_val_pct,
        group_col=group_col,
    )
    outer_test = outer_test.copy()
    outer_test[SPLIT_COLUMN] = "test"
    if "split_source" not in outer_test.columns:
        outer_test["split_source"] = "group_id"
    combined = pd.concat([inner, outer_test], ignore_index=True)
    return combined


def fold_labels(df: pd.DataFrame) -> list[str]:
    return list_assigned_folds(df)


def split_counts(df: pd.DataFrame) -> dict[str, int]:
    if SPLIT_COLUMN not in df.columns or df.empty:
        return {}
    return df[SPLIT_COLUMN].astype(str).value_counts().astype(int).to_dict()


def extract_eval_metrics(payload: dict[str, Any]) -> dict[str, Optional[float]]:
    """Pull pearson/spearman/mse from a peen evaluate result (model-dependent keys)."""
    metrics = payload.get("metrics") if isinstance(payload, dict) else None
    if not isinstance(metrics, dict):
        return {"pearson": None, "spearman": None, "mse": None}

    def _first(*keys: str) -> Optional[float]:
        for key in keys:
            value = metrics.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    spearman = _first("spearman", "spearman_r", "spearmanr", "rho", "averageedited_spearman")
    if spearman is None:
        named = [
            float(raw)
            for key, raw in metrics.items()
            if str(key).endswith("_spearman")
            for parsed in [_try_float(raw)]
            if parsed is not None
        ]
        spearman = max(named) if named else None
    pearson = _first("pearson", "pearson_r", "pearsonr", "r", "averageedited_pearson")
    mse = _first("mse", "MSE", "val_mse")
    return {"pearson": pearson, "spearman": spearman, "mse": mse}


def _try_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_repeats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean/std of test metrics across successful folds/seeds."""
    ok = [row for row in rows if row.get("status") == "ok"]
    summary: dict[str, Any] = {
        "n_repeats": len(rows),
        "n_ok": len(ok),
    }
    for key in ("test_spearman", "test_pearson", "test_mse"):
        values = [
            float(row[key])
            for row in ok
            if row.get(key) is not None
        ]
        if not values:
            summary[f"{key}_mean"] = None
            summary[f"{key}_std"] = None
            continue
        mean = sum(values) / len(values)
        if len(values) == 1:
            std = 0.0
        else:
            std = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = std
    return summary
