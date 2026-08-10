"""Centralized train/validation/test and cross-validation split assignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional

import numpy as np
import pandas as pd

from .data_utils import _stable_group_sort_key, target_location_group_series

SplitStrategy = Literal["none", "holdout_2", "holdout_3", "cv"]
SplitSource = Literal["original_fold", "group_id"]

_PCT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for group-aware dataset splitting."""

    strategy: SplitStrategy = "none"
    train_pct: Optional[float] = None
    val_pct: Optional[float] = None
    test_pct: Optional[float] = None
    cv_folds: Optional[int] = None
    use_original_fold: bool = False
    original_fold_test_value: float = -1.0
    original_fold_col: str = "original_fold"
    group_col: str = "group_id"
    random_state: int = 42
    fold_namespace_prefix: Optional[str] = None
    group_scope_col: Optional[str] = None

    def __post_init__(self) -> None:
        validate_split_config(self)


def validate_split_config(config: SplitConfig) -> None:
    """Validate split parameters for the selected strategy."""
    strategy = config.strategy

    if strategy == "none":
        return

    if strategy == "holdout_2":
        if config.cv_folds is not None:
            raise ValueError("cv_folds cannot be set when strategy is holdout_2")
        if config.val_pct is not None:
            raise ValueError("val_pct cannot be set when strategy is holdout_2")
        if config.train_pct is None or config.test_pct is None:
            raise ValueError("holdout_2 requires train_pct and test_pct")
        _require_fraction_sum({"train": config.train_pct, "test": config.test_pct})
        return

    if strategy == "holdout_3":
        if config.cv_folds is not None:
            raise ValueError("cv_folds cannot be set when strategy is holdout_3")
        if config.train_pct is None or config.val_pct is None or config.test_pct is None:
            raise ValueError("holdout_3 requires train_pct, val_pct, and test_pct")
        _require_fraction_sum(
            {
                "train": config.train_pct,
                "val": config.val_pct,
                "test": config.test_pct,
            }
        )
        return

    if strategy == "cv":
        if config.train_pct is not None or config.val_pct is not None:
            raise ValueError("cv strategy accepts only test_pct (optional holdout), not train_pct or val_pct")
        if config.cv_folds is None or int(config.cv_folds) < 2:
            raise ValueError("cv strategy requires cv_folds >= 2")
        if config.test_pct is not None and not 0 < config.test_pct < 1:
            raise ValueError(f"test_pct must be in (0, 1) for cv holdout, got {config.test_pct}")
        return

    raise ValueError(f"Unsupported split strategy: {strategy!r}")


def _require_fraction_sum(fractions: Mapping[str, float]) -> None:
    total = float(sum(fractions.values()))
    if abs(total - 1.0) > _PCT_TOLERANCE:
        labels = ", ".join(f"{name}={value}" for name, value in fractions.items())
        raise ValueError(f"Split fractions must sum to 1, got {total} ({labels})")


def _fold_label(fold_id: int, prefix: Optional[str]) -> str:
    label = f"fold_{fold_id}"
    if prefix:
        return f"{prefix}|{label}"
    return label


def _shuffled_groups(unique_groups: list[Any], random_state: int) -> list[Any]:
    ordered = sorted(unique_groups, key=_stable_group_sort_key)
    rng = np.random.default_rng(random_state)
    rng.shuffle(ordered)
    return ordered


def _assign_holdout_groups(
    groups: list[Any],
    *,
    train_pct: float,
    val_pct: Optional[float],
    test_pct: float,
    random_state: int,
) -> dict[Any, str]:
    if not groups:
        return {}

    ordered = _shuffled_groups(groups, random_state)
    n = len(ordered)
    n_test = max(1, int(np.ceil(n * test_pct))) if n > 1 else 1
    n_test = min(n_test, n)

    if val_pct is None:
        n_train = n - n_test
        if n_train < 1:
            raise ValueError("Not enough groups for the requested train/test fractions")
        assignment: dict[Any, str] = {}
        for group in ordered[:n_test]:
            assignment[group] = "test"
        for group in ordered[n_test:]:
            assignment[group] = "train"
        return assignment

    n_val = max(1, int(np.ceil(n * val_pct))) if n > 2 else 1
    n_val = min(n_val, max(1, n - n_test))
    n_train = n - n_test - n_val
    if n_train < 1:
        raise ValueError("Not enough groups for the requested train/val/test fractions")

    assignment = {}
    offset = 0
    for group in ordered[offset : offset + n_test]:
        assignment[group] = "test"
    offset += n_test
    for group in ordered[offset : offset + n_val]:
        assignment[group] = "val"
    offset += n_val
    for group in ordered[offset:]:
        assignment[group] = "train"
    return assignment


def _assign_train_val_groups(
    groups: list[Any],
    *,
    train_pct: float,
    val_pct: float,
    random_state: int,
) -> dict[Any, str]:
    """Split groups into train and val partitions (fractions must sum to 1)."""
    if not groups:
        return {}

    _require_fraction_sum({"train": train_pct, "val": val_pct})
    ordered = _shuffled_groups(groups, random_state)
    n = len(ordered)
    n_val = max(1, int(np.ceil(n * val_pct))) if n > 1 else 1
    n_val = min(n_val, max(1, n - 1))
    n_train = n - n_val
    if n_train < 1:
        raise ValueError("Not enough groups for the requested train/val fractions")

    assignment: dict[Any, str] = {}
    for group in ordered[:n_val]:
        assignment[group] = "val"
    for group in ordered[n_val:]:
        assignment[group] = "train"
    return assignment


def _assign_holdout_3_groups(
    unique_groups: list[_SplitGroupKey],
    config: SplitConfig,
) -> tuple[dict[_SplitGroupKey, str], dict[_SplitGroupKey, SplitSource]]:
    """
    Three-way holdout: assign test like holdout_2, then split the remainder train/val.
    """
    group_to_split: dict[_SplitGroupKey, str] = {}
    group_to_source: dict[_SplitGroupKey, SplitSource] = {}
    train_val_pool: list[_SplitGroupKey] = []
    synthetic_groups: list[_SplitGroupKey] = []

    for group in unique_groups:
        if (
            config.use_original_fold
            and isinstance(group, tuple)
            and len(group) == 2
            and group[0] == "author_fold"
        ):
            fold_value = float(group[1])
            if np.isclose(fold_value, config.original_fold_test_value):
                group_to_split[group] = "test"
                group_to_source[group] = "original_fold"
            else:
                train_val_pool.append(group)
        else:
            synthetic_groups.append(group)

    if synthetic_groups:
        test_map = _assign_holdout_groups(
            synthetic_groups,
            train_pct=float(config.train_pct),
            val_pct=None,
            test_pct=float(config.test_pct),
            random_state=config.random_state,
        )
        for group, label in test_map.items():
            if label == "test":
                group_to_split[group] = "test"
                group_to_source[group] = "group_id"
            else:
                train_val_pool.append(group)

    if train_val_pool:
        train_pool_pct = float(config.train_pct)
        val_pool_pct = float(config.val_pct)
        pool_total = train_pool_pct + val_pool_pct
        tv_map = _assign_train_val_groups(
            train_val_pool,
            train_pct=train_pool_pct / pool_total,
            val_pct=val_pool_pct / pool_total,
            random_state=config.random_state,
        )
        for group, label in tv_map.items():
            group_to_split[group] = label
            if (
                isinstance(group, tuple)
                and len(group) == 2
                and group[0] == "author_fold"
            ):
                group_to_source[group] = "original_fold"
            else:
                group_to_source[group] = "group_id"

    return group_to_split, group_to_source


def _assign_cv_groups(
    groups: list[Any],
    *,
    cv_folds: int,
    test_pct: Optional[float],
    random_state: int,
) -> dict[Any, str]:
    if not groups:
        return {}

    ordered = _shuffled_groups(groups, random_state)
    assignment: dict[Any, str] = {}

    if test_pct is not None:
        n_test = max(1, int(np.ceil(len(ordered) * test_pct))) if len(ordered) > 1 else 1
        n_test = min(n_test, len(ordered))
        test_groups = ordered[:n_test]
        cv_groups = ordered[n_test:]
        for group in test_groups:
            assignment[group] = "test"
    else:
        cv_groups = ordered

    if not cv_groups:
        return assignment

    for index, group in enumerate(cv_groups):
        assignment[group] = f"fold_{index % cv_folds}"
    return assignment


def _resolve_group_series(
    df: pd.DataFrame,
    group_col: str,
    composite_group_prefix: Optional[str],
    group_scope_col: Optional[str] = None,
) -> pd.Series:
    if group_col not in df.columns:
        raise ValueError(f"group_col {group_col!r} is missing from dataframe")
    groups = pd.Series(df[group_col], copy=False).astype(str)
    if group_scope_col and group_scope_col in df.columns:
        scopes = df[group_scope_col].astype(str)
        return groups + "@" + scopes
    if composite_group_prefix:
        return groups + "@" + composite_group_prefix
    return pd.Series(df[group_col], copy=False)


_AuthorFoldGroupKey = tuple[str, float]
_SyntheticGroupKey = tuple[str, Any]
_SplitGroupKey = _AuthorFoldGroupKey | _SyntheticGroupKey | Any


def _resolve_split_group_series(
    df: pd.DataFrame,
    config: SplitConfig,
    *,
    composite_group_prefix: Optional[str] = None,
) -> pd.Series:
    """Build per-row split group keys for assignment."""
    if config.use_original_fold and config.original_fold_col in df.columns:
        fold_values = pd.to_numeric(df[config.original_fold_col], errors="coerce")
        author_mask = fold_values.notna()
        row_groups = pd.Series(index=df.index, dtype=object)

        if author_mask.any():
            for idx in df.index[author_mask]:
                row_groups.loc[idx] = ("author_fold", float(fold_values.loc[idx]))

        if (~author_mask).any():
            target_groups = target_location_group_series(df.loc[~author_mask])
            for idx in df.index[~author_mask]:
                row_groups.loc[idx] = ("target_location", target_groups.loc[idx])

        return row_groups

    return _resolve_group_series(
        df,
        config.group_col,
        composite_group_prefix,
        config.group_scope_col,
    )


def _map_original_fold_to_split(
    fold_value: float,
    *,
    strategy: SplitStrategy,
    test_value: float,
    fold_namespace_prefix: Optional[str],
) -> str:
    if np.isclose(fold_value, test_value):
        return "test"

    if strategy in ("holdout_2", "holdout_3"):
        return "train"

    if strategy == "cv":
        if fold_value < 0:
            raise ValueError(
                f"Unexpected original_fold value {fold_value} for cv strategy; "
                f"expected {test_value} for test or non-negative fold ids"
            )
        fold_id = int(fold_value)
        return _fold_label(fold_id, fold_namespace_prefix)

    raise ValueError(f"Cannot map original_fold for strategy {strategy!r}")


def assign_splits(
    df: pd.DataFrame,
    config: SplitConfig,
    *,
    composite_group_prefix: Optional[str] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Assign ``split`` and ``split_source`` columns according to ``config``.

    When ``use_original_fold=True``, author ``original_fold`` values drive assignment
    and stored ``group_id`` values are ignored. Rows without author folds fall back to
    target-location grouping (protospacer). After merging datasheets, callers should
    reassign ``group_id`` via ``reassign_group_ids_by_target_location`` before calling
    this function with ``use_original_fold=False``.

    Args:
        df: Input dataframe with ``group_col`` and optionally ``original_fold_col``.
        config: Split configuration (validated in ``SplitConfig.__post_init__``).
        composite_group_prefix: Deprecated; retained for backward compatibility.

    Returns:
        Copy of ``df`` with ``split`` / ``split_source`` columns, plus summary metadata.
    """
    if config.strategy == "none":
        return df.copy(), {"strategy": "none"}

    if df.empty:
        empty = df.copy()
        empty["split"] = pd.Series(dtype="string")
        empty["split_source"] = pd.Series(dtype="string")
        return empty, {"strategy": config.strategy, "by_partition": {}, "by_source": {}}

    output = df.copy()
    group_series = _resolve_split_group_series(
        output,
        config,
        composite_group_prefix=composite_group_prefix,
    )
    unique_groups = group_series.dropna().unique().tolist()

    fold_prefix = config.fold_namespace_prefix
    if composite_group_prefix and fold_prefix is None:
        fold_prefix = composite_group_prefix

    group_to_split: dict[_SplitGroupKey, str] = {}
    group_to_source: dict[_SplitGroupKey, SplitSource] = {}

    if config.strategy == "holdout_3":
        group_to_split, group_to_source = _assign_holdout_3_groups(unique_groups, config)
    else:
        synthetic_groups: list[_SplitGroupKey] = []
        for group in unique_groups:
            if (
                isinstance(group, tuple)
                and len(group) == 2
                and group[0] == "author_fold"
            ):
                fold_value = float(group[1])
                group_to_split[group] = _map_original_fold_to_split(
                    fold_value,
                    strategy=config.strategy,
                    test_value=config.original_fold_test_value,
                    fold_namespace_prefix=fold_prefix if config.strategy == "cv" else None,
                )
                group_to_source[group] = "original_fold"
            else:
                synthetic_groups.append(group)

        if synthetic_groups:
            if config.strategy == "holdout_2":
                synthetic_map = _assign_holdout_groups(
                    synthetic_groups,
                    train_pct=float(config.train_pct),
                    val_pct=None,
                    test_pct=float(config.test_pct),
                    random_state=config.random_state,
                )
            elif config.strategy == "cv":
                synthetic_map = _assign_cv_groups(
                    synthetic_groups,
                    cv_folds=int(config.cv_folds),
                    test_pct=config.test_pct,
                    random_state=config.random_state,
                )
                if fold_prefix:
                    synthetic_map = {
                        group: (
                            _fold_label(int(label.split("_", 1)[1]), fold_prefix)
                            if label.startswith("fold_")
                            else label
                        )
                        for group, label in synthetic_map.items()
                    }
            else:
                raise ValueError(f"Unsupported split strategy: {config.strategy!r}")

            for group, split_label in synthetic_map.items():
                group_to_split[group] = split_label
                group_to_source[group] = "group_id"

    split_values = group_series.map(group_to_split)
    source_values = group_series.map(group_to_source)

    output["split"] = split_values.astype("string")
    output["split_source"] = source_values.astype("string")

    summary = summarize_splits(output)
    summary["strategy"] = config.strategy
    summary["use_original_fold"] = config.use_original_fold
    if composite_group_prefix:
        summary["composite_group_prefix"] = composite_group_prefix
    return output, summary


def summarize_splits(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize partition and source counts from assigned split columns."""
    if "split" not in df.columns:
        return {"by_partition": {}, "by_source": {}}

    by_partition = (
        df["split"].value_counts(dropna=False).astype(int).to_dict()
        if len(df) > 0
        else {}
    )
    by_source: dict[str, int] = {}
    if "split_source" in df.columns and len(df) > 0:
        by_source = df["split_source"].value_counts(dropna=False).astype(int).to_dict()
    return {"by_partition": by_partition, "by_source": by_source}


def split_config_from_params(
    *,
    strategy: SplitStrategy,
    train_pct: Optional[float] = None,
    val_pct: Optional[float] = None,
    test_pct: Optional[float] = None,
    cv_folds: Optional[int] = None,
    use_original_fold: bool = False,
    original_fold_test_value: float = -1.0,
    random_state: int = 42,
    fold_namespace_prefix: Optional[str] = None,
) -> SplitConfig:
    """Build a validated ``SplitConfig`` from API-style parameters."""
    return SplitConfig(
        strategy=strategy,
        train_pct=train_pct,
        val_pct=val_pct,
        test_pct=test_pct,
        cv_folds=cv_folds,
        use_original_fold=use_original_fold,
        original_fold_test_value=original_fold_test_value,
        random_state=random_state,
        fold_namespace_prefix=fold_namespace_prefix,
    )


SPLIT_COLUMN = "split"
SPLIT_SOURCE_COLUMN = "split_source"


def _split_labels(df: pd.DataFrame, split_col: str = SPLIT_COLUMN) -> set[str]:
    if split_col not in df.columns or df.empty:
        return set()
    return {str(value) for value in df[split_col].dropna().astype(str).unique()}


def list_assigned_folds(df: pd.DataFrame, split_col: str = SPLIT_COLUMN) -> list[str]:
    """Return sorted CV fold labels (``fold_0``, ``fold_1``, or prefixed variants)."""
    labels = _split_labels(df, split_col)

    def _fold_sort_key(label: str) -> tuple[int, str]:
        if "|fold_" in label:
            prefix, _, suffix = label.partition("|fold_")
            return (int(suffix), prefix)
        if label.startswith("fold_"):
            return (int(label.split("_", 1)[1]), "")
        return (-1, label)

    fold_labels = [label for label in labels if "fold_" in label]
    return sorted(fold_labels, key=_fold_sort_key)


def has_assigned_cv_folds(df: pd.DataFrame, split_col: str = SPLIT_COLUMN) -> bool:
    return bool(list_assigned_folds(df, split_col=split_col))


def select_split_partition(
    df: pd.DataFrame,
    partition: str,
    *,
    split_col: str = SPLIT_COLUMN,
) -> pd.DataFrame:
    if split_col not in df.columns:
        raise ValueError(f"Missing {split_col!r} column.")
    mask = df[split_col].astype(str) == str(partition)
    return df.loc[mask].copy().reset_index(drop=True)


def exclude_test_partition(
    df: pd.DataFrame,
    *,
    split_col: str = SPLIT_COLUMN,
) -> pd.DataFrame:
    """Remove held-out test rows before training."""
    if split_col not in df.columns:
        return df.copy()
    return df.loc[df[split_col].astype(str) != "test"].copy().reset_index(drop=True)


def select_cv_fold_partitions(
    df: pd.DataFrame,
    *,
    split_col: str = SPLIT_COLUMN,
) -> pd.DataFrame:
    """Return all rows assigned to a CV fold label (``fold_0``, ``fold_1``, …)."""
    fold_labels = list_assigned_folds(df, split_col=split_col)
    if not fold_labels:
        return df.iloc[0:0].copy()
    split_series = df[split_col].astype(str)
    return df.loc[split_series.isin(fold_labels)].copy().reset_index(drop=True)


def select_evaluation_partition(
    df: pd.DataFrame,
    *,
    split_col: str = SPLIT_COLUMN,
    require_test: bool = True,
) -> pd.DataFrame:
    """Keep rows used for model evaluation.

    Prefer an explicit ``test`` holdout when present. Otherwise, when CV fold
    labels are assigned (``fold_0``, ``fold_1``, …), return all fold rows as a
    pooled validation set so cross-validation exports do not require designating
    one fold as ``test``.
    """
    if split_col not in df.columns:
        return df.copy()
    labels = _split_labels(df, split_col)
    if "test" in labels:
        test_df = select_split_partition(df, "test", split_col=split_col)
        if test_df.empty:
            raise ValueError("Test partition is empty.")
        return test_df

    fold_df = select_cv_fold_partitions(df, split_col=split_col)
    if not fold_df.empty:
        return fold_df

    if require_test:
        raise ValueError(
            "Evaluation data has no test partition or CV fold assignments. "
            "Fetch from PE-DB with a split strategy that defines test rows or CV folds."
        )
    return df.copy()


def iter_assigned_cv_folds(
    df: pd.DataFrame,
    *,
    split_col: str = SPLIT_COLUMN,
):
    """Yield ``(fold_label, train_df, val_df)`` using PE-DB CV assignments."""
    fold_labels = list_assigned_folds(df, split_col=split_col)
    if not fold_labels:
        return
    split_series = df[split_col].astype(str)
    test_mask = split_series == "test"
    for fold_label in fold_labels:
        val_df = df.loc[split_series == fold_label].copy()
        train_df = df.loc[~(test_mask | (split_series == fold_label))].copy()
        if train_df.empty or val_df.empty:
            continue
        yield fold_label, train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def resolve_train_val_from_splits(
    df: pd.DataFrame,
    val_data: Optional[pd.DataFrame] = None,
    *,
    split_col: str = SPLIT_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve train/validation frames from PE-DB ``split`` assignments."""
    if val_data is not None:
        return df.copy().reset_index(drop=True), val_data.copy().reset_index(drop=True)

    if split_col not in df.columns:
        raise ValueError(
            "Training requires PE-DB split assignments (column 'split') or explicit val_data. "
            "Fetch data via GET /api/filter with split_strategy=holdout_3 (or cv)."
        )

    labels = _split_labels(df, split_col)
    if "train" in labels and "val" in labels:
        train_df = select_split_partition(df, "train", split_col=split_col)
        val_df = select_split_partition(df, "val", split_col=split_col)
        if train_df.empty or val_df.empty:
            raise ValueError("Both train and val partitions must be non-empty.")
        return train_df, val_df

    if has_assigned_cv_folds(df, split_col=split_col):
        return resolve_final_train_val_for_cv_export(df, split_col=split_col)

    raise ValueError(
        "Split assignments must include train and val (holdout_3) or CV fold labels. "
        f"Found: {sorted(labels)}"
    )


def resolve_final_train_val_for_cv_export(
    df: pd.DataFrame,
    *,
    split_col: str = SPLIT_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pick a deterministic holdout fold for final early stopping after CV export."""
    fold_labels = list_assigned_folds(df, split_col=split_col)
    if not fold_labels:
        raise ValueError("No CV fold assignments found.")
    holdout_fold = fold_labels[-1]
    split_series = df[split_col].astype(str)
    val_df = df.loc[split_series == holdout_fold].copy().reset_index(drop=True)
    train_df = df.loc[~split_series.isin(["test", holdout_fold])].copy().reset_index(drop=True)
    if train_df.empty or val_df.empty:
        raise ValueError("Unable to derive final train/val frames from CV assignments.")
    return train_df, val_df

