from __future__ import annotations

import pandas as pd
from typing import Optional

def _parse_pridict_location_column(
    location_series: pd.Series, column_name: str
) -> tuple[pd.Series, pd.Series]:
    """Vectorized parser for PRIDICT location strings like '[13, 26]'."""
    series = pd.Series(location_series, copy=False)
    extracted = series.astype('string').str.extract(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")
    invalid = pd.Series(extracted.isna().any(axis=1), index=series.index)
    if bool(invalid.any()):
        bad_examples = [str(value) for value in series[invalid].tolist()[:3]]
        raise ValueError(
            f"Invalid location format in column {column_name}: {bad_examples}"
        )
    return extracted[0].astype(int), extracted[1].astype(int)


def _attach_pridict_outcome_distribution(
    output_df: pd.DataFrame,
    source_df: pd.DataFrame,
    *,
    edited_column: Optional[str] = None,
) -> pd.DataFrame:
    """Preserve PRIDICT edited/unedited/indel fractions for KL/CE distribution training.

    Source tables expose either plain ``average*`` columns (library1) or cell-line
    prefixed names (``HEK293T_averageedited``). When only the edited column is
    known, derive sibling names by suffix substitution.

    Stores the matched trio under ``averageedited`` / ``averageunedited`` /
    ``averageindel`` so convert+KL training use a consistent distribution (not a
    mix of PE2-only efficiency with library-average unedited/indel).
    """
    out = output_df.copy()

    def _series(name: str) -> Optional[pd.Series]:
        if name not in source_df.columns:
            return None
        return pd.to_numeric(source_df[name], errors="coerce")

    edited = _series("averageedited")
    unedited = _series("averageunedited")
    indel = _series("averageindel")

    if edited_column and edited_column in source_df.columns:
        edited = _series(edited_column)
        if edited_column.endswith("averageedited"):
            prefix = edited_column[: -len("averageedited")]
            unedited = unedited if unedited is not None else _series(f"{prefix}averageunedited")
            indel = indel if indel is not None else _series(f"{prefix}averageindel")

    if edited is not None:
        out["averageedited"] = edited.astype(float).to_numpy()
    if unedited is not None:
        out["averageunedited"] = unedited.astype(float).to_numpy()
    if indel is not None:
        out["averageindel"] = indel.astype(float).to_numpy()
    return out


def _correction_type_to_flags(
    correction_type: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    labels = correction_type.astype("string").str.strip().str.lower()
    type_sub = labels.eq("replacement")
    type_ins = labels.eq("insertion")
    type_del = labels.eq("deletion")
    unknown_mask = ~(type_sub | type_ins | type_del)
    if unknown_mask.any():
        unknown_values = labels.loc[unknown_mask].unique().tolist()[:5]
        raise ValueError(f"Unsupported Correction_Type values: {unknown_values}")
    return type_sub.astype(bool), type_ins.astype(bool), type_del.astype(bool)

