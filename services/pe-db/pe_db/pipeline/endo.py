"""Endogenous standardized window: full PE-core rows on a ~200 bp genomic context.

Vendor crops (DeepPrime 74, OPED 47, PRIDICT ~99) are conversion concerns. Endogenous
standardized rows keep a spacer-centered genomic window large enough that those
crops do not have to invent 3' sequence.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from pe_common.constants import DATA_ROOT
from pe_common.sequence_utils import remove_padding, unpadded_coordinate

from .schema import endo_standard_columns

logger = logging.getLogger(__name__)

ENDO_CONTEXT_BP = 200
ENDO_SPACER_LEN = 20
# 90 bp 5' + 20 bp spacer + 90 bp 3'. DeepPrime needs 4 bp 5' of the spacer and
# 50 bp 3' of spacer-start; OPED needs 4 + 43; PRIDICT's author frame is 10 + 89.
ENDO_SPACER_OFFSET = 90

_PAD_BASES = frozenset("NX-")

GEOMETRY_COLUMNS = (
    "protospacer_location_l",
    "protospacer_location_r",
    "pbs_location_l",
    "pbs_location_r",
    "rtt_location_l",
    "rtt_location_r",
    "lha_location_l",
    "lha_location_r",
    "rha_location_l",
    "rha_location_r",
)

DEEPPE_GENOMIC_LOCI_PATH = DATA_ROOT / "raw" / "deeppe" / "deeppe_genomic_loci.json"
PRIDICT1_LIBRARY2_GENOMIC_LOCI_PATH = (
    DATA_ROOT / "raw" / "pridict1" / "pridict1_library2_genomic_loci.json"
)
MINSEPIE_GENOMIC_LOCI_PATH = DATA_ROOT / "raw" / "minsepie" / "minsepie_genomic_loci.json"


def genomic_window_interval(
    spacer_start_0: int,
    *,
    spacer_offset: int = ENDO_SPACER_OFFSET,
    context_bp: int = ENDO_CONTEXT_BP,
) -> tuple[int, int]:
    """0-based half-open genomic interval covering the target-strand context window.

    ``spacer_start_0`` is the lowest genomic coordinate of the 20-mer on either
    strand. Reverse-strand windows use the same interval, then reverse-complement.
    """
    start = int(spacer_start_0) - int(spacer_offset)
    end = int(spacer_start_0) + (int(context_bp) - int(spacer_offset))
    return start, end


def slice_reference_window_to_endo_context(
    window: str,
    spacer_offset: int,
    *,
    context_bp: int = ENDO_CONTEXT_BP,
    target_offset: int = ENDO_SPACER_OFFSET,
) -> tuple[str, int]:
    """Crop a longer cached window (e.g. MinSePIE 220/100) to the shared 200/90 frame."""
    window = str(window).upper()
    spacer_offset = int(spacer_offset)
    if spacer_offset == target_offset and len(window) == context_bp:
        return window, target_offset
    start = spacer_offset - target_offset
    end = start + context_bp
    if start < 0 or end > len(window):
        raise ValueError(
            f"Cannot slice {len(window)} bp window at spacer_offset={spacer_offset} "
            f"to {context_bp} bp / offset {target_offset}."
        )
    return window[start:end], target_offset


@lru_cache(maxsize=8)
def load_loci_json(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Loci JSON must be an object: {path}")
    return payload


def _load_loci_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing genomic loci metadata: {path}")
    return load_loci_json(str(path))


def load_deeppe_genomic_loci() -> dict[str, Any]:
    return _load_loci_payload(DEEPPE_GENOMIC_LOCI_PATH)


def load_pridict1_library2_genomic_loci() -> dict[str, Any]:
    return _load_loci_payload(PRIDICT1_LIBRARY2_GENOMIC_LOCI_PATH)


def load_minsepie_genomic_loci() -> dict[str, Any]:
    return _load_loci_payload(MINSEPIE_GENOMIC_LOCI_PATH)


def reference_windows_for_keys(loci: dict[str, Any], keys: pd.Series) -> pd.Series:
    """Look up cached ``reference_window`` strings; missing keys become ``<NA>``."""

    def _lookup(key: Any) -> Any:
        if key is None or (isinstance(key, float) and pd.isna(key)):
            return pd.NA
        locus = loci.get(str(key))
        if not locus:
            locus = loci.get(str(key).upper())
        if not locus:
            return pd.NA
        window = locus.get("reference_window")
        if not window:
            return pd.NA
        return str(window).upper()

    return keys.map(_lookup)


def empty_endo_coordinate_row(*, locus_id: Any = pd.NA) -> dict[str, Any]:
    return {
        "endo_genome_build": pd.NA,
        "endo_chr": pd.NA,
        "endo_start": pd.NA,
        "endo_end": pd.NA,
        "endo_strand": pd.NA,
        "endo_coord_ref": pd.NA,
        "endo_coord_source": pd.NA,
        "endo_locus_id": locus_id if locus_id == locus_id else pd.NA,
    }


def endo_coordinate_frame_from_loci(
    keys: pd.Series,
    loci: dict[str, Any],
    *,
    source: str,
    fallback_locus_id: Optional[pd.Series] = None,
    default_build: str = "hg38",
) -> pd.DataFrame:
    """Build ``endo_*`` columns from a loci JSON keyed like ``keys``."""
    rows: list[dict[str, Any]] = []
    fallback = (
        fallback_locus_id
        if fallback_locus_id is not None
        else pd.Series(pd.NA, index=keys.index)
    )
    for key, fallback_id in zip(keys, fallback.reindex(keys.index)):
        key_str = str(key)
        locus_id = (
            str(fallback_id)
            if fallback_id is not None and str(fallback_id) not in {"", "nan", "<NA>"}
            else key_str
        )
        locus = loci.get(key_str) or loci.get(key_str.upper())
        if not locus:
            rows.append(empty_endo_coordinate_row(locus_id=locus_id))
            continue
        build = str(locus.get("genome_build") or default_build)
        chrom = locus.get("chrom")
        strand = locus.get("assembly_strand")
        try:
            strand_val: Any = int(strand) if strand is not None else pd.NA
        except (TypeError, ValueError):
            strand_val = pd.NA
        if "spacer_start" in locus:
            start_0 = int(locus["spacer_start"]) - 1
            end_0 = start_0 + ENDO_SPACER_LEN
            coord_ref = "protospacer"
        elif "variant_start" in locus and "variant_end" in locus:
            start_0 = int(locus["variant_start"]) - 1
            end_0 = int(locus["variant_end"])
            coord_ref = str(locus.get("coord_ref") or "variant")
        else:
            rows.append(empty_endo_coordinate_row(locus_id=locus_id))
            continue
        rows.append(
            {
                "endo_genome_build": build,
                "endo_chr": str(chrom) if chrom is not None else pd.NA,
                "endo_start": start_0,
                "endo_end": end_0,
                "endo_strand": strand_val,
                "endo_coord_ref": coord_ref,
                "endo_coord_source": source,
                "endo_locus_id": locus_id,
            }
        )
    return pd.DataFrame(rows, index=keys.index)


def _trim_prefix_chars(seq: str, n_chars: int) -> str:
    n_chars = max(0, int(n_chars))
    return str(seq)[n_chars:]


def _trim_suffix_genomic(seq: str, n_genomic: int) -> str:
    """Drop ``n_genomic`` non-pad bases from the 3' end (pads at the tail go too)."""
    dropped = 0
    i = len(seq)
    target = max(0, int(n_genomic))
    while i > 0 and dropped < target:
        i -= 1
        if seq[i] not in _PAD_BASES:
            dropped += 1
    return seq[:i]


def expand_aligned_to_endo_context(
    wt: str,
    mut: str,
    protospacer_l: int,
    protospacer_r: int,
    window: str,
    *,
    spacer_offset: int = ENDO_SPACER_OFFSET,
    context_bp: int = ENDO_CONTEXT_BP,
) -> Optional[tuple[str, str, int]]:
    """Embed an aligned WT/Mut pair in ``window``; return ``(wt, mut, coord_shift)``.

    Adds (or trims) genomic flanks so the unpadded spacer starts at
    ``spacer_offset``. Internal indel ``N`` pads are preserved. Returns ``None``
    when the cached window does not match this row's spacer.
    """
    wt = str(wt).upper()
    mut = str(mut).upper()
    window = str(window).upper()
    if len(window) != int(context_bp):
        return None
    protospacer_l = int(protospacer_l)
    spacer_unpadded = unpadded_coordinate(wt, protospacer_l)
    unpadded = remove_padding(wt)
    row_spacer = unpadded[spacer_unpadded : spacer_unpadded + ENDO_SPACER_LEN]
    expected_spacer = window[spacer_offset : spacer_offset + len(row_spacer)]
    mismatch = sum(a != b for a, b in zip(row_spacer, expected_spacer)) + abs(
        len(row_spacer) - len(expected_spacer)
    )
    if not row_spacer or mismatch > 2:
        return None

    five_prime_delta = int(spacer_offset) - int(spacer_unpadded)
    coord_shift = 0
    if five_prime_delta > 0:
        left = window[:five_prime_delta]
        wt = left + wt
        mut = left + mut
        coord_shift = five_prime_delta
    elif five_prime_delta < 0:
        trim = -five_prime_delta
        # No 5' pads in these screens; trim the same prefix from WT and Mut.
        if any(base in _PAD_BASES for base in wt[:trim]) or any(
            base in _PAD_BASES for base in mut[:trim]
        ):
            return None
        wt = _trim_prefix_chars(wt, trim)
        mut = _trim_prefix_chars(mut, trim)
        coord_shift = -trim

    genomic = len(remove_padding(wt))
    need = int(context_bp) - genomic
    if need > 0:
        tail = window[genomic:] if genomic <= len(window) else ""
        if len(tail) != need:
            tail = window[-need:]
        wt = wt + tail
        mut = mut + tail
    elif need < 0:
        wt = _trim_suffix_genomic(wt, -need)
        mut = _trim_suffix_genomic(mut, -need)

    if unpadded_coordinate(wt, protospacer_l + coord_shift) != int(spacer_offset):
        return None
    new_unpadded = remove_padding(wt)
    got = new_unpadded[spacer_offset : spacer_offset + len(row_spacer)]
    mismatch = sum(a != b for a, b in zip(got, row_spacer)) + abs(len(got) - len(row_spacer))
    if mismatch > 2:
        return None
    return wt, mut, coord_shift


def expand_endogenous_frame(
    df: pd.DataFrame,
    windows: pd.Series,
    *,
    spacer_offset: int = ENDO_SPACER_OFFSET,
    context_bp: int = ENDO_CONTEXT_BP,
) -> pd.DataFrame:
    """Expand standardized PE-core rows onto cached genomic windows.

    Rows without a window, or whose spacer does not match the window, are left
    unchanged. Coordinate columns shift by the 5' flank that was prepended/trimmed.
    """
    if df.empty:
        return df
    windows = windows.reindex(df.index)
    out = df.copy()
    expanded = 0
    skipped = 0
    new_wt: list[str] = []
    new_mut: list[str] = []
    shifts: list[int] = []
    for idx, row in out.iterrows():
        window = windows.at[idx] if idx in windows.index else pd.NA
        if window is None or (isinstance(window, float) and pd.isna(window)) or pd.isna(window):
            new_wt.append(str(row["wt_sequence"]))
            new_mut.append(str(row["mut_sequence"]))
            shifts.append(0)
            skipped += 1
            continue
        result = expand_aligned_to_endo_context(
            str(row["wt_sequence"]),
            str(row["mut_sequence"]),
            int(row["protospacer_location_l"]),
            int(row["protospacer_location_r"]),
            str(window),
            spacer_offset=spacer_offset,
            context_bp=context_bp,
        )
        if result is None:
            new_wt.append(str(row["wt_sequence"]))
            new_mut.append(str(row["mut_sequence"]))
            shifts.append(0)
            skipped += 1
            continue
        wt, mut, shift = result
        new_wt.append(wt)
        new_mut.append(mut)
        shifts.append(shift)
        expanded += 1
    out["wt_sequence"] = new_wt
    out["mut_sequence"] = new_mut
    shift_series = pd.Series(shifts, index=out.index, dtype=int)
    if shift_series.ne(0).any():
        for column in GEOMETRY_COLUMNS:
            if column in out.columns:
                out[column] = out[column].astype(int) + shift_series
    logger.info(
        "Expanded %s endogenous row(s) onto %s bp windows (%s unchanged).",
        expanded,
        context_bp,
        skipped,
    )
    return out


def assert_endo_columns(df: pd.DataFrame) -> None:
    missing = [column for column in endo_standard_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Endogenous frame missing columns: {missing}")
