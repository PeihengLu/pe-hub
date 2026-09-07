from __future__ import annotations

import json
import logging
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from pe_common.constants import DATA_ROOT
from pe_common.sequence_utils import (
    align_wt_mut_sequences,
    reverse_complement,
    shift_coords_after_indel_pad,
)

from ..catalog.records import DatasheetScaffoldAssignment
from ..catalog.scaffolds import (
    MINSEPIE_DATASET_SCAFFOLD_ID,
    SCAFFOLD_ID_CONVENTIONAL,
    SCAFFOLD_ID_OPTIPRIME_BLPI_FE,
    default_scaffold_for_pridict,
    scaffold_id_from_deepprime_label,
)
from ..catalog.studies import get_dataset_record
from ..config import get_settings
from ..pipeline.names import _normalize_name
from ..pipeline.registry import StudyPipeline, register_study
from ..pipeline.schema import (
    _attach_endo_coordinate_columns,
    _build_standardized_output_df,
    _coerce_original_fold,
    _shift_coords_after_wt_mut_align,
    _write_partial_standardized_output,
    endo_standard_columns,
)
from ..utils.deepspcas9 import fill_missing_spcas9_scores

logger = logging.getLogger(__name__)

from .deepprime import _standardize_deepprime_ontarget

def _clean_deeppe_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten multi-line Excel headers into single-line column labels."""
    out = df.copy()
    out.columns = [str(col).replace("\n", " ").strip() for col in out.columns]
    return out


def _deeppe_br_tr_replicate_columns(columns, prefix: str) -> list[str]:
    """
    Return BR1/BR2 × TR1/TR2 efficiency columns for a cell-line prefix (e.g. HCT, MDA, Endo).
    """
    expected_order = [
        f"{prefix}-BR1-TR1",
        f"{prefix}-BR1-TR2",
        f"{prefix}-BR2-TR1",
        f"{prefix}-BR2-TR2",
    ]
    by_short: dict[str, str] = {}
    for col in columns:
        col_str = str(col).strip()
        for key in expected_order:
            if col_str.startswith(key):
                by_short[key] = col
                break
    return [by_short[key] for key in expected_order if key in by_short]


def _average_deeppe_replicates(df: pd.DataFrame, replicate_columns: list[str]) -> pd.Series:
    """Mean editing efficiency across biological and technical replicates (skipna)."""
    if not replicate_columns:
        raise ValueError("No replicate columns provided for DeepPE averaging.")
    numeric = df[replicate_columns].apply(pd.to_numeric, errors="coerce")
    return numeric.mean(axis=1, skipna=True)


def _save_deeppe_export(
    df: pd.DataFrame,
    *,
    dataset: str,
    cell_line: str,
    pe_system: str = "pe2",
) -> None:
    output_path = (
        DATA_ROOT / "exported" / "deeppe" / dataset / f"{cell_line}-{pe_system}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved DeepPE datasheet: %s (%s rows)", output_path, len(df))


def _read_deeppe_moesm4_sheet(excel_path: Path, sheet_name: str) -> pd.DataFrame:
    return _clean_deeppe_dataframe_columns(
        pd.read_excel(excel_path, sheet_name=sheet_name, header=1)
    )


def _export_deeppe_moesm4_datasheets() -> None:
    """
    Export DeepPE high-throughput libraries from Kim et al. MOESM4 (Suppl. Tables 3–4).

    Library 1 → ``deeppe-ht``; library 2 type/position splits → ``deeppe-type`` /
    ``deeppe-position``. All measured in HEK293T with PE2 on lentiviral reporters.
    """
    excel_path = DATA_ROOT / "raw" / "deeppe" / "41587_2020_677_MOESM4_ESM.xlsx"
    cell_line = _normalize_name("HEK293T")
    pe_system = "pe2"

    ht_data = _read_deeppe_moesm4_sheet(excel_path, "Library 1 (HT-training, test)")
    ht_data = ht_data.rename(
        columns={
            "Datat set name": "dataset_split",
            "Measured PE efficiency": "editing_efficiency",
        }
    )
    _save_deeppe_export(
        ht_data,
        dataset="deeppe-ht",
        cell_line=cell_line,
        pe_system=pe_system,
    )

    position_type_data = _read_deeppe_moesm4_sheet(excel_path, "Library 2 (Position, Type)")
    position_type_data = position_type_data.rename(
        columns={
            "Datat set name": "dataset_split",
            "Measured PE efficiency": "editing_efficiency",
        }
    )
    split_series = position_type_data["dataset_split"].astype(str)
    type_data = position_type_data[split_series.str.contains("Type", case=False, na=False)]
    position_data = position_type_data[split_series.str.contains("Position", case=False, na=False)]

    _save_deeppe_export(
        type_data,
        dataset="deeppe-type",
        cell_line=cell_line,
        pe_system=pe_system,
    )
    _save_deeppe_export(
        position_data,
        dataset="deeppe-position",
        cell_line=cell_line,
        pe_system=pe_system,
    )


def _drop_deeppe_replicate_columns(df: pd.DataFrame, replicate_columns: list[str]) -> pd.DataFrame:
    """Remove raw replicate efficiency columns after ``editing_efficiency`` is computed."""
    return df.drop(columns=replicate_columns, errors="ignore")


def _export_deeppe_endogenous_datasheets() -> None:
    """
    Export DeepPE endogenous validation sets with replicate-averaged efficiency.

    Sources:
      - ``deeppe_endogenous.xlsx`` (Suppl. Table 3): HEK293T, 33 sites
      - ``41587_2020_677_MOESM5_ESM.xlsx`` (Suppl. Table 5): HCT116 and MDA-MB-231

    ``editing_efficiency`` is the mean of BR1/BR2 × TR1/TR2 replicate columns per cell line.
    """
    pe_system = "pe2"
    endogenous_exports = (
        (
            DATA_ROOT / "raw" / "deeppe" / "deeppe_endogenous.xlsx",
            "Data set Endo",
            "Endo",
            _normalize_name("HEK293T"),
        ),
        (
            DATA_ROOT / "raw" / "deeppe" / "41587_2020_677_MOESM5_ESM.xlsx",
            "Data set HCT and MDA",
            "HCT",
            _normalize_name("HCT116"),
        ),
        (
            DATA_ROOT / "raw" / "deeppe" / "41587_2020_677_MOESM5_ESM.xlsx",
            "Data set HCT and MDA",
            "MDA",
            _normalize_name("MDA-MB-231"),
        ),
    )

    for excel_path, sheet_name, replicate_prefix, cell_line in endogenous_exports:
        df = _clean_deeppe_dataframe_columns(
            pd.read_excel(excel_path, sheet_name=sheet_name, header=1)
        )
        replicate_columns = _deeppe_br_tr_replicate_columns(df.columns, replicate_prefix)
        if len(replicate_columns) != 4:
            raise ValueError(
                f"Expected 4 replicate columns for prefix {replicate_prefix!r} in {excel_path}, "
                f"found {len(replicate_columns)}: {replicate_columns}"
            )
        export_df = df.copy()
        export_df["editing_efficiency"] = _average_deeppe_replicates(export_df, replicate_columns)
        export_df = _drop_deeppe_replicate_columns(export_df, replicate_columns)
        # MOESM5 contains both HCT and MDA columns; keep only the target line's replicates.
        if replicate_prefix in {"HCT", "MDA"}:
            other_prefix = "MDA" if replicate_prefix == "HCT" else "HCT"
            other_cols = [
                col
                for col in export_df.columns
                if str(col).strip().startswith(f"{other_prefix}-BR")
            ]
            export_df = export_df.drop(columns=other_cols, errors="ignore")
        elif replicate_prefix == "Endo":
            extra_endo_cols = [
                col
                for col in export_df.columns
                if str(col).strip().startswith("Endo-BR")
                and col not in replicate_columns
            ]
            export_df = export_df.drop(columns=extra_endo_cols, errors="ignore")
        _save_deeppe_export(
            export_df,
            dataset="deeppe-endo",
            cell_line=cell_line,
            pe_system=pe_system,
        )


def _export_deeppe_datasheets() -> None:
    """Export all DeepPE supplementary tables to ``datasets/exported/deeppe/``."""
    _export_deeppe_moesm4_datasheets()
    _export_deeppe_endogenous_datasheets()


def _load_deeppe_genomic_loci() -> dict[str, Any]:
    """Load hg38 protospacer anchors for DeepPE endogenous wide targets."""
    if not _DEEPPE_GENOMIC_LOCI_PATH.exists():
        raise FileNotFoundError(
            f"Missing DeepPE genomic loci metadata: {_DEEPPE_GENOMIC_LOCI_PATH}"
        )
    with _DEEPPE_GENOMIC_LOCI_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def _load_pridict1_library2_genomic_loci() -> dict[str, Any]:
    """Load hg38/mm39 anchors for PRIDICT1 library2-invivo Names."""
    if not _PRIDICT1_LIBRARY2_GENOMIC_LOCI_PATH.exists():
        raise FileNotFoundError(
            "Missing PRIDICT1 library2 genomic loci metadata: "
            f"{_PRIDICT1_LIBRARY2_GENOMIC_LOCI_PATH}"
        )
    with _PRIDICT1_LIBRARY2_GENOMIC_LOCI_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _deeppe_endo_coordinates(wt_sequences: pd.Series) -> pd.DataFrame:
    """Map DeepPE endogenous wide-target sequences to hg38 protospacer coordinates."""
    meta = _load_deeppe_genomic_loci()
    loci = meta.get("loci", {})
    rows: list[dict[str, Any]] = []
    for seq in wt_sequences.astype(str).str.upper().str.replace("U", "T", regex=False):
        locus = loci.get(seq)
        if not locus or "spacer_start" not in locus:
            rows.append(
                {
                    "endo_genome_build": pd.NA,
                    "endo_chr": pd.NA,
                    "endo_start": pd.NA,
                    "endo_end": pd.NA,
                    "endo_strand": pd.NA,
                    "endo_coord_ref": pd.NA,
                    "endo_coord_source": pd.NA,
                    "endo_locus_id": pd.NA,
                }
            )
            continue
        spacer_start_0 = int(locus["spacer_start"]) - 1
        rows.append(
            {
                "endo_genome_build": "hg38",
                "endo_chr": str(locus["chrom"]),
                "endo_start": spacer_start_0,
                "endo_end": spacer_start_0 + 20,
                "endo_strand": int(locus["assembly_strand"]),
                "endo_coord_ref": "protospacer",
                "endo_coord_source": "deeppe_genomic_loci.json",
                "endo_locus_id": f"{locus['chrom']}:{locus['spacer_start']}",
            }
        )
    return pd.DataFrame(rows, index=wt_sequences.index)



# DeepPE (Kim et al. 2021) uses 47 bp wide-target reporters with X-masked
# prime-edited sequences (library 1) or pegRNA 3' extensions (libraries 2 / endo).
_DEEPPE_WIDE_COLUMNS = (
    "wt_sequence",
    (
        "Wide target sequence (Total 47 bps = 4 bp neighboring sequence + 20 bp "
        "protospacer + 3 bp NGG PAM+ 20 bp neighboring sequence)"
    ),
)
_DEEPPE_MUT_COLUMNS = (
    "mut_sequence",
    (
        "Prime edited sequence (input for deep learning, A/C/G/T indicates 3' "
        "extension (RT template-PBS) binding region)"
    ),
)
_DEEPPE_EXT_COLUMNS = ("pegRNA_3extension", "3' extension sequence of pegRNA")
_DEEPPE_PBSLEN_COLUMNS = ("pbslen", "PBS length", "PBS length (nt)")
_DEEPPE_RTLEN_COLUMNS = ("rtlen", "RT length", "RT template length (nt)")
_DEEPPE_RTPBSLEN_COLUMNS = ("rt-pbslen", "PBS-RT length")
_DEEPPE_SPCAS9_COLUMNS = ("deepspcas9_score", "DeepSpCas9 score")
_DEEPPE_EFF_COLUMNS = ("measured_pe_efficiency", "editing_efficiency", "Measured PE efficiency")
_DEEPPE_SPLIT_COLUMNS = ("dataset_split", "Datat set name")


def _deeppe_pick_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(f"DeepPE export is missing expected column (tried {candidates})")


def _deeppe_mask_bounds(mut_sequence: str) -> tuple[int, int]:
    """Return (left, right) indices of the non-masked binding region in mut_sequence."""
    mut_upper = str(mut_sequence).upper()
    non_masked = [index for index, base in enumerate(mut_upper) if base != "X"]
    if not non_masked:
        raise ValueError("DeepPE mut_sequence has no unmasked binding region.")
    return non_masked[0], non_masked[-1] + 1


def _deeppe_build_mut_sequence(
    wt_sequence: str,
    extension: str,
    pbs_rt_len: int,
    pbs_len: int,
) -> str:
    """
    Build an X-masked prime-edited target sequence from the wide target and pegRNA 3' extension.

    The extension is treated as RNA 5'→3' (PBS then RT template); its reverse complement
    is aligned to the 47 bp wide target by maximizing PBS identity.
    """
    wt = str(wt_sequence).upper()
    extension_dna = reverse_complement(
        str(extension).upper().replace("U", "T")[:pbs_rt_len],
        mode="rna_to_dna",
    )
    pbs = extension_dna[:pbs_len]
    best_score, best_start = -1, 0
    for start in range(0, len(wt) - pbs_rt_len + 1):
        score = sum(
            left == right for left, right in zip(wt[start : start + pbs_len], pbs)
        )
        if score > best_score:
            best_score, best_start = score, start
    return (
        "X" * best_start
        + extension_dna
        + "X" * (len(wt) - best_start - pbs_rt_len)
    )


def _deeppe_split_to_fold(split_name: str) -> int:
    """Map DeepPE supplementary split labels to fold ids (-1 for held-out test)."""
    label = str(split_name).strip().lower()
    if "test" in label:
        return -1
    if "ht" in label:
        return 0
    if "type" in label:
        return 1
    if "position" in label:
        return 2
    return 0


def _deeppe_infer_rt_edit(
    wt_sequence: str,
    mut_sequence: str,
    pbs_len: int,
    pbs_rt_len: int,
) -> tuple[str, int, int]:
    """
    Infer substitution/insertion/deletion type and size within the RT template region.

    Returns:
        edit kind ('sub', 'ins', or 'del'), edit length, and 0-based edit start in RT.
    """
    pbs_left, _ = _deeppe_mask_bounds(mut_sequence)
    pbs_right = pbs_left + pbs_len
    rtt_right = pbs_left + pbs_rt_len
    wt_rt = str(wt_sequence)[pbs_right:rtt_right]
    mut_rt = str(mut_sequence).upper()[pbs_right:rtt_right]
    matcher = SequenceMatcher(None, wt_rt, mut_rt)
    insertions = deletions = substitutions = 0
    edit_start: Optional[int] = None
    for tag, wt_start, wt_end, mut_start, mut_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        if edit_start is None:
            edit_start = wt_start
        if tag == "insert":
            insertions += mut_end - mut_start
        elif tag == "delete":
            deletions += wt_end - wt_start
        elif tag == "replace":
            substitutions += max(wt_end - wt_start, mut_end - mut_start)
    if insertions and not deletions and not substitutions:
        return "ins", insertions, edit_start or 0
    if deletions and not insertions and not substitutions:
        return "del", deletions, edit_start or 0
    if substitutions and not insertions and not deletions:
        return "sub", substitutions, edit_start or 0
    dominant = max(insertions, deletions, substitutions)
    if dominant == insertions:
        return "ins", insertions, edit_start or 0
    if dominant == deletions:
        return "del", deletions, edit_start or 0
    return "sub", substitutions, edit_start or 0


def _prepare_deeppe_export_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize exported DeepPE columns to the DeepPrime-style schema for standardization."""
    source = df.copy()
    wide_col = _deeppe_pick_column(source, _DEEPPE_WIDE_COLUMNS)
    pbs_col = _deeppe_pick_column(source, _DEEPPE_PBSLEN_COLUMNS)
    rt_col = _deeppe_pick_column(source, _DEEPPE_RTLEN_COLUMNS)
    rt_pbs_col = next((name for name in _DEEPPE_RTPBSLEN_COLUMNS if name in source.columns), None)
    spcas9_col = next((name for name in _DEEPPE_SPCAS9_COLUMNS if name in source.columns), None)
    eff_col = _deeppe_pick_column(source, _DEEPPE_EFF_COLUMNS)

    prepared = pd.DataFrame()
    prepared["wt_sequence"] = source[wide_col].astype(str).str.upper()
    prepared["pbslen"] = pd.to_numeric(source[pbs_col], errors="raise").astype(int)
    prepared["rtlen"] = pd.to_numeric(source[rt_col], errors="raise").astype(int)
    if rt_pbs_col is not None:
        rt_pbs_lengths = pd.to_numeric(source[rt_pbs_col], errors="raise").astype(int)
    else:
        rt_pbs_lengths = prepared["pbslen"] + prepared["rtlen"]
    prepared["rt-pbslen"] = rt_pbs_lengths
    if any(name in source.columns for name in _DEEPPE_MUT_COLUMNS):
        mut_col = _deeppe_pick_column(source, _DEEPPE_MUT_COLUMNS)
        prepared["mut_sequence"] = source[mut_col].astype(str)
    else:
        ext_col = _deeppe_pick_column(source, _DEEPPE_EXT_COLUMNS)
        prepared["mut_sequence"] = [
            _deeppe_build_mut_sequence(
                wt,
                ext,
                int(pbs_rt_len),
                int(pbs_len),
            )
            for wt, ext, pbs_rt_len, pbs_len in zip(
                prepared["wt_sequence"],
                source[ext_col],
                rt_pbs_lengths,
                prepared["pbslen"],
            )
        ]
    if spcas9_col is not None:
        prepared["deepspcas9_score"] = pd.to_numeric(source[spcas9_col], errors="coerce")
    else:
        prepared["deepspcas9_score"] = np.nan
    prepared["measured_pe_efficiency"] = pd.to_numeric(source[eff_col], errors="coerce")

    split_col = next((name for name in _DEEPPE_SPLIT_COLUMNS if name in source.columns), None)
    if split_col is not None:
        prepared["fold"] = source[split_col].map(_deeppe_split_to_fold)
    else:
        prepared["fold"] = np.nan

    mutation_rows = [
        _deeppe_infer_rt_edit(
            wt,
            mut,
            int(pbs_len),
            int(pbs_rt_len),
        )
        for wt, mut, pbs_len, pbs_rt_len in zip(
            prepared["wt_sequence"],
            prepared["mut_sequence"],
            prepared["pbslen"],
            prepared["rt-pbslen"],
        )
    ]
    prepared["type_sub"] = [kind == "sub" for kind, _, _ in mutation_rows]
    prepared["type_ins"] = [kind == "ins" for kind, _, _ in mutation_rows]
    prepared["type_del"] = [kind == "del" for kind, _, _ in mutation_rows]
    prepared["edit_len"] = [edit_len for _, edit_len, _ in mutation_rows]
    prepared["rha_len"] = [
        max(int(rt_len) - int(edit_pos) - int(edit_len), 0)
        for (_, edit_len, edit_pos), rt_len in zip(
            mutation_rows,
            prepared["rtlen"],
        )
    ]
    return prepared


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


def _standardize_deeppe_ontarget(
    data: Optional[pd.DataFrame],
    cell_line: str,
    pe_system: str,
    dataset: str,
) -> None:
    """
    Standardize DeepPE on-target datasets to the shared PE schema.

    DeepPE supplementary tables use 47 bp wide-target reporters. Library 1 includes
    a masked prime-edited sequence; libraries 2 and endogenous validation rebuild
    that sequence from the pegRNA 3' extension before applying the DeepPrime layout
    rules (protospacer at positions 4–24).
    """
    dataset = _normalize_name(dataset)
    cell_line = _normalize_name(cell_line)
    pe_system = _normalize_name(pe_system)
    input_name = f"{cell_line}-{pe_system}.csv"
    if data is None:
        data = pd.read_csv(DATA_ROOT / "exported" / "deeppe" / dataset / input_name)
    prepared = _prepare_deeppe_export_df(data)
    endo_coords = None
    if dataset == "deeppe_endo":
        # Map using the pre-alignment 47 bp wide target (alignment may inject Ns).
        endo_coords = _deeppe_endo_coordinates(prepared["wt_sequence"])
    _standardize_deepprime_ontarget(
        prepared,
        cell_line,
        pe_system,
        dataset,
        study_key="deeppe",
    )
    if endo_coords is not None:
        output_path = (
            DATA_ROOT / "standardized" / "deeppe" / dataset / f"{cell_line}-{pe_system}.parquet"
        )
        output_df = pd.read_parquet(output_path)
        output_df = _attach_endo_coordinate_columns(output_df, endo_coords)
        output_df.to_parquet(output_path, index=False)
        logger.info("Attached DeepPE endogenous coordinates: %s", output_path)


def _scaffold_assignments(data_root=None):
    from ..catalog.datasheets import build_deeppe_scaffold_assignments
    return build_deeppe_scaffold_assignments()


register_study(StudyPipeline(
    key="deeppe",
    exporters=(_export_deeppe_datasheets,),
    standardizers={
        "deeppe_ht": _standardize_deeppe_ontarget,
        "deeppe_position": _standardize_deeppe_ontarget,
        "deeppe_type": _standardize_deeppe_ontarget,
        "deeppe_endo": _standardize_deeppe_ontarget,
    },
    scaffold_assignments=_scaffold_assignments,
))
