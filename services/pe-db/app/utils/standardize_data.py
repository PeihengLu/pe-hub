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
from pe_common.sequence_utils import align_wt_mut_sequences, reverse_complement

from .deepspcas9 import fill_missing_spcas9_scores
from ..catalog.scaffolds import SCAFFOLD_MINSEPIE_CODON_VARIANT
from ..catalog.studies import get_dataset_record

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SUPPORTED_STUDIES = {"deeppe", "deepprime", "pridict2", "pridict1", "minsepie"}
PARTIAL_STANDARDIZABLE_DATASETS: set[tuple[str, str]] = {
    ("pridict1", "endogenous"),
    ("pridict2", "trip_analysis"),
    ("deepprime", "deepprime_off"),
    ("deepprime", "deepprime_off_subpool"),
}

DATASETS = {
    "deeppe": [
        "deeppe_ht",
        "deeppe_position",
        "deeppe_type",
        "deeppe_endo",
    ],
    "deepprime": [
        "deepprime_clinvar",
        "deepprime_small",
        "deepprime_off",
        "deepprime_off_subpool",
    ],
    "pridict2": [
        "library_diverse",
        "library_diverse_invivo",
        "trip_analysis",
    ],
    "pridict1": [
        "library1",
        "library2",
        "library2_invivo",
        "endogenous",
    ],
    "minsepie": [
        "library_insert_set12",
        "library_insert_18nt",
        "library_insert_codon_variant",
        "library_insert_codon_hek3",
        "library_insert_piggybac",
    ],
}


def _normalize_name(value: str) -> str:
    """Normalize dataset/cell line/PE system names for filenames."""
    return str(value).strip().lower().replace("-", "_")


def is_standardizable(study: str, dataset: str) -> bool:
    """Return True when catalog marks the dataset as standardizable."""
    record = get_dataset_record(study, dataset)
    return bool(record and record.standardizable)


def is_partially_standardizable(study: str, dataset: str) -> bool:
    """Return True for datasets with minimal entry-level standardization support."""
    return (_normalize_name(study), _normalize_name(dataset)) in PARTIAL_STANDARDIZABLE_DATASETS


def export_original_data(study: Optional[str] = None, force_reexport: bool = False) -> None:
    """
    Export raw study files into standardized exported datasheets.

    Output naming convention:
    ``{study}/{dataset}/{cell_line}-{pe_system}.csv``

    After export, updates Datasheet rows in the catalog DB from ``datasets/exported/``.

    Args:
        study: Study key. If None, export all supported studies.
        force_reexport: If True, re-export all data even if it already exists.
    """
    exporters = {
        "deeppe": [_export_deeppe_datasheets],
        "deepprime": [_export_deepprime_datasheets],
        "pridict1": [
            _export_pridict1_datasheets,
            _export_pridict1_library2_datasheets,
            _export_pridict1_endogenous_datasheets,
        ],
        "pridict2": [_export_pridict2_library_diverse_datasheets, _export_pridict2_endogenous_datasheets],
        "minsepie": [_export_minsepie_datasheets],
    }

    target_studies = sorted(exporters) if study is None else [study]
    for study_name in target_studies:
        if study_name not in exporters:
            raise ValueError(f"Study={study_name} not supported")
        exported_marker = DATA_ROOT / "exported" / study_name
        if force_reexport or not exported_marker.exists():
            logger.info("Exporting study=%s", study_name)
            for exporter in exporters[study_name]:
                exporter()
        else:
            logger.info("Study=%s already exported", study_name)

    from ..catalog.datasheets import index_exported_datasheets

    index_exported_datasheets()


def standardize_exported_data(
    study: Optional[str] = None,
    *,
    force: bool = False,
) -> int:
    """
    Standardize all exported CSV datasheets under ``datasets/exported/``.

    Writes parquet files to ``datasets/standardized/{study}/{dataset}/``.

    Returns the number of datasheets standardized (or skipped when already present).
    """
    exported_root = DATA_ROOT / "exported"
    if not exported_root.exists():
        logger.warning("No exported data directory at %s", exported_root)
        return 0

    studies = sorted(SUPPORTED_STUDIES) if study is None else [study.strip().lower()]
    count = 0
    for study_key in studies:
        study_dir = exported_root / study_key
        if not study_dir.is_dir():
            continue
        for csv_path in study_dir.rglob("*.csv"):
            if not csv_path.is_file():
                continue
            rel_parts = csv_path.relative_to(study_dir).parts
            if len(rel_parts) != 2 or "-" not in csv_path.stem:
                continue
            dataset_name, _filename = rel_parts
            cell_line, pe_system = csv_path.stem.rsplit("-", 1)
            normalized_dataset = _normalize_name(dataset_name)
            output_path = (
                DATA_ROOT / "standardized" / study_key / normalized_dataset /
                f"{_normalize_name(cell_line)}-{_normalize_name(pe_system)}.parquet"
            )
            if not (
                is_standardizable(study_key, dataset_name)
                or is_partially_standardizable(study_key, dataset_name)
            ):
                logger.debug(
                    "Skipping standardization for %s/%s (export-only or unsupported schema)",
                    study_key,
                    dataset_name,
                )
                continue
            if output_path.exists() and not force:
                logger.debug("Already standardized: %s", output_path)
                continue
            logger.info(
                "Standardizing %s/%s %s-%s",
                study_key,
                dataset_name,
                cell_line,
                pe_system,
            )
            try:
                standardize_pe_data(
                    study=study_key,
                    dataset=dataset_name,
                    cell_line=cell_line,
                    pe_system=pe_system,
                )
            except Exception as exc:
                logger.error(
                    "Failed to standardize %s/%s %s-%s: %s",
                    study_key,
                    dataset_name,
                    cell_line,
                    pe_system,
                    exc,
                )
                continue
            count += 1
    logger.info("Standardized %s exported datasheet(s)", count)
    return count

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


def _export_deepprime_datasheets() -> None:
    """
    Export all sheets from the original DeepPrime Excel file to CSV format.
    
    Args:
        original_excel_path: Path to the original DeepPrime Excel file.
                            If None, uses default path in raw/deepprime-org/
    """
    original_excel_path = DATA_ROOT / "raw" / "deepprime" / "deepprime-org.xlsx"
    # Summary sheet is the catalog for the individual experiment sheets.
    original_data_catalog = pd.read_excel(
        original_excel_path, sheet_name="Summary", header=0,
    )
    original_data_catalog.rename(columns={"Index": "Sheet name"}, inplace=True)

    logger.info("Processing %s DeepPrime sheets", len(original_data_catalog))

    dataset_renames = {
        "Library-ClinVar": "deepprime-clinvar",
        "Library-Small": "deepprime-small",
        "Library-Off": "deepprime-off",
        "Library-Off(sub-pool)": "deepprime-off-subpool",
    }

    def _read_from_deepprime_org(excel_path: Path, sheet_name: str = "1") -> pd.DataFrame:
        """
        Read from the DeepPrime original excel file
        
        Args:
            excel_path: Path to the original deep prime Excel file
            sheet_name: Sheet name to read from
            
        Returns:
            DataFrame with cleaned column names
        """
        # Skip metadata rows and use the 4th row as the header.
        original_data = pd.read_excel(excel_path, sheet_name=sheet_name, skiprows=3, header=0)

        # Normalize column names for downstream usage.
        original_data.columns = (original_data.columns
                                .str.replace(" ", "_")
                                .str.replace("\n", "")
                                .str.replace("\t", ""))

        original_data.columns = original_data.columns.str.lower()

        # Rename long opaque columns used in sequence processing.
        original_data.rename(columns={
            "wide_target_sequence(target_74bps_=_4bp_neighboring_sequence_+_20_bp_protospacer_+_3_bp_ngg_+_47_bp_neighboring_sequence)": "wt_sequence",
            "edited_target_sequence(target_74bps_=_rt-pbs_corresponding_region_and_masked_by_'x')": "mut_sequence",
        }, inplace=True)

        return original_data

    for _, row in original_data_catalog.iterrows():
        sheet_name = str(row["Sheet name"])
        cell_line = _normalize_name(str(row["Cell line"]))
        pe_system = _normalize_name(str(row["PE system"]))
        dataset = str(row["Library"])
        if dataset in dataset_renames:
            dataset = dataset_renames[dataset]

        logger.debug(
            "DeepPrime sheet=%s dataset=%s cell_line=%s pe_system=%s",
            sheet_name,
            dataset,
            cell_line,
            pe_system,
        )

        original_data = _read_from_deepprime_org(
            original_excel_path, sheet_name=sheet_name
        )

        output_path = (
            DATA_ROOT / "exported" / "deepprime" / dataset / f"{cell_line}-{pe_system}.csv"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        original_data.to_csv(output_path, index=False)
        logger.info("Saved DeepPrime datasheet: %s", output_path)

def _export_pridict1_datasheets() -> None:
    """
    Export the PRIDICT1 datasheets
    """
    # starting from restoring the pridict1 data into one file
    # Load the three parts of the PRIDICT data
    part1 = pd.read_csv(DATA_ROOT / 'raw' / 'pridict1' / 'pridict1_library1_part1.csv')
    part2 = pd.read_csv(DATA_ROOT / 'raw' / 'pridict1' / 'pridict1_library1_part2.csv')
    part3 = pd.read_csv(DATA_ROOT / 'raw' / 'pridict1' / 'pridict1_library1_part3.csv')

    # Concatenate the parts back into a single DataFrame
    restored_data = pd.concat([part1, part2, part3], ignore_index=True)
    library1_pe_system = 'pe2'
    library1_cell_line = 'hek293t'

    # Save the restored data as one exported datasheet for standardization.
    output_path = (
        DATA_ROOT / 'exported' / 'pridict1' / 'library1' /
        f'{library1_cell_line}-{library1_pe_system}.csv'
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    restored_data.to_csv(output_path, index=False)
    logger.info(f"Saved restored PRIDICT1 library1 data to {output_path}")


# (dataset folder, cell_line, pe_system, source efficiency column)
_PRIDICT1_LIBRARY2_EXPORTS: tuple[tuple[str, str, str, str], ...] = (
    ("library2", "hek293t", "pe2", "HEKOpti-Scaffold_PE2_averageedited"),
    ("library2", "hek293tmlh1dn", "pe2", "HEKOpti-Scaffold_PE2-dnMLH1_averageedited"),
    ("library2", "u2os", "pe2", "U2OS_PE2_averageedited"),
    ("library2", "u2osmlh1dn", "pe2", "U2OS_PE2-dnMLH1_averageedited"),
    ("library2", "u2os", "pemax", "U2OS_Pemax_averageedited"),
    ("library2", "u2osmlh1dn", "pemax", "U2OS_Pemax-dnMLH1_averageedited"),
    ("library2", "k562", "pe2", "K562_PE2_averageedited"),
    ("library2", "k562mlh1dn", "pe2", "K562_PE2-dnMLH1_averageedited"),
    ("library2", "k562", "pemax", "K562_Pemax_averageedited"),
    ("library2", "k562mlh1dn", "pemax", "K562_Pemax-dnMLH1_averageedited"),
    ("library2-invivo", "liver_gfpplus", "pe2", "Liver-GFPplus_PE2Adeno_averageedited"),
)


def _export_pridict1_library2_datasheets() -> None:
    """
    Export PRIDICT1 disease-focused subscreen (``pridict_library2.csv``).

    Source: supplementary disease-block subscreen (~1.9k pegRNAs) with editing
    measured across HEK293T, U2OS, and K562 in vitro (PE2 / PEmax; MLH1−/− as
    separate cell lines), plus GFP+ mouse liver in vivo (library2-invivo).
    """
    source_path = DATA_ROOT / "raw" / "pridict1" / "pridict_library2.csv"
    df = pd.read_csv(source_path)
    efficiency_columns = [col for col in df.columns if col.endswith("averageedited")]

    for dataset_name, cell_line, pe_system, efficiency_col in _PRIDICT1_LIBRARY2_EXPORTS:
        if efficiency_col not in df.columns:
            logger.warning(
                "PRIDICT1 library2 missing column %s; skipping export", efficiency_col
            )
            continue

        export_df = df.dropna(subset=[efficiency_col]).copy()
        other_efficiency_cols = [col for col in efficiency_columns if col != efficiency_col]
        export_df = export_df.drop(columns=other_efficiency_cols, errors="ignore")
        export_df = export_df.rename(columns={efficiency_col: "averageedited"})

        output_path = (
            DATA_ROOT
            / "exported"
            / "pridict1"
            / dataset_name
            / f"{cell_line}-{pe_system}.csv"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_df.to_csv(output_path, index=False)
        logger.info(
            "Saved PRIDICT1 %s data (%s-%s): %s (%s rows)",
            dataset_name,
            cell_line,
            pe_system,
            output_path,
            len(export_df),
        )


_PRIDICT1_ENDOGENOUS_METADATA_PATH = (
    DATA_ROOT / "raw" / "pridict1" / "pridict1_endogenous_edit_metadata.csv"
)


def _assign_pridict1_endogenous_peg_roles(df: pd.DataFrame) -> pd.Series:
    """Map seq_id order within locus to library best / endo / worst pegRNA roles."""
    if not {"seq_id", "locus"}.issubset(df.columns):
        return pd.Series(pd.NA, index=df.index, dtype="string")

    seq_num = (
        df["seq_id"]
        .astype("string")
        .str.extract(r"(\d+)", expand=False)
        .astype("Int64")
    )
    rank_in_locus = seq_num.groupby(df["locus"]).rank(method="first")
    return rank_in_locus.map({1: "best", 2: "endo", 3: "worst"}).astype("string")


def _load_pridict1_endogenous_edit_metadata() -> pd.DataFrame:
    """
    Per-pegRNA correction metadata derived from PRIDICT supplementary CRISPResso
    batch amplicons (uzh-dqbm-cmi/PRIDICT supplementary_files).
    """
    if not _PRIDICT1_ENDOGENOUS_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing PRIDICT1 endogenous metadata: {_PRIDICT1_ENDOGENOUS_METADATA_PATH}"
        )
    meta = pd.read_csv(_PRIDICT1_ENDOGENOUS_METADATA_PATH, dtype={"locus": "string"})
    meta["peg_role"] = meta["peg_role"].astype("string").str.strip().str.lower()
    meta["correction_type"] = (
        meta["correction_type"].astype("string").str.strip().str.lower()
    )
    return meta


def _attach_pridict1_endogenous_correction_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Join Correction_Type / Correction_Length and boolean type flags onto export rows."""
    out = df.copy()
    out["peg_role"] = _assign_pridict1_endogenous_peg_roles(out)
    meta = _load_pridict1_endogenous_edit_metadata()
    out["locus_key"] = out["locus"].astype("string")
    merged = out.merge(
        meta,
        left_on=["locus_key", "peg_role"],
        right_on=["locus", "peg_role"],
        how="left",
        suffixes=("", "_meta"),
    )
    if merged["correction_type"].isna().any():
        missing = merged.loc[merged["correction_type"].isna(), ["seq_id", "locus", "peg_role"]]
        raise ValueError(
            "PRIDICT1 endogenous rows missing correction metadata:\n"
            f"{missing.head().to_string()}"
        )

    correction_type = merged["correction_type"]
    merged["Correction_Type"] = correction_type.str.capitalize()
    merged["Correction_Length"] = pd.to_numeric(
        merged["correction_length"], errors="raise"
    )
    merged["type_sub"] = correction_type.eq("replacement")
    merged["type_ins"] = correction_type.eq("insertion")
    merged["type_del"] = correction_type.eq("deletion")
    return merged.drop(
        columns=["locus_key", "locus_meta", "correction_type", "correction_length"],
        errors="ignore",
    )


def _export_pridict1_endogenous_datasheets() -> None:
    """
    Export PRIDICT1 endogenous-locus validation (Mathis et al., Nat. Biotechnol. 2023).

    Source: ``raw/pridict1/pridict_endogenous.csv`` — arrayed pegRNAs at native
    genomic loci with measured editing in HEK293T and K562 (PE2 conditions in the
    original study). Not part of the lentiviral self-targeting HTS library.

    Edit types come from ``pridict1_endogenous_edit_metadata.csv`` (amplicon vs HDR
    in the authors' endogenous CRISPResso batch). Rows are keyed by locus and peg
    role (seq_id rank: best, endo, worst library pegRNA per site).
    """
    source_path = DATA_ROOT / "raw" / "pridict1" / "pridict_endogenous.csv"
    df = pd.read_csv(source_path)
    base_df = _attach_pridict1_endogenous_correction_fields(df)

    # Mathis et al. 2023 endogenous validation: PE2 + plasmid transfection in each line.
    exports = (
        ("hek293t", "pe2", "HEK293T_averageedited"),
        ("k562", "pe2", "K562_averageedited"),
    )
    for cell_line, pe_system, efficiency_col in exports:
        # Keep only the target cell line's primary endogenous efficiency column.
        export_df = base_df.copy()
        other_efficiency_cols = [
            col
            for col in export_df.columns
            if col.endswith("_averageedited") and col != efficiency_col
        ]
        export_df = export_df.drop(columns=other_efficiency_cols, errors="ignore")

        out_dir = DATA_ROOT / "exported" / "pridict1" / "endogenous"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{cell_line}-{pe_system}.csv"
        export_df.to_csv(output_path, index=False)
        logger.info(
            "Saved PRIDICT1 endogenous data (%s, kept primary column %s): %s (%s rows)",
            cell_line,
            efficiency_col,
            output_path,
            len(export_df),
        )


def _export_pridict2_library_diverse_datasheets() -> None:
    """
    Export the PRIDICT2 Library Diverse datasheet (supplementary table 2).
    """
    original_excel_path = DATA_ROOT / "raw" / "pridict2" / "pridict2-org.xlsx"
    # Sheet 2 contains library diverse
    library_diverse_data = pd.read_excel(original_excel_path, sheet_name=2, header=0)
    efficiency_columns = [col for col in library_diverse_data.columns if 'averageedited' in col]
    # library diverse contains editing data for four cell lines: HEK, K562, K562MLH1dn, AdV
    # each cell line has a separate sheet
    for cell_line in ['HEK', 'K562', 'K562MLH1dn', 'AdV']:
        efficiency_column = f'{cell_line}averageedited'
        data = library_diverse_data[efficiency_column]
        # concatenate all columns not containing any efficiency column
        information_columns = [col for col in library_diverse_data.columns if col not in efficiency_columns]
        data = pd.concat([data] + [library_diverse_data[col] for col in information_columns], axis=1)

        # remove columns where editing efficiency is NaN
        data = data.dropna(subset=[efficiency_column])

        cell_line_norm = cell_line.lower().replace("-", "_")
        if cell_line == 'AdV':
            dataset_name = 'library-diverse-invivo'
            output_path = (
                DATA_ROOT / 'exported' / 'pridict2' / dataset_name /
                f'{cell_line_norm}-pe2.csv'
            )
        else:
            dataset_name = 'library-diverse'
            output_path = (
                DATA_ROOT / 'exported' / 'pridict2' / dataset_name /
                f'{cell_line_norm}-pe2.csv'
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path, index=False)
        logger.info(f"Saved PRIDICT2 library diverse data for {cell_line} to {output_path}")


def _export_pridict2_endogenous_datasheets() -> None:
    """
    Export PRIDICT2 TRIP endogenous editing survey (Mathis et al., Nat. Biotechnol. 2024).

    Source: ``pridict2-org.xlsx`` supplementary table 12 — TRIP library editing
    results at endogenous genomic sites (Fig. 3, Ext. Fig. 4–5). Per the publication
    and ePRIDICT training data, TRIP was performed in K562 with prime editor PE2.
    """
    original_excel_path = DATA_ROOT / "raw" / "pridict2" / "pridict2-org.xlsx"
    trip_df = pd.read_excel(original_excel_path, sheet_name="12", header=0)

    cell_line = "k562"
    pe_system = "pe2"
    out_dir = DATA_ROOT / "exported" / "pridict2" / "trip_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{cell_line}-{pe_system}.csv"
    trip_df.to_csv(output_path, index=False)
    logger.info(
        "Saved PRIDICT2 TRIP analysis (%s-%s): %s (%s rows)",
        cell_line,
        pe_system,
        output_path,
        len(trip_df),
    )


# ------------------------------------------------------------------------------
# MinSePIE constants
# ------------------------------------------------------------------------------
# Mapping from experiment name (in MOESM5 ``experiment`` column) to the ST6
# pegRNA row identified by (target, purpose). See Koeppel et al., Nature
# Biotechnology 2023, Supplementary Tables 4-6 for details.
_MINSEPIE_EXPERIMENT_TO_PEGRNA: dict[str, tuple[str, str]] = {
    # Main prime insertion screen (+1 insertions at nick)
    "CLYBL_293T_PE2_3": ("CLYBL", "prime insertion screen"),
    "CLYBL_293T_PE2_6": ("CLYBL", "prime insertion screen"),
    "EMX1_293T_PE2_3": ("EMX1", "prime insertion screen"),
    "EMX1_293T_PE2_6": ("EMX1", "prime insertion screen"),
    "FANCF_293T_PE2_3": ("FANCF", "prime insertion screen"),
    "FANCF_293T_PE2_6": ("FANCF", "prime insertion screen"),
    "FANCF_HAP1_PBPE_8": ("FANCF", "prime insertion screen"),
    "FANCF_HAP1dMLH1_PBPE_7": ("FANCF", "prime insertion screen"),
    "HEK3_293T_PE2_1": ("HEK3", "prime insertion screen"),
    "HEK3_293T_PE2_6": ("HEK3", "prime insertion screen"),
    "HEK3_293T_PE2_EV": ("HEK3", "prime insertion screen"),
    "HEK3_293T_PE2_FEN1": ("HEK3", "prime insertion screen"),
    "HEK3_293T_PE2_TREX1": ("HEK3", "prime insertion screen"),
    "HEK3_293T_PE2_TREX2": ("HEK3", "prime insertion screen"),
    "HEK3_HAP1_PBPE_8": ("HEK3", "prime insertion screen"),
    "HEK3_HAP1dMLH1_PBPE_7": ("HEK3", "prime insertion screen"),
    # HEK3 codon-validation barnacle: spacer/PBS/HA from main HEK3 pegRNA (+1);
    # ordered oligos used the codon-variant scaffold (applied in export enrichment).
    "HEK3_293T_barnacle": ("HEK3", "prime insertion screen"),
    # Engineered epegRNA
    "HEK3_293T_epeg_10": ("HEK3", "prime insertion screen epegRNA"),
    # 18nt insertion library (HEK3 main site labelled HEK3-S1 in experiment names)
    "HEK3-S1_293T_5": ("HEK3", "prime insertion screen with 18 nt insertions"),
    "HEK3-S2_293T_5": ("HEK3-S2", "prime insertion screen with 18 nt insertions"),
    "HEK3-S3_293T_5": ("HEK3-S3", "prime insertion screen with 18 nt insertions"),
    "HEK3-S4_293T_5": ("HEK3-S4", "prime insertion screen with 18 nt insertions"),
    "HEK3-S5_293T_5": ("HEK3-S5", "prime insertion screen with 18 nt insertions"),
    "HEK3-S6_293T_5": ("HEK3-S6", "prime insertion screen with 18 nt insertions"),
    # Codon-variant (barnacle) library at endogenous loci
    "LMNB1_293T_barnacle": ("LMNB1", "prime insertion screen with codon variants"),
    "LMNB1rc_293T_barnacle": ("LMNB1_rc", "prime insertion screen with codon variants"),
    "ACTB_293T_barnacle": ("ACTB", "prime insertion screen with codon variants"),
    "ACTBrc_293T_barnacle": ("ACTB_rc", "prime insertion screen with codon variants"),
    "NOLC1_293T_barnacle": ("NOLC1", "prime insertion screen with codon variants"),
    "TP53_293T_barnacle": ("TP53", "prime insertion screen with codon variants"),
    "TP53rc_293T_barnacle": ("TP53_rc", "prime insertion screen with codon variants"),
    # RNF1 in experiment names corresponds to RNF2 in ST6 (author typo).
    "RNF1_293T_barnacle": ("RNF2", "prime insertion screen with codon variants"),
}

# Tokens in MOESM5 ``experiment`` that separate target from delivery/condition.
_MINSEPIE_EXPERIMENT_CELL_TOKENS: frozenset[str] = frozenset({"293T", "HAP1", "HAP1dMLH1"})

# MOESM5 experiments delivered via PiggyBac-integrated PE2 (HAP1 screens).
_MINSEPIE_PIGGYBAC_EXPERIMENTS: frozenset[str] = frozenset({
    "FANCF_HAP1_PBPE_8",
    "FANCF_HAP1dMLH1_PBPE_7",
    "HEK3_HAP1_PBPE_8",
    "HEK3_HAP1dMLH1_PBPE_7",
})

_MINSEPIE_GENOMIC_LOCI_PATH = DATA_ROOT / "raw" / "minsepie" / "minsepie_genomic_loci.json"
_MINSEPIE_WIDE_FLANK_BP = 100


@lru_cache(maxsize=1)
def _load_minsepie_genomic_loci() -> dict[str, Any]:
    """Load hg38 spacer anchors and cached 220 bp reference windows (100 bp flanks)."""
    if not _MINSEPIE_GENOMIC_LOCI_PATH.exists():
        raise FileNotFoundError(
            f"Missing MinSePIE genomic loci metadata: {_MINSEPIE_GENOMIC_LOCI_PATH}"
        )
    with _MINSEPIE_GENOMIC_LOCI_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _minsepie_genomic_target_key(experiment: str) -> str:
    """Return the ST6 ``target`` name for a MOESM5 experiment (e.g. HEK3, HEK3-S2)."""
    return _MINSEPIE_EXPERIMENT_TO_PEGRNA[experiment][0]


def _minsepie_reference_window_for_target(target_key: str) -> Optional[tuple[str, int]]:
    """Return (reference_window, spacer_offset) when hg38 coordinates are known."""
    meta = _load_minsepie_genomic_loci()
    locus = meta.get("loci", {}).get(target_key)
    if not locus or "reference_window" not in locus:
        return None
    window = str(locus["reference_window"]).upper()
    spacer_offset = int(locus["spacer_offset"])
    return window, spacer_offset


def _build_minsepie_core_target_sequences(
    spacer: str,
    ha_left: str,
    ha_right: str,
    insertion: str,
) -> tuple[str, str, int, int]:
    """Build pegRNA-derived target-strand WT/Mut without genomic flanks (legacy path)."""
    rc_ha_right = reverse_complement(ha_right)
    rc_ha_left = reverse_complement(ha_left)
    edit_len = len(insertion)
    edit_position = 17 + len(rc_ha_right)
    head = spacer[:17]
    wt = head + rc_ha_right + ("N" * edit_len) + rc_ha_left
    mut = head + rc_ha_right + insertion + rc_ha_left
    return wt, mut, edit_len, edit_position


def _build_minsepie_wide_from_reference(
    reference_window: str,
    spacer_offset: int,
    spacer: str,
    ha_left: str,
    ha_right: str,
    insertion: str,
    ins_position: int,
) -> tuple[str, str, int, int]:
    """
    Build WT/Mut on the target strand using a cached hg38 window (100 bp flanks).

    The window is oriented so ``reference_window[spacer_offset:spacer_offset+20]``
    equals the ST6 spacer. Insertions use the same nick / HA placement rules as
    the pegRNA-only path, shifted by ``spacer_offset``.
    """
    window = str(reference_window).upper()
    if window[spacer_offset : spacer_offset + 20] != spacer[:20]:
        raise ValueError(
            "MinSePIE reference window does not match pegRNA spacer at annotated offset."
        )
    rc_ha_right = reverse_complement(ha_right)
    rc_ha_left = reverse_complement(ha_left)
    edit_len = len(insertion)
    edit_position = spacer_offset + 17 + len(rc_ha_right) + max(ins_position - 1, 0)
    wt = window[:edit_position] + ("N" * edit_len) + window[edit_position:]
    mut = window[:edit_position] + insertion + window[edit_position:]
    return wt, mut, edit_len, edit_position


def _parse_minsepie_experiment(experiment: str) -> tuple[str, str, str]:
    """Parse a MinSePIE experiment string into (target_variant, cell_abbr, condition).

    Experiment strings follow ``<target>_<cell_abbr>_<condition>`` where
    ``cell_abbr`` is one of {``293T``, ``HAP1``, ``HAP1dMLH1``}. ``target`` and
    ``condition`` may themselves contain underscores (e.g. ``HEK3-S2`` or
    ``PE2_FEN1``). The condition encodes library subset or perturbation, not the
    prime editor version (see ``_minsepie_pe_system_for_experiment``).
    """
    tokens = experiment.split("_")
    for index, token in enumerate(tokens):
        if token in _MINSEPIE_EXPERIMENT_CELL_TOKENS:
            target_variant = "_".join(tokens[:index])
            condition = "_".join(tokens[index + 1:])
            return target_variant, token, condition
    raise ValueError(f"Could not parse cell line from MinSePIE experiment '{experiment}'")


MINSEPIE_LIBRARY_INSERT_SET12_DATASET = "library-insert-set12"
MINSEPIE_LIBRARY_INSERT_18NT_DATASET = "library-insert-18nt"
MINSEPIE_LIBRARY_INSERT_CODON_VARIANT_DATASET = "library-insert-codon-variant"
MINSEPIE_LIBRARY_INSERT_CODON_HEK3_DATASET = "library-insert-codon-hek3"
MINSEPIE_LIBRARY_INSERT_PIGGYBAC_DATASET = "library-insert-piggybac"

_MINSEPIE_18NT_EXPERIMENTS: frozenset[str] = frozenset({
    "HEK3-S1_293T_5",
    "HEK3-S2_293T_5",
    "HEK3-S3_293T_5",
    "HEK3-S4_293T_5",
    "HEK3-S5_293T_5",
    "HEK3-S6_293T_5",
})

_MINSEPIE_CODON_ENDOGENOUS_EXPERIMENTS: frozenset[str] = frozenset({
    "ACTB_293T_barnacle",
    "ACTBrc_293T_barnacle",
    "LMNB1_293T_barnacle",
    "LMNB1rc_293T_barnacle",
    "NOLC1_293T_barnacle",
    "TP53_293T_barnacle",
    "TP53rc_293T_barnacle",
    "RNF1_293T_barnacle",
})


def _minsepie_dataset_for_experiment(experiment: str) -> str:
    """Map each MOESM5 experiment to a catalog dataset (library type × delivery)."""
    if experiment in _MINSEPIE_PIGGYBAC_EXPERIMENTS:
        return MINSEPIE_LIBRARY_INSERT_PIGGYBAC_DATASET
    if experiment == "HEK3_293T_barnacle":
        return MINSEPIE_LIBRARY_INSERT_CODON_HEK3_DATASET
    if experiment in _MINSEPIE_CODON_ENDOGENOUS_EXPERIMENTS:
        return MINSEPIE_LIBRARY_INSERT_CODON_VARIANT_DATASET
    if experiment in _MINSEPIE_18NT_EXPERIMENTS:
        return MINSEPIE_LIBRARY_INSERT_18NT_DATASET
    return MINSEPIE_LIBRARY_INSERT_SET12_DATASET


def _minsepie_pe_system_for_experiment(experiment: str) -> str:
    """Prime editor version (Koeppel et al. 2023 MOESM5 screens only)."""
    if experiment == "HEK3_293T_epeg_10":
        return "pe2_epegrna"
    if experiment == "EMX1_293T_PE2_6":
        return "pe3"
    return "pe2"


def _minsepie_cell_line_from_raw(cell_line: str) -> str:
    """Normalize MOESM5 ``cell_line`` (HEK293T, HAP1, HAP1dMLH1, rc, …)."""
    return str(cell_line).strip().lower()


def _parse_minsepie_insertion_position(insertions_text: str) -> int:
    """Extract the ``+N`` insertion position from an ST6 ``insertions`` cell.

    Falls back to ``+1`` (the most common) when the text does not explicitly
    encode a position (e.g. the epegRNA "Structure library" entry).
    """
    import re

    match = re.search(r"\+\s*(\d+)\s*position", str(insertions_text).lower())
    if match:
        return int(match.group(1))
    return 1


def _load_minsepie_pegrna_table() -> pd.DataFrame:
    """Load and index ST6_pegRNAs from the MinSePIE supplementary workbook."""
    xlsx_path = DATA_ROOT / "raw" / "minsepie" / "41587_2023_1678_MOESM3_ESM.xlsx"
    pegrna_df = pd.read_excel(xlsx_path, sheet_name="ST6_pegRNAs", header=0)
    pegrna_df = pegrna_df.rename(
        columns={
            "ha": "pegrna_ha",
            "pbs": "pegrna_pbs",
            "spacer": "pegrna_spacer",
            "scaffold": "pegrna_scaffold_seq",
        }
    )
    pegrna_df["target"] = pegrna_df["target"].astype(str)
    pegrna_df["purpose"] = pegrna_df["purpose"].astype(str)
    return pegrna_df


def _prepare_minsepie_export_frame(data_root: Optional[Path] = None) -> pd.DataFrame:
    """Load MinSePIE MOESM4/5 tables and attach pegRNA + grouping columns."""
    root = data_root or DATA_ROOT
    raw_dir = root / "raw" / "minsepie"
    data_path = raw_dir / "41587_2023_1678_MOESM5_ESM.tsv"
    library_path = raw_dir / "41587_2023_1678_MOESM4_ESM.tsv"

    data = pd.read_csv(data_path, sep="\t")
    library = pd.read_csv(library_path, sep="\t")

    library_dedup = library.drop_duplicates(subset="name", keep="first")
    data = data.merge(library_dedup[["name", "set"]], on="name", how="left")

    pegrna_df = _load_minsepie_pegrna_table()
    pegrna_lookup: dict[tuple[str, str], pd.Series] = {
        (str(row["target"]), str(row["purpose"])): row
        for _, row in pegrna_df.iterrows()
    }

    pegrna_columns = {
        "pegrna_spacer": [],
        "pegrna_ha": [],
        "pegrna_pbs": [],
        "pegrna_ins_position": [],
        "pegrna_purpose": [],
        "pegrna_scaffold": [],
    }
    for experiment in data["experiment"].astype(str):
        key = _MINSEPIE_EXPERIMENT_TO_PEGRNA[experiment]
        pegrna_row = pegrna_lookup[key]
        pegrna_columns["pegrna_spacer"].append(str(pegrna_row["pegrna_spacer"]))
        pegrna_columns["pegrna_ha"].append(str(pegrna_row["pegrna_ha"]))
        pegrna_columns["pegrna_pbs"].append(str(pegrna_row["pegrna_pbs"]))
        pegrna_columns["pegrna_ins_position"].append(
            _parse_minsepie_insertion_position(pegrna_row["insertions"])
        )
        pegrna_columns["pegrna_purpose"].append(str(pegrna_row["purpose"]))
        pegrna_columns["pegrna_scaffold"].append(
            str(pegrna_row["pegrna_scaffold_seq"]).upper()
        )

    for column, values in pegrna_columns.items():
        data[column] = values

    # Codon-validation oligos at HEK3 use the codon-variant scaffold (Methods) while
    # spacer/PBS/HA remain the main HEK3 prime-insertion design from ST6.
    hek3_barnacle = data["experiment"].astype(str) == "HEK3_293T_barnacle"
    data.loc[hek3_barnacle, "pegrna_scaffold"] = SCAFFOLD_MINSEPIE_CODON_VARIANT

    data["_cell_line"] = data["cell_line"].map(_minsepie_cell_line_from_raw)
    data["_pe_system"] = data["experiment"].map(_minsepie_pe_system_for_experiment).map(
        _normalize_name
    )
    data["_dataset"] = data["experiment"].map(_minsepie_dataset_for_experiment)
    return data


def iter_minsepie_consolidated_datasheet_specs(
    data_root: Optional[Path] = None,
) -> list[tuple[str, str, str]]:
    """Return MinSePIE export keys: one row per ``{cell_line}-{pe_system}.csv``.

    Each tuple is ``(dataset, cell_line, pe_system)`` under ``exported/minsepie/``.
    """
    data = _prepare_minsepie_export_frame(data_root)
    specs: list[tuple[str, str, str]] = []
    for (dataset_name, cell_line, pe_system), _group_df in data.groupby(
        ["_dataset", "_cell_line", "_pe_system"], sort=False
    ):
        specs.append((dataset_name, cell_line, pe_system))
    return specs


def _export_minsepie_datasheets() -> None:
    """
    Export the MinSePIE datasheets (Koeppel et al., Nat. Biotechnol. 2023).

    MinSePIE is an insertion-only PE dataset that measures editing efficiency
    for libraries of DNA insertions at specific genomic target sites across
    different cell lines and PE system variants.

    Raw data (under ``raw/minsepie/``):
      - MOESM3 (Suppl. Table 3): Contains two sheets
          * ``ST5_gene_fragments``: reference gene fragments (e.g. ``puroR``) used in the screen
          * ``ST6_pegRNAs``: the pegRNA design for each screen (spacer / HA / PBS /
            insertion position marker). This is what we need to reconstruct
            WT/mutant target sequences and positional fields.
      - MOESM4 (Suppl. Table 4): Insert library mapping ``name`` → ``insert_sequence``
        plus a ``set`` column that identifies the library subset.
      - MOESM5 (Suppl. Table 5): Experimental results (editing efficiency per
        insert / target / condition).

    Each row in MOESM5 belongs to exactly one ``experiment`` (32 total). We
    enrich MOESM5 with pegRNA design info from ST6 and library metadata from
    MOESM4, then merge rows sharing the same library-type dataset, MOESM5
    ``cell_line``, and prime editor version into ``{cell_line}-{pe_system}.csv``
    under ``library-insert-set12``, ``library-insert-18nt``,
    ``library-insert-codon-variant``, ``library-insert-codon-hek3``, or
    ``library-insert-piggybac``.
    """
    data = _prepare_minsepie_export_frame()

    for (dataset_name, cell_line, pe_system), group_df in data.groupby(
        ["_dataset", "_cell_line", "_pe_system"], sort=False
    ):
        export_df = group_df.drop(columns=["_cell_line", "_pe_system", "_dataset"])
        dataset_root = DATA_ROOT / "exported" / "minsepie" / dataset_name
        dataset_root.mkdir(parents=True, exist_ok=True)
        output_path = dataset_root / f"{cell_line}-{pe_system}.csv"
        export_df.to_csv(output_path, index=False)
        logger.info("Saved MinSePIE datasheet: %s (%s rows)", output_path, len(export_df))

# ==============================================================================
# Standardize PE datasets into the shared schema.
# ==============================================================================
standard_pe_data_columns = [
        'group_id', 'type_sub', 'type_ins', 'type_del', 'edit_len', 
        'wt_sequence', 'mut_sequence', # sequences padded with N if necessary
        'protospacer_location_l', 'protospacer_location_r',  
        'pbs_location_l', 'pbs_location_r', 'rtt_location_l', 'rtt_location_r', 
        'lha_location_l', 'lha_location_r', 'rha_location_l', 'rha_location_r', 
        'spcas9_score', 'editing_efficiency', 'original_fold']


def _coerce_original_fold(
    original_fold: Optional[pd.Series | np.ndarray],
    *,
    length: int,
) -> pd.Series:
    """Return nullable float fold ids; all-NaN when the source has no split metadata."""
    if original_fold is None:
        return pd.Series(np.nan, index=range(length), dtype="Float64")
    return pd.to_numeric(pd.Series(original_fold, copy=False), errors="coerce").astype(
        "Float64"
    )


def _build_standardized_output_df(
    group_id: pd.Series | np.ndarray, 
    type_sub: pd.Series | np.ndarray, type_ins: pd.Series | np.ndarray, type_del: pd.Series | np.ndarray, edit_len: pd.Series | np.ndarray, 
    wt_sequence: pd.Series | np.ndarray, mut_sequence: pd.Series | np.ndarray, 
    protospacer_location_l: int | np.ndarray | pd.Series, protospacer_location_r: int | np.ndarray | pd.Series, 
    pbs_location_l: pd.Series | np.ndarray, pbs_location_r: pd.Series | np.ndarray, 
    rtt_location_l: pd.Series | np.ndarray, rtt_location_r: pd.Series | np.ndarray, 
    lha_location_l: pd.Series | np.ndarray, lha_location_r: pd.Series | np.ndarray, 
    rha_location_l: pd.Series | np.ndarray, rha_location_r: pd.Series | np.ndarray, 
    spcas9_score: pd.Series | np.ndarray, editing_efficiency: pd.Series | np.ndarray, 
    original_fold: Optional[pd.Series | np.ndarray] = None) -> pd.DataFrame:
    """
    Standardize the data types in output dataframe
    
    Args:
        group_id: Series of group IDs
        type_sub: Series of boolean type_sub values
        type_ins: Series of boolean type_ins values
        type_del: Series of boolean type_del values
        edit_len: Series of edit lengths
        wt_sequence: Series of wild type sequences
        mut_sequence: Series of mutated sequences
        protospacer_location_l: int of protospacer location left
        protospacer_location_r: int of protospacer location right
        pbs_location_l: Series of PBS location left
        pbs_location_r: Series of PBS location right
        rtt_location_l: Series of RTT location left
        rtt_location_r: Series of RTT location right
        lha_location_l: Series of LHA location left
        lha_location_r: Series of LHA location right
        rha_location_l: Series of RHA location left
        rha_location_r: Series of RHA location right
        spcas9_score: Series of spcas9 scores
        editing_efficiency: Series of editing efficiencies
        original_fold: Series of author-provided fold ids (NaN when unknown)
    Returns:
        DataFrame with correct types
    """
    n_rows = len(group_id)
    output_df = pd.DataFrame({
        'group_id': group_id,
        'type_sub': type_sub,
        'type_ins': type_ins,
        'type_del': type_del,
        'edit_len': edit_len,
        'wt_sequence': wt_sequence,
        'mut_sequence': mut_sequence,
        'protospacer_location_l': protospacer_location_l,
        'protospacer_location_r': protospacer_location_r,
        'pbs_location_l': pbs_location_l,
        'pbs_location_r': pbs_location_r,
        'rtt_location_l': rtt_location_l,
        'rtt_location_r': rtt_location_r,
        'lha_location_l': lha_location_l,
        'lha_location_r': lha_location_r,
        'rha_location_l': rha_location_l,
        'rha_location_r': rha_location_r,
        'spcas9_score': spcas9_score,
        'editing_efficiency': editing_efficiency,
        'original_fold': _coerce_original_fold(original_fold, length=n_rows),
    })

    # String transformations
    output_df['wt_sequence'] = output_df['wt_sequence'].str.upper()
    output_df['mut_sequence'] = output_df['mut_sequence'].str.upper()

    # Batch type conversions
    bool_columns = ['type_sub', 'type_ins', 'type_del']
    int_columns = [
        'group_id', 'edit_len',
        'protospacer_location_l', 'protospacer_location_r',
        'pbs_location_l', 'pbs_location_r',
        'rtt_location_l', 'rtt_location_r',
        'lha_location_l', 'lha_location_r',
        'rha_location_l', 'rha_location_r',
    ]
    float_columns = ['spcas9_score', 'editing_efficiency', 'original_fold']

    output_df[bool_columns] = output_df[bool_columns].astype(bool)
    output_df[int_columns] = output_df[int_columns].astype(int)
    output_df[float_columns] = output_df[float_columns].astype(float)

    return output_df

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

def _standardize_deepprime_ontarget(
    data: Optional[pd.DataFrame],
    cell_line: str,
    pe_system: str,
    dataset: str,
    *,
    study_key: str = "deepprime",
) -> None:
    """
    Standardize DeepPrime-style on-target datasets to the shared PE schema.

    Also used for DeepPE after ``_prepare_deeppe_export_df`` maps Kim et al. exports
    into the same masked-sequence column layout.
    """
    dataset = _normalize_name(dataset)
    cell_line = _normalize_name(cell_line)
    pe_system = _normalize_name(pe_system)
    study_key = _normalize_name(study_key)
    input_name = f"{cell_line}-{pe_system}.csv"
    output_name = f"{cell_line}-{pe_system}.parquet"
    if data is None:
        data = pd.read_csv(DATA_ROOT / 'exported' / 'deepprime' / dataset / input_name)

    logger.info(
        "Standardizing %s dataset=%s cell_line=%s pe_system=%s rows=%s",
        study_key,
        dataset,
        cell_line,
        pe_system,
        len(data),
    )
    df = data.copy()

    # ---- Step 1: Determine mutation type and filter invalid rows ----
    # Keep the one-hot booleans for output; derive an integer mut_type for internal calculations
    mutation_flags = df[['type_sub', 'type_ins', 'type_del']].fillna(False).astype(bool)
    valid_mutation_mask = pd.Series(mutation_flags.any(axis=1), index=df.index, dtype=bool)
    df = df.loc[valid_mutation_mask].reset_index(drop=True)
    mutation_flags = df[['type_sub', 'type_ins', 'type_del']].fillna(False).astype(bool)
    df[['type_sub', 'type_ins', 'type_del']] = mutation_flags
    # used for alignment, 0 for substitution, 1 for insertion, 2 for deletion
    df['mut_type'] = np.select(
        [mutation_flags['type_sub'], mutation_flags['type_ins'], mutation_flags['type_del']],
        [0, 1, 2],
        default=-1
    )

    # ---- Step 2: Compute protospacer and assign group IDs ----
    # Group rows with identical protospacers to prevent data leakage
    # deepprime sequence starts from 4bp upstream of the 20bp protospacer
    PROTOSPACER_L, PROTOSPACER_R = 4, 24  # 0-indexed
    wt_sequence = pd.Series(df['wt_sequence'], dtype='string')
    protospacer = wt_sequence.map(
        lambda seq: seq[PROTOSPACER_L:PROTOSPACER_R] if isinstance(seq, str) else ''
    )
    df['protospacer'] = protospacer
    # group rows by protospacer and save the group id
    df['group_id'] = df.groupby('protospacer').ngroup()

    # ---- Step 3: Compute PBS and LHA locations ----
    # In DeepPrime's format, mut_sequence uses leading 'x'/'X' characters to mask
    # positions upstream of the PBS that are not involved in the editing process.
    # The PBS left boundary is therefore the index of the first non-mask character.
    mut_sequence = pd.Series(df['mut_sequence'], dtype='string').fillna('')
    df['pbs_l'] = mut_sequence.map(lambda seq: len(seq) - len(seq.lstrip('xX')))
    df['pbs_r'] = df['pbs_l'] + df['pbslen']
    df['lha_l'] = df['pbs_r']
    df['lha_r'] = np.where(
        df['type_del'],
        df['pbs_r'] + (df['rt-pbslen'] - df['pbslen'] - df['rha_len']),
        df['pbs_r'] + (df['rt-pbslen'] - df['pbslen'] - df['rha_len'] - df['edit_len'])
    )

    # ---- Step 4: Compute RHA and RTT locations ----
    # The wt offset is +edit_len for deletion, -edit_len for insertion, 0 for substitution
    base = df['pbs_l'] + df['rt-pbslen']
    wt_offset = np.select(
        [df['type_del'], df['type_ins']],
        [df['edit_len'], -df['edit_len']],
        default=0
    )
    rha_wt_r = base + wt_offset

    df['rha_l'] = base - df['rha_len'] + wt_offset
    df['rha_r'] = np.where(df['type_del'], rha_wt_r, base)
    df['rtt_l'] = df['pbs_r']
    df['rtt_r'] = np.where(df['type_del'], rha_wt_r, base)

    # ---- Step 5: Reconstruct mutated sequences and align with wt ----
    # Remove the masking from the mutated sequence
    df['_rha_wt_r'] = rha_wt_r  # store intermediate for per-row string ops

    def _reconstruct_and_align(row):
        """
        Reconstruct the unmasked mutated sequence from the wild type
        and align with the wild type sequence
        """
        wt = str(row['wt_sequence'])
        lha_r = int(row['lha_r'])
        rha_wt_r_val = int(row['_rha_wt_r'])
        edit_len = int(row['edit_len'])
        type_sub = bool(row['type_sub'])
        type_ins = bool(row['type_ins'])
        type_del = bool(row['type_del'])

        # Reconstruct the unmasked mutated sequence by injecting the observed
        # RT-PBS segment from mut_sequence into WT context.
        masked_mut = str(row['mut_sequence'])
        pbs_l = int(row['pbs_l'])
        rt_pbs_len = int(row['rt-pbslen'])
        rt_pbs_right = pbs_l + rt_pbs_len
        observed_rt_pbs = masked_mut[pbs_l:rt_pbs_right].upper().replace("U", "T")

        if type_ins:
            wt_window_len = max(rt_pbs_len - edit_len, 0)
        elif type_del:
            wt_window_len = rt_pbs_len + edit_len
        else:
            wt_window_len = rt_pbs_len
        wt_suffix_start = min(max(pbs_l + wt_window_len, 0), len(wt))
        mut = wt[:pbs_l] + observed_rt_pbs + wt[wt_suffix_start:]

        # Pad with N at the edit position to align wt/mut to the same length
        mut_type = row['mut_type']
        return align_wt_mut_sequences(
                wt, mut, lha_r, edit_length=edit_len, edit_type=mut_type)

    aligned = df.apply(_reconstruct_and_align, axis=1, result_type='expand')
    aligned.columns = ['wt_aligned', 'mut_aligned']

    # ---- Step 6: Build output DataFrame ----
    if "fold" in df.columns:
        original_fold = pd.to_numeric(
            df["fold"].replace("Test", -1),
            errors="coerce",
        )
    else:
        original_fold = None
    output_df = _build_standardized_output_df(
        df['group_id'], df['type_sub'], df['type_ins'], df['type_del'], df['edit_len'], 
        aligned['wt_aligned'], aligned['mut_aligned'], PROTOSPACER_L, PROTOSPACER_R, 
        df['pbs_l'], df['pbs_r'], df['rtt_l'], df['rtt_r'], df['lha_l'], df['lha_r'], df['rha_l'], df['rha_r'], 
        df['deepspcas9_score'], df['measured_pe_efficiency'], original_fold)

    # export the data to a parquet file
    output_path = DATA_ROOT / "standardized" / study_key / dataset / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(output_path, index=False)
    logger.info("Saved standardized %s data: %s", study_key, output_path)


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
    _standardize_deepprime_ontarget(
        prepared,
        cell_line,
        pe_system,
        dataset,
        study_key="deeppe",
    )


def _standardize_pridict2_library_diverse(
        data: Optional[pd.DataFrame], cell_line: str, pe_system: str, dataset: str) -> None:
    """
    Standardize PRIDICT2 library-diverse data to the shared PE schema.
    """
    dataset = _normalize_name(dataset)
    cell_line = _normalize_name(cell_line)
    pe_system = _normalize_name(pe_system)
    input_name = f"{cell_line}-{pe_system}.csv"
    output_name = f"{cell_line}-{pe_system}.parquet"
    if data is None:
        data = pd.read_csv(DATA_ROOT / 'exported' / 'pridict2' / dataset / input_name)
    logger.info(
        "Standardizing PRIDICT2 dataset=%s cell_line=%s pe_system=%s rows=%s",
        dataset,
        cell_line,
        pe_system,
        len(data),
    )

    # ---- Step 1: calculate group ID based on the spacer column
    df = data.copy()
    df['group_id'] = df.groupby('spacer').ngroup()

    # ---- Step 2: determine mutation type and edit length ----
    correction_type = pd.Series(df['Correction_Type'], copy=False).astype('string').str.strip().str.lower()
    type_sub = correction_type.eq('replacement')
    type_ins = correction_type.eq('insertion')
    type_del = correction_type.eq('deletion')

    unknown_mask = ~(type_sub | type_ins | type_del)
    if unknown_mask.any():
        unknown_values = df.loc[unknown_mask, 'Correction_Type'].astype(str).unique().tolist()[:5]
        print(unknown_values)
        raise ValueError(f"Unsupported Correction_Type values: {unknown_values}")

    # used for alignment, 0 for substitution, 1 for insertion, 2 for deletion
    edit_type = pd.Series(
        np.select([type_sub, type_ins, type_del], [0, 1, 2], default=-1),
        index=df.index,
    ).astype(int)
    edit_len = pd.Series(pd.to_numeric(df['Correction_Length'], errors='raise'), index=df.index).astype(int)

    # ---- Step 3: compute protospacer / PBS / RTT / LHA / RHA locations ----
    wt_sequence = pd.Series(df['wide_initial_target'], copy=False).astype('string').str.upper()
    mut_sequence = pd.Series(df['wide_mutated_target'], copy=False).astype('string').str.upper()

    protospacer_l, protospacer_r = _parse_pridict_location_column(
        pd.Series(df['protospacerlocation_only_initial'], copy=False), 'protospacerlocation_only_initial'
    )
    pbs_l, pbs_r = _parse_pridict_location_column(pd.Series(df['PBSlocation'], copy=False), 'PBSlocation')
    rtt_wt_l, rtt_wt_r = _parse_pridict_location_column(
        pd.Series(df['RT_initial_location'], copy=False), 'RT_initial_location'
    )
    rtt_mut_l, rtt_mut_r = _parse_pridict_location_column(
        pd.Series(df['RT_mutated_location'], copy=False), 'RT_mutated_location'
    )

    rha_len = pd.Series(df['RTToverhang'], copy=False).astype('string').str.upper().str.len().astype(int)
    lha_len = pd.Series(np.where(
        type_del,
        rtt_mut_r - rtt_mut_l - rha_len,
        rtt_mut_r - rtt_mut_l - rha_len - edit_len,
    ), index=df.index).astype(int)
    lha_l = rtt_wt_l
    lha_r = rtt_wt_l + lha_len

    rha_wt_l = rtt_wt_r - rha_len
    rha_mut_r = rtt_mut_r

    # ---- Step 4: align WT and mut sequences on edit position ----
    aligned = pd.DataFrame(
        {
            'wt': wt_sequence,
            'mut': mut_sequence,
            'lha_r': lha_r,
            'edit_len': edit_len,
            'edit_type': edit_type,
        }
    ).apply(
        lambda row: align_wt_mut_sequences(
            row['wt'],
            row['mut'],
            int(row['lha_r']),
            edit_length=int(row['edit_len']),
            edit_type=int(row['edit_type']),
        ),
        axis=1,
        result_type='expand',
    )
    aligned.columns = ['wt_sequence', 'mut_sequence']
    wt_aligned = pd.Series(aligned['wt_sequence'], index=df.index)
    mut_aligned = pd.Series(aligned['mut_sequence'], index=df.index)

    # ---- Step 5: assemble score/efficiency/fold fields ----
    spcas9_score = pd.Series(pd.to_numeric(df['deepcas9'], errors='coerce'), index=df.index)
    efficiency_column = next(
        (col for col in df.columns if col.lower().endswith('averageedited')),
        None,
    )
    if efficiency_column is None:
        raise ValueError("Could not find cell-line specific averageedited column in PRIDICT2 data.")
    editing_efficiency = pd.Series(pd.to_numeric(df[efficiency_column], errors='coerce'), index=df.index)
    if "testset_fold" in df.columns:
        original_fold = pd.to_numeric(df["testset_fold"], errors="coerce")
    else:
        original_fold = None

    # ---- Step 6: build and save standardized output ----
    output_df = _build_standardized_output_df(
        pd.Series(df['group_id'], index=df.index), type_sub, type_ins, type_del, edit_len,
        wt_aligned, mut_aligned, protospacer_l, protospacer_r,
        pbs_l, pbs_r, rtt_wt_l, rtt_mut_r, lha_l, lha_r, rha_wt_l, rha_mut_r,
        spcas9_score, editing_efficiency, original_fold)

    output_path = DATA_ROOT / 'standardized' / 'pridict2' / dataset / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_df.to_parquet(output_path, index=False)
    logger.info("Saved standardized PRIDICT2 data: %s", output_path)

def _standardize_pridict1(
        data: Optional[pd.DataFrame], cell_line: str, pe_system: str, dataset: str) -> None:
    """
    Standardize PRIDICT1 library exports (library1, library2, library2-invivo)
    to the shared PE schema.
    """
    dataset = _normalize_name(dataset)
    cell_line = _normalize_name(cell_line)
    pe_system = _normalize_name(pe_system)
    input_name = f"{cell_line}-{pe_system}.csv"
    output_name = f"{cell_line}-{pe_system}.parquet"
    if data is None:
        data = pd.read_csv(DATA_ROOT / 'exported' / 'pridict1' / dataset / input_name)
    logger.info(
        "Standardizing PRIDICT1 dataset=%s cell_line=%s pe_system=%s rows=%s",
        dataset,
        cell_line,
        pe_system,
        len(data),
    )

    # PRIDICT target strand values are quoted in the source CSV ("'Fw'" / "'Rv'").
    # Keep the provided wide_* sequence orientation for both strands.
    # These sequences/locations are already internally consistent with PRIDICT's
    # pegRNA design output.
    # Step 
    df = data.copy()

    # ---- Step 1: Determine mutation type and filter invalid rows ----
    correction_type = pd.Series(df['Correction_Type'], copy=False).astype('string').str.strip().str.lower()
    type_sub = correction_type.eq('replacement')
    type_ins = correction_type.eq('insertion')
    type_del = correction_type.eq('deletion')
    unknown_mask = ~(type_sub | type_ins | type_del)
    if unknown_mask.any():
        unknown_values = df.loc[unknown_mask, 'Correction_Type'].astype(str).unique().tolist()[:5]
        raise ValueError(f"Unsupported Correction_Type values: {unknown_values}")

    # used for alignment, 0 for substitution, 1 for insertion, 2 for deletion
    edit_type = pd.Series(
        np.select([type_sub, type_ins, type_del], [0, 1, 2], default=-1),
        index=df.index,
    ).astype(int)
    edit_len = pd.Series(pd.to_numeric(df['Correction_Length'], errors='raise'), index=df.index).astype(int)

    # ---- Step 2: Compute protospacer and assign group IDs ----
    wt_sequence = pd.Series(df['wide_initial_target'], copy=False).astype('string').str.upper()
    mut_sequence = pd.Series(df['wide_mutated_target'], copy=False).astype('string').str.upper()

    protospacer_l, protospacer_r = _parse_pridict_location_column(
        pd.Series(df['protospacerlocation_only_initial'], copy=False), 'protospacerlocation_only_initial'
    )
    protospacer_bounds = pd.DataFrame(
        {"seq": wt_sequence, "l": protospacer_l, "r": protospacer_r},
        index=df.index,
    )
    df['protospacer'] = protospacer_bounds.apply(
        lambda row: (
            row["seq"][int(row["l"]):int(row["r"])]
            if isinstance(row["seq"], str)
            else ""
        ),
        axis=1,
    )
    df['group_id'] = df.groupby('protospacer').ngroup()

    # ---- Step 3: Compute PBS, RTT, LHA and RHA locations ----
    pbs_l, pbs_r = _parse_pridict_location_column(pd.Series(df['PBSlocation'], copy=False), 'PBSlocation')
    rtt_wt_l, rtt_wt_r = _parse_pridict_location_column(
        pd.Series(df['RT_initial_location'], copy=False), 'RT_initial_location'
    )
    rtt_mut_l, rtt_mut_r = _parse_pridict_location_column(
        pd.Series(df['RT_mutated_location'], copy=False), 'RT_mutated_location'
    )

    rha_len = pd.Series(df['RToverhang_seq'], copy=False).astype('string').str.upper().str.len().astype(int)
    lha_len = pd.Series(np.where(
        type_del,
        rtt_mut_r - rtt_mut_l - rha_len, 
        rtt_mut_r - rtt_mut_l - rha_len - edit_len,
    ), index=df.index).astype(int)
    lha_l = rtt_wt_l
    lha_r = rtt_wt_l + lha_len

    rha_wt_l = rtt_wt_r - rha_len
    rha_wt_r = rtt_wt_r
    rha_mut_r = rtt_mut_r

    # ---- Step 4: Align the wt and mut sequences ----
    aligned = pd.DataFrame(
        {
            'wt': wt_sequence,
            'mut': mut_sequence,
            'lha_r': lha_r,
            'edit_len': edit_len,
            'edit_type': edit_type,
        }
    ).apply(
        lambda row: align_wt_mut_sequences(
            row['wt'],
            row['mut'],
            int(row['lha_r']),
            edit_length=int(row['edit_len']),
            edit_type=int(row['edit_type']),
        ),
        axis=1,
        result_type='expand',
    )
    aligned.columns = ['wt_sequence', 'mut_sequence']

    # ---- Step 5: Concatenate spcas9 score and editing efficiency ----
    spcas9_score = pd.Series(pd.to_numeric(df['deepcas9'], errors='coerce'), index=df.index)
    average_edited = pd.Series(pd.to_numeric(df['averageedited'], errors='coerce'), index=df.index)
    if 'PE2df_percentageedited' in df.columns:
        pe2_edited = pd.Series(pd.to_numeric(df['PE2df_percentageedited'], errors='coerce'), index=df.index)
        editing_efficiency = pe2_edited.where(pe2_edited.notna(), average_edited)
    else:
        editing_efficiency = average_edited

    # ---- Step 6: Build output DataFrame ----
    output_df = _build_standardized_output_df(
        pd.Series(df['group_id'], index=df.index), type_sub, type_ins, type_del, edit_len, 
        wt_sequence, mut_sequence, protospacer_l, protospacer_r, 
        pbs_l, pbs_r, rtt_wt_l, rtt_mut_r, lha_l, lha_r, rha_wt_l, rha_mut_r, 
        spcas9_score, editing_efficiency)

    output_path = DATA_ROOT / 'standardized' / 'pridict1' / dataset / f"{output_name}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(output_path, index=False)
    logger.info(f"Saved standardized PRIDICT1 data to {output_path}")

def _standardize_minsepie(
        data: Optional[pd.DataFrame], cell_line: str, pe_system: str, dataset: str) -> None:
    """
    Convert MinSePIE data to the shared PE schema.

    MinSePIE is an insertion-only PE dataset. The pegRNA design information
    (``spacer`` / ``ha`` / ``pbs`` / insertion position) is enriched onto each
    experimental row during export from ST6 so this step reconstructs wide
    target-strand WT/Mut sequences and derives positional fields.

    When hg38 coordinates are known (see ``minsepie_genomic_loci.json``), each
    sequence is a 220 bp window with 100 bp genomic flanks on both sides of the
    20 bp protospacer. Targets without a mapped locus fall back to pegRNA-only
    reconstruction padded with 100 Ns per flank.

    Coordinate system (all positions are 0-indexed, left-inclusive right-exclusive):
      - The SpCas9 nick sits between protospacer positions 17 and 18; we anchor
        the protospacer at ``spacer_offset`` (100 for hg38-backed rows).
      - The ST6 ``ha`` column is stored in pegRNA (RTT) orientation. Where the
        insertion is not at +1, it contains a literal "-Ins-" marker that
        splits ``ha`` into (HA_left, HA_right) on the pegRNA. Relative to the
        target strand, the bases immediately 3' of the nick are RC(HA_right)
        and the bases further 3' (past the insertion) are RC(HA_left).
      - WT is padded with N's at the insertion position so WT and Mut share length.
    """
    dataset = _normalize_name(dataset)
    cell_line = _normalize_name(cell_line)
    pe_system = _normalize_name(pe_system)
    input_name = f"{cell_line}-{pe_system}.csv"
    output_name = f"{cell_line}-{pe_system}.parquet"
    if data is None:
        data = pd.read_csv(DATA_ROOT / "exported" / "minsepie" / dataset / input_name)

    logger.info(
        "Standardizing MinSePIE dataset=%s cell_line=%s pe_system=%s rows=%s",
        dataset,
        cell_line,
        pe_system,
        len(data),
    )

    df = data.copy()

    # ---- Step 1: clean up pegRNA design columns ----
    spacer_upper = df["pegrna_spacer"].astype(str).str.upper()
    pbs_upper = df["pegrna_pbs"].astype(str).str.upper()
    ha_upper = df["pegrna_ha"].astype(str).str.upper()
    ins_position = df["pegrna_ins_position"].astype(int)
    insertion_seq = df["insertion"].astype(str).str.upper()

    # Split HA around the "-Ins-" marker. If absent, the insertion sits at +1
    # (right at the nick) and the entire HA is on the 5' (HA_left) side.
    ha_split = ha_upper.str.split("-INS-", n=1, expand=True)
    if ha_split.shape[1] == 1:
        ha_split[1] = ""
    ha_left = ha_split[0].fillna("")
    ha_right = ha_split[1].fillna("")

    # HA without the marker (for sanity / full RC of downstream genomic region).
    ha_full = ha_left + ha_right

    pbs_len = pbs_upper.str.len().astype(int)

    # ---- Step 2: reconstruct wide WT and mut target-strand sequences ----
    rc_ha_right = ha_right.map(reverse_complement)
    rc_ha_left = ha_left.map(reverse_complement)
    edit_len = insertion_seq.str.len().astype(int)

    def _build_row_sequences(row: pd.Series) -> tuple[str, str, int, int]:
        target_key = _minsepie_genomic_target_key(str(row["experiment"]))
        ref = _minsepie_reference_window_for_target(target_key)
        if ref is not None:
            reference_window, spacer_offset = ref
            return _build_minsepie_wide_from_reference(
                reference_window,
                spacer_offset,
                str(row["spacer"]).upper(),
                str(row["ha_left"]),
                str(row["ha_right"]),
                str(row["insertion"]).upper(),
                int(row["ins_position"]),
            )
        core_wt, core_mut, row_edit_len, edit_position = _build_minsepie_core_target_sequences(
            str(row["spacer"]).upper(),
            str(row["ha_left"]),
            str(row["ha_right"]),
            str(row["insertion"]).upper(),
        )
        pad = "N" * _MINSEPIE_WIDE_FLANK_BP
        return pad + core_wt + pad, pad + core_mut + pad, row_edit_len, edit_position + _MINSEPIE_WIDE_FLANK_BP

    ha_split_df = pd.DataFrame({
        "experiment": df["experiment"],
        "spacer": spacer_upper,
        "ha_left": ha_left,
        "ha_right": ha_right,
        "insertion": insertion_seq,
        "ins_position": ins_position,
    })
    aligned = ha_split_df.apply(_build_row_sequences, axis=1, result_type="expand")
    aligned.columns = ["wt_sequence", "mut_sequence", "edit_len", "edit_position"]
    edit_len = aligned["edit_len"].astype(int)
    edit_position = aligned["edit_position"].astype(int)

    # ---- Step 3: assign group IDs (one group per unique protospacer) ----
    df["group_id"] = spacer_upper.groupby(spacer_upper).ngroup()

    # ---- Step 4: compute positional fields in the wide sequence ----
    protospacer_l = pd.Series(_MINSEPIE_WIDE_FLANK_BP, index=df.index, dtype=int)
    within_protospacer = edit_position < (protospacer_l + 20)
    protospacer_r = pd.Series(
        np.where(within_protospacer, protospacer_l + 20 + edit_len, protospacer_l + 20),
        index=df.index,
    ).astype(int)

    pbs_l = (protospacer_l + 17 - pbs_len).clip(lower=0).astype(int)
    pbs_r = protospacer_l + 17

    lha_l = protospacer_l + 17
    lha_r = edit_position.astype(int)
    rha_l = (edit_position + edit_len).astype(int)
    rha_r = (edit_position + edit_len + rc_ha_left.str.len()).astype(int)
    rtt_l = protospacer_l + 17
    rtt_r = rha_r

    # ---- Step 5: assemble efficiency, score and fold fields ----
    editing_efficiency = pd.to_numeric(df.get("percIns"), errors="coerce")
    spcas9_score = pd.Series(np.nan, index=df.index, dtype=float)

    # MinSePIE MOESM4 ``set`` labels library batches, not author train/test folds.

    # ---- Step 6: build and persist the standardized output ----
    type_ins = pd.Series(True, index=df.index)
    type_sub = pd.Series(False, index=df.index)
    type_del = pd.Series(False, index=df.index)

    output_df = _build_standardized_output_df(
        df["group_id"], type_sub, type_ins, type_del, edit_len,
        aligned["wt_sequence"], aligned["mut_sequence"],
        protospacer_l, protospacer_r,
        pbs_l, pbs_r, rtt_l, rtt_r,
        lha_l, lha_r, rha_l, rha_r,
        spcas9_score, editing_efficiency,
    )
    output_df = fill_missing_spcas9_scores(output_df)

    output_path = DATA_ROOT / "standardized" / "minsepie" / dataset / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(output_path, index=False)
    logger.info("Saved standardized MinSePIE data: %s", output_path)


def standardize_pe_data(
    *,
    study: str,
    cell_line: str,
    pe_system: str,
    dataset: Optional[str] = None,
) -> pd.DataFrame:
    """
    Standardize exported PE data into the shared schema.

    Input hierarchy:
    exported/{study}/{dataset}/{cell_line}-{pe_system}.csv

    Raises:
        ValueError: If the dataset has no supported standardization path.
    """
    study = _normalize_name(study)
    cell_line = _normalize_name(cell_line)
    pe_system = _normalize_name(pe_system)

    if dataset is None:
        available = DATASETS.get(study, [])
        if len(available) == 1:
            dataset = available[0]
        else:
            raise ValueError(
                f"dataset is required for study={study}. Available datasets: {available}"
            )
    dataset = str(dataset).strip().lower()
    normalized_dataset = _normalize_name(dataset)
    if not (is_standardizable(study, dataset) or is_partially_standardizable(study, dataset)):
        raise ValueError(
            f"Dataset {study}/{dataset} is not standardizable "
            "(full or partial entry-level path unavailable)."
        )
    dataset_candidates = [dataset]
    if normalized_dataset not in dataset_candidates:
        dataset_candidates.append(normalized_dataset)
    hyphenated = normalized_dataset.replace("_", "-")
    if hyphenated not in dataset_candidates:
        dataset_candidates.append(hyphenated)

    input_path = None
    for dataset_candidate in dataset_candidates:
        candidate_path = (
            DATA_ROOT / "exported" / study / dataset_candidate / f"{cell_line}-{pe_system}.csv"
        )
        if candidate_path.exists():
            input_path = candidate_path
            break
    if input_path is None:
        raise FileNotFoundError(
            "Exported input file not found for any dataset variant: "
            f"{dataset_candidates} (study={study}, cell_line={cell_line}, pe_system={pe_system})"
        )

    data = pd.read_csv(input_path)
    if study == "deeppe":
        _standardize_deeppe_ontarget(data, cell_line, pe_system, normalized_dataset)
    elif study == "deepprime":
        if normalized_dataset in {"deepprime_off", "deepprime_off_subpool"}:
            _standardize_deepprime_off(data, cell_line, pe_system, normalized_dataset)
        else:
            _standardize_deepprime_ontarget(data, cell_line, pe_system, normalized_dataset)
    elif study == "pridict1":
        if normalized_dataset in {"library1", "library2", "library2_invivo"}:
            _standardize_pridict1(data, cell_line, pe_system, normalized_dataset)
        elif normalized_dataset == "endogenous":
            _standardize_pridict1_endo(data, cell_line, pe_system, normalized_dataset)
        else:
            raise ValueError(
                f"Unsupported dataset for study=pridict1: {dataset}. "
                "Supported: ['library1', 'library2', 'library2-invivo', 'endogenous(partial)']"
            )
    elif study == "pridict2":
        if normalized_dataset in {"library_diverse", "library_diverse_invivo"}:
            _standardize_pridict2_library_diverse(data, cell_line, pe_system, normalized_dataset)
        elif normalized_dataset == "trip_analysis":
            _standardize_pridict2_trip(data, cell_line, pe_system, normalized_dataset)
        else:
            raise ValueError(
                "Unsupported dataset for study=pridict2: "
                f"{dataset}. Supported: ['library-diverse', 'library-diverse-invivo', 'trip-analysis(partial)']"
            )
    elif study == "minsepie":
        _standardize_minsepie(data, cell_line, pe_system, normalized_dataset)
    else:
        raise ValueError(f"Unsupported study: {study}")

    output_path = (
        DATA_ROOT / "standardized" / study / normalized_dataset /
        f"{cell_line}-{pe_system}.parquet"
    )
    if not output_path.exists():
        raise FileNotFoundError(
            "Standardization did not produce expected output file: "
            f"{output_path}"
        )
    return pd.read_parquet(output_path)


def _write_partial_standardized_output(
    partial_df: pd.DataFrame,
    *,
    study: str,
    dataset: str,
    cell_line: str,
    pe_system: str,
) -> None:
    """Persist a filter-only standardized parquet (not valid for model training)."""
    output_path = (
        DATA_ROOT / "standardized" / study / dataset / f"{cell_line}-{pe_system}.parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_df.to_parquet(output_path, index=False)
    logger.info(
        "Saved partial standardized data for filter-only use: %s (%s rows)",
        output_path,
        len(partial_df),
    )


def _standardize_deepprime_off(
    data: pd.DataFrame,
    cell_line: str,
    pe_system: str,
    dataset: str,
    *,
    study_key: str = "deepprime",
) -> None:
    """Partial standardization for DeepPrime off-target mismatch libraries.

    Supports ``deepprime-off`` and ``deepprime-off-subpool`` exports. Some sheets
    (e.g. hek293t-pe2) are already in DeepPrime model layout with ``type_sub`` /
    ``type_ins`` / ``type_del``; others use the supplementary ``edit_type`` column
    (``transv``, ``transi``, ``transv2``, ``Ins``, ``Del``).
    """
    cell_line = _normalize_name(cell_line)
    pe_system = _normalize_name(pe_system)
    dataset = _normalize_name(dataset)

    if {"type_sub", "type_ins", "type_del"}.issubset(data.columns):
        type_sub = data["type_sub"].astype(bool)
        type_ins = data["type_ins"].astype(bool)
        type_del = data["type_del"].astype(bool)
    elif "edit_type" in data.columns:
        labels = data["edit_type"].astype("string").str.strip().str.lower()
        # Mismatch-screen categories; mapped to PE-DB sub/ins/del for statistics only.
        type_sub = labels.isin({"transv", "transi", "transv2"})
        type_ins = labels.isin({"ins", "insertion"})
        type_del = labels.isin({"del", "deletion"})
    else:
        raise ValueError(
            f"DeepPrime off-target export for {dataset} {cell_line}-{pe_system} "
            "must include type_sub/type_ins/type_del or edit_type."
        )

    if "edit_length" in data.columns:
        edit_len = pd.to_numeric(data["edit_length"], errors="coerce")
    elif "edit_len" in data.columns:
        edit_len = pd.to_numeric(data["edit_len"], errors="coerce")
    else:
        edit_len = pd.Series(np.nan, index=data.index, dtype=float)

    if "avg.on-target_efficiency" in data.columns:
        efficiency_col = "avg.on-target_efficiency"
    else:
        efficiency_col = None
        for candidate in ("editing_efficiency", "measured_pe_efficiency"):
            if candidate in data.columns:
                efficiency_col = candidate
                break
    if efficiency_col is None:
        editing_efficiency = pd.Series(0.0, index=data.index, dtype=float)
    else:
        editing_efficiency = pd.to_numeric(data[efficiency_col], errors="coerce").fillna(0.0)

    _write_partial_standardized_output(
        pd.DataFrame(
            {
                "type_sub": type_sub,
                "type_ins": type_ins,
                "type_del": type_del,
                "edit_len": edit_len,
                "editing_efficiency": editing_efficiency.astype(float),
            }
        ),
        study=study_key,
        dataset=dataset,
        cell_line=cell_line,
        pe_system=pe_system,
    )


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


def _standardize_pridict1_endo(
    data: pd.DataFrame,
    cell_line: str,
    pe_system: str,
    dataset: str,
) -> None:
    """Partial standardization for PRIDICT1 endogenous validation pegRNAs."""
    cell_line = _normalize_name(cell_line)
    pe_system = _normalize_name(pe_system)
    dataset = _normalize_name(dataset)

    if {"type_sub", "type_ins", "type_del"}.issubset(data.columns):
        type_sub = data["type_sub"].astype(bool)
        type_ins = data["type_ins"].astype(bool)
        type_del = data["type_del"].astype(bool)
    elif "Correction_Type" in data.columns:
        type_sub, type_ins, type_del = _correction_type_to_flags(data["Correction_Type"])
    else:
        enriched = _attach_pridict1_endogenous_correction_fields(data)
        type_sub = enriched["type_sub"].astype(bool)
        type_ins = enriched["type_ins"].astype(bool)
        type_del = enriched["type_del"].astype(bool)

    preferred_efficiency = {
        "hek293t": "HEK293T_averageedited",
        "k562": "K562_averageedited",
    }
    efficiency_col = preferred_efficiency.get(cell_line)
    if efficiency_col is None or efficiency_col not in data.columns:
        raise ValueError(
            f"PRIDICT1 endogenous export for {cell_line}-{pe_system} "
            f"is missing efficiency column {efficiency_col!r}."
        )
    editing_efficiency = pd.to_numeric(data[efficiency_col], errors="coerce").fillna(0.0)

    if "Correction_Length" in data.columns:
        edit_len = pd.to_numeric(data["Correction_Length"], errors="coerce")
    else:
        edit_len = pd.Series(np.nan, index=data.index, dtype=float)

    _write_partial_standardized_output(
        pd.DataFrame(
            {
                "type_sub": type_sub,
                "type_ins": type_ins,
                "type_del": type_del,
                "edit_len": edit_len,
                "editing_efficiency": editing_efficiency.astype(float),
            }
        ),
        study="pridict1",
        dataset=dataset,
        cell_line=cell_line,
        pe_system=pe_system,
    )


def _standardize_pridict2_trip(
    data: pd.DataFrame,
    cell_line: str,
    pe_system: str,
    dataset: str,
) -> None:
    """Partial standardization for PRIDICT2 TRIP endogenous chromatin survey.

    TRIP supplementary table 12 reports barcode, genomic coordinates, and editing
    efficiencies only. All integrations were edited with the same pegRNA
    (``TRIP_pegRNA_GtoC`` in supplementary table 5): a uniform 1-bp G→C substitution
    used to survey chromatin context (Mathis et al., Nat. Biotechnol. 2024).
    """
    cell_line = _normalize_name(cell_line)
    pe_system = _normalize_name(pe_system)
    dataset = _normalize_name(dataset)

    if "PE_editing_efficiency" not in data.columns:
        raise ValueError(
            f"PRIDICT2 TRIP export for {cell_line}-{pe_system} "
            "is missing PE_editing_efficiency."
        )
    editing_efficiency = pd.to_numeric(
        data["PE_editing_efficiency"], errors="coerce"
    ).fillna(0.0)

    type_sub = pd.Series(True, index=data.index, dtype=bool)
    type_ins = pd.Series(False, index=data.index, dtype=bool)
    type_del = pd.Series(False, index=data.index, dtype=bool)
    edit_len = pd.Series(1.0, index=data.index, dtype=float)

    _write_partial_standardized_output(
        pd.DataFrame(
            {
                "type_sub": type_sub,
                "type_ins": type_ins,
                "type_del": type_del,
                "edit_len": edit_len,
                "editing_efficiency": editing_efficiency.astype(float),
            }
        ),
        study="pridict2",
        dataset=dataset,
        cell_line=cell_line,
        pe_system=pe_system,
    )
