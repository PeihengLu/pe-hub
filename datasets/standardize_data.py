import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pe_common.constants import DATA_ROOT
from pe_common.sequence_utils import align_wt_mut_sequences, reverse_complement

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SUPPORTED_STUDIES = {"deepprime", "pridict2", "pridict1", "minsepie"}


def _normalize_name(value: str) -> str:
    """Normalize dataset/cell line/PE system names for filenames."""
    return str(value).strip().lower().replace("-", "_")


def export_original_data(study: Optional[str] = None) -> None:
    """
    Export raw study files into standardized exported datasheets.

    Output naming convention:
    ``{study}/{dataset}/{cell_line}-{pe_system}.csv``

    Args:
        study: Study key. If None, export all supported studies.
    """
    exporters = {
        "deepprime": _export_deepprime_datasheets,
        "pridict1": _export_pridict1_datasheets,
        "pridict2": _export_pridict2_library_diverse_datasheets,
        "minsepie": _export_minsepie_datasheets,
    }

    target_studies = sorted(exporters) if study is None else [study]
    for study_name in target_studies:
        if study_name not in exporters:
            raise ValueError(f"Unknown study: {study_name}")
        logger.info("Exporting study=%s", study_name)
        exporters[study_name]()

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

    # TODO: work on library2 and endogenous data


def _export_pridict2_library_diverse_datasheets() -> None:
    """
    Export the PRIDICT2 Library Diverse datasheet
    """
    original_excel_path = DATA_ROOT / 'raw' / 'pridict2' / 'pridict2-org.xlsx'
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

        output_path = (DATA_ROOT / 'exported' / 'pridict2' / 'library-diverse' /
                       f'{cell_line.lower().replace("-", "_")}-pe2.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path, index=False)
        logger.info(f"Saved PRIDICT2 library diverse data for {cell_line} to {output_path}")

def _export_minsepie_datasheets() -> None:
    """
    Export the MinSePIE datasheets.

    MinSePIE (Nature Biotechnology, 2023) is an insertion-only PE dataset that
    measures editing efficiency for a library of DNA insertions at specific
    genomic target sites across different cell lines and PE system variants.

    Raw data consists of two supplementary tables:
      - MOESM4 (Suppl. Table 4): Insert library mapping names → sequences + set/fold
      - MOESM5 (Suppl. Table 5): Experimental results (editing efficiency per insert/target/condition)

    TODO: reformat this
    Data is grouped by experiment and exported as separate CSV files with naming:
        {target}-{cell_line}-{pe_condition}.csv
    """
    # Read the main experimental data (Supplementary Table 5)
    data_path = DATA_ROOT / 'raw' / 'minsepie' / '41587_2023_1678_MOESM5_ESM.tsv'
    data = pd.read_csv(data_path, sep='\t')

    # Read the insert library (Supplementary Table 4) for set/fold assignments
    library_path = DATA_ROOT / 'raw' / 'minsepie' / '41587_2023_1678_MOESM4_ESM.tsv'
    library = pd.read_csv(library_path, sep='\t')

    # The library contains forward/reverse complement pairs with the same name
    # but different sequences. Since we only need the 'set' column (same for both),
    # deduplicate by name before merging to avoid row explosion.
    library_dedup = library.drop_duplicates(subset='name', keep='first')

    # Left-join to add set info (some names won't match — those get NaN)
    data = data.merge(library_dedup[['name', 'set']], on='name', how='left')

    # Fix "rc" cell_line values — these are HEK293T experiments with reverse
    # complement target orientation, not a different cell line.
    data.loc[data['cell_line'] == 'rc', 'cell_line'] = 'HEK293T'

    # Experiment names use abbreviated cell line names (e.g. "293T" for HEK293T)
    cell_abbr_to_full = {'293T': 'hek293t', 'HAP1': 'hap1', 'HAP1dMLH1': 'hap1dmlh1'}

    logger.info(f"Exporting MinSePIE data: {len(data)} entries across "
                f"{data['experiment'].nunique()} experiments")

    # TODO: work on the library insert data

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

def standardize_pe_data(
    study: Optional[str] = None,
    dataset: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    cell_line: Optional[str] = None,
    pe_system: Optional[str] = None,
) -> None:
    """
    Standardize the PE data format

    Args:
        study: Study name. If None, standardize all studies under exported/.
        dataset: Dataset name
        data: DataFrame in original format
        cell_line: Cell line name
        pe_system: PE system name
        
    Returns:
        DataFrame in standardized format
    """
    def _dispatch_single_file(
        current_study: str,
        current_dataset: str,
        file_data: Optional[pd.DataFrame],
        file_cell_line: str,
        file_pe_system: str,
    ) -> None:
        if current_study == "deepprime":
            if current_dataset in {"deepprime-clinvar", "deepprime-small"}:
                _standardize_deepprime_ontarget(
                    file_data, file_cell_line, file_pe_system, current_dataset
                )
            else:
                raise ValueError(
                    "Unsupported DeepPrime dataset for standardization: "
                    f"{current_dataset}"
                )
        elif current_study == "pridict2":
            if current_dataset == "library-diverse":
                if file_data is None:
                    raise ValueError(
                        "PRIDICT2 standardization requires file-backed input data."
                    )
                _standardize_pridict2_library_diverse(
                    file_data, file_cell_line, file_pe_system, current_dataset
                )
            else:
                raise ValueError(
                    "Unsupported PRIDICT2 dataset for standardization: "
                    f"{current_dataset}"
                )
        elif current_study == "pridict1":
            if current_dataset == "library1":
                _standardize_pridict1_library1(
                    file_data, file_cell_line, file_pe_system, current_dataset
                )
            else:
                raise ValueError(
                    "Unsupported PRIDICT1 dataset for standardization: "
                    f"{current_dataset}"
                )
        elif current_study == "minsepie":
            _standardize_minsepie(
                file_data, file_cell_line, file_pe_system, current_dataset
            )
        else:
            raise ValueError(f"Unknown study: {current_study}")

    if data is not None:
        if None in (study, dataset, cell_line, pe_system):
            raise ValueError(
                "When `data` is provided, you must also provide "
                "`study`, `dataset`, `cell_line`, and `pe_system`."
            )
        assert study is not None
        assert dataset is not None
        assert cell_line is not None
        assert pe_system is not None
        provided_study = study
        provided_dataset = dataset
        provided_cell_line = cell_line
        provided_pe_system = pe_system
        _dispatch_single_file(
            provided_study, provided_dataset, data, provided_cell_line, provided_pe_system
        )
        return

    exported_root = DATA_ROOT / "exported"
    if study is None:
        study_names = sorted(
            p.name for p in exported_root.iterdir()
            if p.is_dir() and p.name in SUPPORTED_STUDIES
        )
    else:
        if study not in SUPPORTED_STUDIES:
            raise ValueError(f"Unknown study: {study}")
        study_names = [study]

    for study_name in study_names:
        study_root = exported_root / study_name
        if not study_root.exists():
            logger.warning(f"Exported study directory not found: {study_root}")
            continue

        if dataset is None:
            dataset_names = sorted(p.name for p in study_root.iterdir() if p.is_dir())
        else:
            dataset_names = [dataset]

        for dataset_name in dataset_names:
            dataset_root = study_root / dataset_name
            if not dataset_root.exists() or not dataset_root.is_dir():
                logger.warning(f"Exported dataset directory not found: {dataset_root}")
                continue

            if cell_line is None or pe_system is None:
                csv_files = sorted(dataset_root.glob("*.csv"))
                parquet_files = sorted(dataset_root.glob("*.parquet"))
                # Prefer CSV exports; keep parquet as backward-compatible fallback.
                files_to_standardize = csv_files if csv_files else parquet_files

                for file_path in files_to_standardize:
                    logger.info("Standardizing file=%s dataset=%s", file_path.name, dataset_name)
                    stem_parts = file_path.stem.split("-", 1)
                    if len(stem_parts) != 2:
                        logger.warning(
                            "Skipping file with unexpected name format "
                            f"(expected <cell_line>-<pe_system>.<ext>): {file_path.name}"
                        )
                        continue
                    file_cell_line, file_pe_system = stem_parts
                    if file_path.suffix == ".csv":
                        file_data = pd.read_csv(file_path)
                    elif file_path.suffix == ".parquet":
                        file_data = pd.read_parquet(file_path)
                    else:
                        logger.warning("Skipping unsupported file type: %s", file_path)
                        continue
                    _dispatch_single_file(
                        study_name, dataset_name, file_data, file_cell_line, file_pe_system
                    )
                    logger.info("Completed file=%s dataset=%s", file_path.name, dataset_name)
            else:
                _dispatch_single_file(study_name, dataset_name, None, cell_line, pe_system)

def _standardize_data_types(data: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the data types in output dataframe
    
    Args:
        data: DataFrame in standardized format with possibly incorrect types
        
    Returns:
        DataFrame with correct types
    """
    # String transformations
    data['wt_sequence'] = data['wt_sequence'].str.upper()
    data['mut_sequence'] = data['mut_sequence'].str.upper()

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
    float_columns = ['spcas9_score', 'editing_efficiency']

    data[bool_columns] = data[bool_columns].astype(bool)
    data[int_columns] = data[int_columns].astype(int)
    data[float_columns] = data[float_columns].astype(float)
    data['original_fold'] = data['original_fold'].astype(int) if 'original_fold' in data.columns else np.nan

    return data

def _standardize_deepprime_ontarget(
        data: Optional[pd.DataFrame], cell_line: str, pe_system: str, dataset: str) -> None:
    """
    Standardize DeepPrime on-target datasets to the shared PE schema.
    """
    dataset = _normalize_name(dataset)
    cell_line = _normalize_name(cell_line)
    pe_system = _normalize_name(pe_system)
    input_name = f"{cell_line}-{pe_system}.csv"
    output_name = f"{cell_line}-{pe_system}.parquet"
    if data is None:
        data = pd.read_csv(DATA_ROOT / 'exported' / 'deepprime' / dataset / input_name)

    logger.info(
        "Standardizing DeepPrime dataset=%s cell_line=%s pe_system=%s rows=%s",
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
    # In DeepPrime's format, mut_sequence uses leading 'x' characters to mask
    # positions upstream of the PBS that are not involved in the editing process.
    # The PBS left boundary is therefore the index of the first non-'x' character.
    mut_sequence = pd.Series(df['mut_sequence'], dtype='string').fillna('')
    df['pbs_l'] = mut_sequence.map(lambda seq: len(seq) - len(seq.lstrip('x')))
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

        # Reconstruct the unmasked mutated sequence from the wild type
        mut = wt[:lha_r]
        if type_ins:  # insertion: trim trailing bases
            mut += wt[rha_wt_r_val:len(wt) - edit_len]
        elif type_del:  # deletion: trim leading bases
            mut += wt[rha_wt_r_val + edit_len:]
        else:  # substitution: trim leading and trailing bases
            mut += wt[rha_wt_r_val:]

        # Pad with N at the edit position to align wt/mut to the same length
        mut_type = row['mut_type']
        return align_wt_mut_sequences(
                wt, mut, lha_r, edit_length=edit_len, edit_type=mut_type)

    aligned = df.apply(_reconstruct_and_align, axis=1, result_type='expand')
    aligned.columns = ['wt_aligned', 'mut_aligned']

    # ---- Step 6: Build output DataFrame ----
    output_df = pd.DataFrame({
        'group_id': df['group_id'],
        'type_sub': df['type_sub'],
        'type_ins': df['type_ins'],
        'type_del': df['type_del'],
        'edit_len': df['edit_len'],
        'wt_sequence': aligned['wt_aligned'],
        'mut_sequence': aligned['mut_aligned'],
        'protospacer_location_l': PROTOSPACER_L,
        'protospacer_location_r': PROTOSPACER_R,
        'pbs_location_l': df['pbs_l'],
        'pbs_location_r': df['pbs_r'],
        'rtt_location_l': df['rtt_l'],
        'rtt_location_r': df['rtt_r'],
        'lha_location_l': df['lha_l'],
        'lha_location_r': df['lha_r'],
        'rha_location_l': df['rha_l'],
        'rha_location_r': df['rha_r'],
        'spcas9_score': df['deepspcas9_score'],
        'editing_efficiency': df['measured_pe_efficiency'],
        'original_fold': df['fold'] if 'fold' in df.columns else np.nan,
    })
    # replace 'Test' in original_fold with -1
    output_df['original_fold'] = output_df['original_fold'].replace('Test', -1)
    # cast to correct type before saving
    output_df = _standardize_data_types(output_df)

    # export the data to a parquet file
    output_path = DATA_ROOT / 'standardized' / 'deepprime' / dataset / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(output_path, index=False)
    logger.info("Saved standardized DeepPrime data: %s", output_path)

def _standardize_pridict2_library_diverse(
        data: Optional[pd.DataFrame], cell_line: str, pe_system: str, dataset: str) -> None:
    """
    Standardize PRIDICT2 library-diverse data to the shared PE schema.
    """
    dataset = _normalize_name(dataset)
    cell_line = _normalize_name(cell_line)
    pe_system = _normalize_name(pe_system)
    output_name = f"{dataset}-{cell_line}-{pe_system}.parquet"
    if data is None:
        data = pd.read_parquet(DATA_ROOT / 'exported' / 'pridict2' / dataset / output_name)
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
    raise NotImplementedError("Standardization logic for PRIDICT2 is not implemented yet")

def _standardize_pridict1_library1(
        data: Optional[pd.DataFrame], cell_line: str, pe_system: str, dataset: str) -> None:
    """
    Standardize PRIDICT1 library1 data to the shared PE schema.
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

    def _parse_location_column(location_series: pd.Series, column_name: str) -> tuple[pd.Series, pd.Series]:
        """Vectorized parser for location strings like '[13, 26]'."""
        series = pd.Series(location_series, copy=False)
        extracted = series.astype('string').str.extract(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")
        invalid = pd.Series(extracted.isna().any(axis=1), index=series.index)
        if bool(invalid.any()):
            bad_examples = [str(value) for value in series[invalid].tolist()[:3]]
            raise ValueError(
                f"Invalid location format in column {column_name}: {bad_examples}"
            )
        return extracted[0].astype(int), extracted[1].astype(int)

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

    protospacer_l, protospacer_r = _parse_location_column(
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
    pbs_l, pbs_r = _parse_location_column(pd.Series(df['PBSlocation'], copy=False), 'PBSlocation')
    rtt_wt_l, rtt_wt_r = _parse_location_column(
        pd.Series(df['RT_initial_location'], copy=False), 'RT_initial_location'
    )
    rtt_mut_l, rtt_mut_r = _parse_location_column(
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
    output_df = pd.DataFrame({
        'group_id': df['group_id'],
        'type_sub': type_sub,
        'type_ins': type_ins,
        'type_del': type_del,
        'edit_len': edit_len,
        'wt_sequence': aligned['wt_sequence'],
        'mut_sequence': aligned['mut_sequence'],
        'protospacer_location_l': protospacer_l,
        'protospacer_location_r': protospacer_r,
        'pbs_location_l': pbs_l,
        'pbs_location_r': pbs_r,
        'rtt_location_l': rtt_wt_l,
        # deletion results in the same location as the wild type after alignment
        'rtt_location_r': np.where(type_del, rtt_wt_r, rtt_mut_r),
        'lha_location_l': lha_l,
        'lha_location_r': lha_r,
        'rha_location_l': rha_wt_l,
        # deletion results in the same location as the wild type after alignment
        'rha_location_r': np.where(type_del, rha_wt_r, rha_mut_r), 
        'spcas9_score': spcas9_score,
        'editing_efficiency': editing_efficiency,
    })
    # cast to correct type before saving
    output_df = _standardize_data_types(output_df)

    output_path = DATA_ROOT / 'standardized' / 'pridict1' / dataset / f"{output_name}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(output_path, index=False)
    logger.info(f"Saved standardized PRIDICT1 data to {output_path}")

def _standardize_minsepie(
        data: Optional[pd.DataFrame], cell_line: str, pe_system: str, dataset: str) -> None:
    """
    Convert MinSePIE data to the shared PE schema.

    MinSePIE is an insertion-only PE dataset. The pegRNA design information
    (spacer, PBS, HA) is loaded from MOESM3 (Supplementary Table 3) to
    reconstruct the full WT and mutated target strand sequences and compute
    all standard positional indices.

    Key reconstruction:
      - WT target strand = spacer[:17] + RC(HA)  (guide-same strand, 5'→3')
      - Mut target strand = wt[:ins_pos] + insertion + wt[ins_pos:]
        where ins_pos = 17 + len(HA_right)  (insertion offset from nick)
      - Mut is padded with N at the edit position to align with WT
    """
    pass

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Standardize data from various PE studies")
    argparser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = argparser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
        exit(0)

    # export_original_data()

    # standardize_pe_data(study="deepprime", dataset="deepprime-small")
    # standardize_pe_data(study="deepprime", dataset="deepprime-clinvar")
    standardize_pe_data(study="pridict1", dataset="library1")