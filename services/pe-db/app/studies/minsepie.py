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

from ..catalog.scaffolds import SCAFFOLD_MINSEPIE_CODON_VARIANT

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

def _minsepie_endo_coordinates(experiments: pd.Series) -> pd.DataFrame:
    """Map MinSePIE experiments to hg38 protospacer coordinates when known."""
    meta = _load_minsepie_genomic_loci()
    loci = meta.get("loci", {})
    rows: list[dict[str, Any]] = []
    for experiment in experiments.astype(str):
        target_key = _minsepie_genomic_target_key(experiment)
        locus = loci.get(target_key)
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
                    "endo_locus_id": target_key,
                }
            )
            continue
        # JSON spacer_start is 1-based; store 0-based half-open protospacer interval.
        spacer_start_0 = int(locus["spacer_start"]) - 1
        rows.append(
            {
                "endo_genome_build": "hg38",
                "endo_chr": str(locus["chrom"]),
                "endo_start": spacer_start_0,
                "endo_end": spacer_start_0 + 20,
                "endo_strand": int(locus["assembly_strand"]),
                "endo_coord_ref": "protospacer",
                "endo_coord_source": "minsepie_genomic_loci.json",
                "endo_locus_id": target_key,
            }
        )
    return pd.DataFrame(rows, index=experiments.index)


_DEEPPE_GENOMIC_LOCI_PATH = DATA_ROOT / "raw" / "deeppe" / "deeppe_genomic_loci.json"
_PRIDICT1_LIBRARY2_GENOMIC_LOCI_PATH = (
    DATA_ROOT / "raw" / "pridict1" / "pridict1_library2_genomic_loci.json"
)



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
    # Store the 20 bp protospacer, not spacer+PAM. Nick is protospacer_l+17
    # (SpCas9, 3 bp upstream of PAM), independent of indel pads that may sit
    # inside the last 3 bp of the spacer.
    protospacer_l = pd.Series(_MINSEPIE_WIDE_FLANK_BP, index=df.index, dtype=int)
    protospacer_r = (protospacer_l + 20).astype(int)

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
    output_df = _attach_endo_coordinate_columns(
        output_df,
        _minsepie_endo_coordinates(df["experiment"]),
    )

    output_path = DATA_ROOT / "standardized" / "minsepie" / dataset / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(output_path, index=False)
    logger.info("Saved standardized MinSePIE data: %s", output_path)


def _scaffold_assignments(data_root=None):
    from ..catalog.datasheets import build_minsepie_scaffold_assignments
    return build_minsepie_scaffold_assignments(data_root)


register_study(StudyPipeline(
    key="minsepie",
    exporters=(_export_minsepie_datasheets,),
    standardizers={
        "library_insert_set12": _standardize_minsepie,
        "library_insert_18nt": _standardize_minsepie,
        "library_insert_codon_variant": _standardize_minsepie,
        "library_insert_codon_hek3": _standardize_minsepie,
        "library_insert_piggybac": _standardize_minsepie,
    },
    scaffold_assignments=_scaffold_assignments,
))
