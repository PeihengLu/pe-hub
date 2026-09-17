"""Anzalone et al. 2019 endogenous PE2/PE3 validation (Easy-Prime reformatted)."""

from __future__ import annotations

import logging
import re
from typing import Optional

import pandas as pd

from pe_common.constants import DATA_ROOT
from pe_common.sequence_utils import reverse_complement

from ..catalog.records import DatasheetScaffoldAssignment
from ..catalog.scaffolds import SCAFFOLD_ID_CONVENTIONAL
from ..pipeline.endo import expand_endogenous_frame, load_loci_json
from ..pipeline.names import _normalize_name
from ..pipeline.registry import StudyPipeline, register_study
from ..pipeline.schema import _attach_endo_coordinate_columns
from .deeppe import _prepare_deeppe_export_df
from .deepprime import _standardize_deepprime_ontarget

logger = logging.getLogger(__name__)

_RAW_TABLE = DATA_ROOT / "raw" / "anzalone" / "anzalone_raw_table.csv"
_FEATURE_MATRIX = DATA_ROOT / "raw" / "anzalone" / "anzalone_feature_matrix.csv"
# Liu OPED Fig. 2i: 350 bp sense windows (165+20+165) from local hg38 spacer match.
_GENOMIC_LOCI = DATA_ROOT / "raw" / "anzalone" / "anzalone_genomic_loci.json"
_OPED_CONTEXT_BP = 350
_OPED_SPACER_OFFSET = 165


# DeepPrime native window (Yu et al. Cell 2023). Cropping to DeepPE's 47 bp
# and right-padding with N under-states Fig. 3H (DeepPrime encodes N as zeros).
_WIDE_COL = (
    "Wide target sequence (Total 74 bps = 4 bp neighboring sequence + 20 bp "
    "protospacer + 3 bp NGG PAM+ 47 bp neighboring sequence)"
)
_GUIDE_COL = "Guide sequence ((G/g)N19)"
_EXT_COL = "3' extension sequence of pegRNA"
_PBS_COL = "PBS length (nt)"
_RTT_COL = "RT template length (nt)"
_WIDE_LEN = 74
_SPACER_UPSTREAM = 4
_SPACER_LEN = 20
_PAM_LEN = 3
_DOWNSTREAM = _WIDE_LEN - _SPACER_UPSTREAM - _SPACER_LEN - _PAM_LEN  # 47

_REP_SUFFIX = re.compile(r"_REP[123]$", re.IGNORECASE)


def _design_id(sample_id: str) -> str:
    return _REP_SUFFIX.sub("", str(sample_id).strip())


def _spacer_sense_amplicon(reference_amplicon: str, spacer: str, strand: str) -> tuple[str, str, int]:
    """
    Orient the amplicon so the pegRNA spacer appears on the + strand.

    Returns (sense_amplicon, sense_spacer, spacer_start).
    """
    wt = str(reference_amplicon).upper().replace("U", "T")
    spacer_dna = str(spacer).upper().replace("U", "T")
    strand_token = str(strand).strip()
    if strand_token == "-":
        sense = reverse_complement(wt)
        idx = sense.find(spacer_dna)
        if idx < 0:
            raise ValueError("Spacer not found on reverse-complemented amplicon")
        return sense, spacer_dna, idx
    idx = wt.find(spacer_dna)
    if idx < 0:
        raise ValueError("Spacer not found on amplicon")
    return wt, spacer_dna, idx


def _to_deepprime_wide_row(row: pd.Series) -> dict[str, object]:
    """Crop Easy-Prime amplicons to the DeepPrime 74 bp wide-target layout."""
    sense_wt, spacer, idx = _spacer_sense_amplicon(
        str(row["reference_amplicon"]),
        str(row["correct_pegRNA"]),
        str(row["pegRNA_strand"]),
    )
    left = idx - _SPACER_UPSTREAM
    right = idx + _SPACER_LEN + _PAM_LEN + _DOWNSTREAM
    pad_l = max(0, -left)
    pad_r = max(0, right - len(sense_wt))
    if pad_l or pad_r:
        sense_wt = ("N" * pad_l) + sense_wt + ("N" * pad_r)
        idx = idx + pad_l
        left = idx - _SPACER_UPSTREAM
        right = left + _WIDE_LEN
    wide = sense_wt[left:right]
    if len(wide) != _WIDE_LEN or wide[_SPACER_UPSTREAM : _SPACER_UPSTREAM + _SPACER_LEN] != spacer:
        raise ValueError(
            f"Failed to build {_WIDE_LEN} bp wide target for {row['design_id']!r} "
            f"(len={len(wide)}, spacer_match="
            f"{wide[_SPACER_UPSTREAM:_SPACER_UPSTREAM + _SPACER_LEN] == spacer})"
        )
    pbs = str(row["PBS_seq"]).upper().replace("U", "T")
    rtt = str(row["RTS_seq"]).upper().replace("U", "T")
    # pegRNA 3' extension is RTT then PBS (5'→3'); DeepPE prepare RC-aligns PBS.
    extension = rtt + pbs
    efficiency_pct = float(row["editing_frequency"]) * 100.0
    return {
        _WIDE_COL: wide,
        _GUIDE_COL: spacer,
        _EXT_COL: extension,
        _PBS_COL: int(row["PBS_length"]),
        _RTT_COL: int(row["RTS_length"]),
        "editing_efficiency": efficiency_pct,
        "design_id": str(row["design_id"]),
        "gene": str(row["gene"]),
        "chrom": row.get("CHROM"),
        "genomic_pos": row.get("POS"),
        "ref_allele": row.get("REF"),
        "alt_allele": row.get("ALT"),
    }


# Back-compat alias for older imports / tests.
_to_deeppe_wide_row = _to_deepprime_wide_row


def _load_averaged_designs() -> pd.DataFrame:
    """
    Average Easy-Prime replicates and keep the curated PE2/PE3 feature-matrix designs.

    PE2 = null ``nick_to_pegRNA`` (n=199); PE3 = non-null (n=278).
    """
    raw = pd.read_csv(_RAW_TABLE)
    feat = pd.read_csv(_FEATURE_MATRIX)
    raw = raw.copy()
    raw["design_id"] = raw["sample_ID"].map(_design_id)
    agg = (
        raw.groupby("design_id", as_index=False)
        .agg(
            {
                "editing_frequency": "mean",
                "gene": "first",
                "correct_pegRNA": "first",
                "PBS_seq": "first",
                "RTS_seq": "first",
                "PBS_length": "first",
                "RTS_length": "first",
                "reference_amplicon": "first",
                "pegRNA_strand": "first",
                "CHROM": "first",
                "POS": "first",
                "REF": "first",
                "ALT": "first",
            }
        )
    )
    merged = agg.merge(
        feat[["sample_name", "nick_to_pegRNA", "Target"]],
        left_on="design_id",
        right_on="sample_name",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(feat):
        raise ValueError(
            f"Anzalone raw/feature join expected {len(feat)} designs, got {len(merged)}"
        )
    return merged


def _export_anzalone_endogenous_datasheets() -> None:
    """
    Export Anzalone endogenous PE2/PE3 sheets in DeepPrime 74 bp wide-target columns.

    Source: Li et al. Easy-Prime reformatting of Anzalone et al. 2019 HEK293T
    endogenous experiments (doi:10.1186/s13059-021-02458-0). Amplicons are long
    enough for a true Yu-style 74-mer (no 3' N pad).
    """
    if not _RAW_TABLE.exists() or not _FEATURE_MATRIX.exists():
        raise FileNotFoundError(
            f"Anzalone raw inputs missing under {DATA_ROOT / 'raw' / 'anzalone'}"
        )
    designs = _load_averaged_designs()
    cell_line = _normalize_name("HEK293T")
    splits = (
        ("pe2", designs["nick_to_pegRNA"].isna()),
        ("pe3", designs["nick_to_pegRNA"].notna()),
    )
    for pe_system, mask in splits:
        subset = designs.loc[mask]
        rows: list[dict[str, object]] = []
        for _, row in subset.iterrows():
            rows.append(_to_deepprime_wide_row(row))
        export_df = pd.DataFrame(rows)
        output_path = (
            DATA_ROOT
            / "exported"
            / "anzalone"
            / "anzalone-endo"
            / f"{cell_line}-{pe_system}.csv"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_df.to_csv(output_path, index=False)
        logger.info(
            "Saved Anzalone datasheet: %s (%s rows)",
            output_path,
            len(export_df),
        )


def _load_anzalone_genomic_loci() -> dict:
    """Load cached hg38 350 bp OPED windows keyed by Easy-Prime design_id."""
    if not _GENOMIC_LOCI.exists():
        raise FileNotFoundError(
            f"Missing Anzalone genomic loci for OPED 350 bp windows: {_GENOMIC_LOCI}"
        )
    return load_loci_json(str(_GENOMIC_LOCI))


def _standardize_anzalone_endo(
    data: Optional[pd.DataFrame],
    cell_line: str,
    pe_system: str,
    dataset: str,
) -> None:
    """Standardize Anzalone via DeepPE column normalize → DeepPrime layout path.

    After the 74 bp PE-core standardize, expand onto Liu Fig. 2i 350 bp hg38
    windows (spacer at offset 165). OPED then receives the full unpadded WT
    (no further crop or invented flanks). DeepPrime / PRIDICT converters still crop from the spacer.
    """
    dataset = _normalize_name(dataset)
    cell_line = _normalize_name(cell_line)
    pe_system = _normalize_name(pe_system)
    input_name = f"{cell_line}-{pe_system}.csv"
    if data is None:
        export_dir = DATA_ROOT / "exported" / "anzalone"
        candidates = [
            export_dir / dataset / input_name,
            export_dir / dataset.replace("_", "-") / input_name,
        ]
        for path in candidates:
            if path.exists():
                data = pd.read_csv(path)
                break
        else:
            raise FileNotFoundError(
                f"Anzalone export not found for {dataset}/{input_name} "
                f"(tried {[str(p) for p in candidates]})"
            )
    design_ids = (
        data["design_id"].astype(str).tolist()
        if "design_id" in data.columns
        else None
    )
    prepared = _prepare_deeppe_export_df(data)
    _standardize_deepprime_ontarget(
        prepared,
        cell_line,
        pe_system,
        dataset,
        study_key="anzalone",
    )
    output_path = (
        DATA_ROOT
        / "standardized"
        / "anzalone"
        / dataset
        / f"{cell_line}-{pe_system}.parquet"
    )
    output_df = pd.read_parquet(output_path).reset_index(drop=True)
    if design_ids is None or len(design_ids) != len(output_df):
        logger.warning(
            "Anzalone OPED 350 bp expand skipped: design_id alignment "
            "missing or length mismatch (%s vs %s)",
            None if design_ids is None else len(design_ids),
            len(output_df),
        )
        return
    loci_payload = _load_anzalone_genomic_loci()
    loci = loci_payload.get("loci", {})
    windows: list[object] = []
    coord_rows: list[dict] = []
    for design_id in design_ids:
        entry = loci.get(str(design_id))
        if not entry:
            windows.append(pd.NA)
            coord_rows.append({})
            continue
        windows.append(str(entry["reference_window"]).upper())
        strand = str(entry.get("strand") or "").strip()
        spacer_start = entry.get("spacer_start_0")
        chrom = entry.get("chrom")
        if spacer_start is None or chrom is None:
            coord_rows.append({})
            continue
        start = int(spacer_start) - _OPED_SPACER_OFFSET
        end = start + _OPED_CONTEXT_BP
        strand_val = 1 if strand == "+" else (-1 if strand == "-" else pd.NA)
        coord_rows.append(
            {
                "endo_genome_build": str(entry.get("genome_build") or "hg38"),
                "endo_chr": str(chrom),
                "endo_start": int(start),
                "endo_end": int(end),
                "endo_strand": strand_val,
                "endo_coord_ref": str(design_id),
                "endo_coord_source": "anzalone_genomic_loci",
                "endo_locus_id": str(design_id),
            }
        )
    output_df = _attach_endo_coordinate_columns(
        output_df, pd.DataFrame(coord_rows, index=output_df.index)
    )
    output_df = expand_endogenous_frame(
        output_df,
        pd.Series(windows, index=output_df.index, dtype="string"),
        spacer_offset=_OPED_SPACER_OFFSET,
        context_bp=_OPED_CONTEXT_BP,
    )
    output_df.to_parquet(output_path, index=False)
    logger.info(
        "Expanded Anzalone rows onto %s bp OPED windows: %s",
        _OPED_CONTEXT_BP,
        output_path,
    )



def _anzalone_scaffold_assignments(
    data_root=None,  # noqa: ARG001 — signature matches other builders
) -> list[DatasheetScaffoldAssignment]:
    """Anzalone 2019 used the original SpCas9 pegRNA scaffold (catalog: conventional)."""
    specs = (
        ("anzalone-endo", "hek293t", "pe2"),
        ("anzalone-endo", "hek293t", "pe3"),
    )
    return [
        DatasheetScaffoldAssignment(
            study="anzalone",
            dataset=dataset,
            cell_line=cell_line,
            pe_system=pe_system,
            scaffold_id=SCAFFOLD_ID_CONVENTIONAL,
            scaffold_source="anzalone_2019_conventional",
        )
        for dataset, cell_line, pe_system in specs
    ]


register_study(
    StudyPipeline(
        key="anzalone",
        exporters=(_export_anzalone_endogenous_datasheets,),
        standardizers={
            "anzalone_endo": _standardize_anzalone_endo,
            "anzalone-endo": _standardize_anzalone_endo,
        },
        scaffold_assignments=_anzalone_scaffold_assignments,
    )
)
