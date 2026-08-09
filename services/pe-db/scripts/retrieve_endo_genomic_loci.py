#!/usr/bin/env python3
"""Retrieve endogenous genomic loci JSON for DeepPE and PRIDICT1 library2-invivo.

Writes curated JSON consumed at standardize time (no live network during standardize):

  datasets/raw/deeppe/deeppe_genomic_loci.json
  datasets/raw/pridict1/pridict1_library2_genomic_loci.json

Preferred method: exact (+ RC) search against local chromosome FASTAs under
``datasets/reference/hg38_chroms`` (and ``mm39_chroms`` for mouse). Human ClinVar
HGVS names are resolved via Ensembl VEP.

Usage (from repo root)::

  PYTHONPATH=packages/pe-common:services/pe-db \\
    python services/pe-db/scripts/retrieve_endo_genomic_loci.py --all
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import pandas as pd
import requests

from pe_common.constants import DATA_ROOT, PROJECT_ROOT

from scripts.local_genome_map import iter_fasta_records, reverse_complement

logger = logging.getLogger(__name__)

ENSEMBL_VEP = "https://rest.ensembl.org/vep"
DEEPPE_SPACER_OFFSET = 4
DEEPPE_SPACER_LEN = 20
HG38_CHROM_DIR = PROJECT_ROOT / "datasets" / "reference" / "hg38_chroms"
MM39_CHROM_DIR = PROJECT_ROOT / "datasets" / "reference" / "mm39_chroms"


def _batch_exact_search(
    queries: dict[str, str],
    chrom_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    prepared = {qid: seq.upper().replace("U", "T") for qid, seq in queries.items()}
    rcs = {qid: reverse_complement(seq) for qid, seq in prepared.items()}
    hits: dict[str, list[dict[str, Any]]] = {qid: [] for qid in prepared}
    paths = sorted(chrom_dir.glob("chr*.fa"))
    if not paths:
        raise FileNotFoundError(f"No chr*.fa under {chrom_dir}")
    for path in paths:
        logger.info("scan %s", path.name)
        for header, seq in iter_fasta_records(path):
            chrom = header[3:] if header.startswith("chr") else header
            for qid, query in prepared.items():
                for strand, needle, label in (
                    (1, query, "exact"),
                    (-1, rcs[qid], "exact_rc"),
                ):
                    start = 0
                    while True:
                        idx = seq.find(needle, start)
                        if idx < 0:
                            break
                        hits[qid].append(
                            {
                                "chrom": chrom,
                                "start_0": idx,
                                "end_0": idx + len(needle),
                                "assembly_strand": strand,
                                "match": label,
                            }
                        )
                        start = idx + 1
    return hits


def _unique_primary(hits: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    primary = {
        *map(str, range(1, 23)),
        "X",
        "Y",
        "M",
    }
    filtered = [h for h in hits if h["chrom"] in primary]
    if len(filtered) == 1:
        return filtered[0]
    if len(hits) == 1:
        return hits[0]
    return None


def retrieve_deeppe_loci() -> dict[str, Any]:
    """Exact-map unique DeepPE endo wide targets (or spacer+PAM) onto local hg38."""
    exported = DATA_ROOT / "exported" / "deeppe" / "deeppe-endo"
    sequences: set[str] = set()
    for path in sorted(exported.glob("*.csv")):
        df = pd.read_csv(path)
        col = next(c for c in df.columns if "Wide target" in c)
        sequences.update(df[col].astype(str).str.upper().str.replace("U", "T"))
    ordered = sorted(s for s in sequences if re.fullmatch(r"[ACGT]+", s) and len(s) == 47)
    logger.info("DeepPE unique wide targets: %s", len(ordered))

    queries = {f"dp{i}": seq for i, seq in enumerate(ordered)}
    hit_map = _batch_exact_search(queries, HG38_CHROM_DIR)
    loci: dict[str, Any] = {}
    retry: dict[str, str] = {}
    for i, seq in enumerate(ordered):
        hit = _unique_primary(hit_map[f"dp{i}"])
        if hit is None:
            retry[seq] = seq[DEEPPE_SPACER_OFFSET : DEEPPE_SPACER_OFFSET + 23]
            continue
        if hit["assembly_strand"] == 1:
            spacer_start_0 = hit["start_0"] + DEEPPE_SPACER_OFFSET
        else:
            coords = [
                hit["end_0"] - 1 - j
                for j in range(DEEPPE_SPACER_OFFSET, DEEPPE_SPACER_OFFSET + DEEPPE_SPACER_LEN)
            ]
            spacer_start_0 = min(coords)
        loci[seq] = {
            "chrom": hit["chrom"],
            "spacer_start": spacer_start_0 + 1,
            "assembly_strand": hit["assembly_strand"],
            "wide_start": hit["start_0"] + 1,
            "wide_end": hit["end_0"],
            "match": hit["match"],
        }

    if retry:
        logger.info("Retry %s DeepPE sites with spacer+PAM", len(retry))
        rq = {f"r{i}": s for i, s in enumerate(retry.values())}
        rseq = list(retry.keys())
        rh = _batch_exact_search(rq, HG38_CHROM_DIR)
        for i, wide in enumerate(rseq):
            hit = _unique_primary(rh[f"r{i}"])
            if hit is None:
                logger.warning("Unmapped DeepPE wide %s…", wide[:24])
                continue
            spacer_start_0 = hit["start_0"]
            if hit["assembly_strand"] == -1:
                # query = spacer+PAM; RC places PAM at the low genomic end.
                spacer_start_0 = hit["start_0"] + 3
            loci[wide] = {
                "chrom": hit["chrom"],
                "spacer_start": spacer_start_0 + 1,
                "assembly_strand": hit["assembly_strand"],
                "match": hit["match"] + "_spacer_pam",
            }

    return {
        "_comment": (
            "GRCh38/hg38 protospacer anchors for DeepPE endogenous validation wide "
            "targets (Kim et al. 2021). Keys are uppercased 47 bp wide-target sequences. "
            "spacer_start is 1-based; protospacer is positions [4,24) within the wide window."
        ),
        "genome_build": "hg38",
        "coord_ref": "protospacer",
        "spacer_offset": DEEPPE_SPACER_OFFSET,
        "loci": loci,
    }


def _parse_clinvar_hgvs(name: str) -> Optional[str]:
    m = re.match(
        r"^(NM_\d+\.\d+)(?:\([^)]+\))?:(c\.[^ \t]+)",
        str(name).strip(),
    )
    if not m:
        return None
    return f"{m.group(1)}:{m.group(2)}"


def _ensembl_vep_hgvs(hgvs: str) -> Optional[dict[str, Any]]:
    url = f"{ENSEMBL_VEP}/human/hgvs/{quote(hgvs, safe='')}"
    r = requests.get(
        url,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        timeout=60,
    )
    if not r.ok:
        logger.warning("VEP %s -> HTTP %s", hgvs, r.status_code)
        return None
    payload = r.json()
    if not payload:
        return None
    row = payload[0]
    chrom = row.get("seq_region_name")
    start = row.get("start")
    end = row.get("end")
    strand = row.get("strand")
    if chrom is None or start is None or end is None:
        return None
    return {
        "chrom": str(chrom),
        "start_1based": int(start),
        "end_1based": int(end),
        "assembly_strand": int(strand) if strand in (-1, 1) else None,
    }


def retrieve_pridict1_library2_loci(*, sleep_between: float = 0.25) -> dict[str, Any]:
    """Resolve library2-invivo Name → genomic coords (VEP + local genome)."""
    path = DATA_ROOT / "exported" / "pridict1" / "library2-invivo" / "liver_gfpplus-pe2.csv"
    df = pd.read_csv(path)
    uniq = (
        df[["Name", "Gene", "protospacer", "wide_initial_target"]]
        .drop_duplicates(subset=["Name"])
        .reset_index(drop=True)
    )
    loci: dict[str, Any] = {}

    human = uniq[
        ~uniq["Gene"].astype(str).str.endswith("_mouse")
        & ~uniq["Name"].astype(str).str.endswith("_mouse")
    ]
    for i, row in human.iterrows():
        name = str(row["Name"])
        gene = str(row["Gene"])
        hgvs = _parse_clinvar_hgvs(name)
        if hgvs is None:
            continue
        logger.info("[%s/%s] VEP %s", i + 1, len(human), name[:60])
        vep = _ensembl_vep_hgvs(hgvs)
        time.sleep(sleep_between)
        if vep is None:
            continue
        loci[name] = {
            "genome_build": "hg38",
            "chrom": vep["chrom"],
            "variant_start": vep["start_1based"],
            "variant_end": vep["end_1based"],
            "assembly_strand": vep["assembly_strand"],
            "coord_ref": "variant",
            "gene": gene,
            "hgvs": hgvs,
            "match": "ensembl_vep",
        }

    # Non-HGVS human names + any still-missing: local hg38 protospacer search.
    missing_human = human[~human["Name"].astype(str).isin(loci)]
    if len(missing_human) and HG38_CHROM_DIR.exists():
        queries = {
            str(r.Name): str(r.protospacer).upper().replace("U", "T")
            for r in missing_human.itertuples(index=False)
            if re.fullmatch(r"[ACGT]+", str(r.protospacer).upper().replace("U", "T") or "")
        }
        logger.info("Local hg38 search for %s non-HGVS / unresolved names", len(queries))
        hit_map = _batch_exact_search(queries, HG38_CHROM_DIR)
        for row in missing_human.itertuples(index=False):
            name = str(row.Name)
            gene = str(row.Gene)
            hit = _unique_primary(hit_map.get(name) or [])
            if hit is None:
                continue
            loci[name] = {
                "genome_build": "hg38",
                "chrom": hit["chrom"],
                "spacer_start": hit["start_0"] + 1,
                "assembly_strand": hit["assembly_strand"],
                "coord_ref": "protospacer",
                "gene": gene,
                "match": hit["match"],
            }

    mouse = uniq[
        uniq["Gene"].astype(str).str.endswith("_mouse")
        | uniq["Name"].astype(str).str.endswith("_mouse")
    ]
    if len(mouse) and MM39_CHROM_DIR.exists():
        queries = {
            str(r.Name): str(r.protospacer).upper().replace("U", "T")
            for r in mouse.itertuples(index=False)
            if re.fullmatch(r"[ACGT]+", str(r.protospacer).upper().replace("U", "T") or "")
        }
        logger.info("Local mm39 search for %s mouse names", len(queries))
        hit_map = _batch_exact_search(queries, MM39_CHROM_DIR)
        for row in mouse.itertuples(index=False):
            name = str(row.Name)
            gene = str(row.Gene)
            hit = _unique_primary(hit_map.get(name) or [])
            if hit is None:
                logger.warning("Unmapped mouse locus %s", name)
                continue
            loci[name] = {
                "genome_build": "mm39",
                "chrom": hit["chrom"],
                "spacer_start": hit["start_0"] + 1,
                "assembly_strand": hit["assembly_strand"],
                "coord_ref": "protospacer",
                "gene": gene,
                "match": hit["match"],
            }
    elif len(mouse):
        logger.warning("mm39 chrom dir missing (%s); mouse loci left unmapped", MM39_CHROM_DIR)

    return {
        "_comment": (
            "Genomic anchors for PRIDICT1 library2-invivo pegRNAs (Mathis et al.). "
            "Keys are the exported Name field. Human ClinVar-style names use Ensembl VEP "
            "(variant interval, hg38). Other human names use local hg38 protospacer match; "
            "mouse orthologs use local mm39 when available."
        ),
        "loci": loci,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s (%s loci)", path, len(payload.get("loci", {})))


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deeppe", action="store_true")
    parser.add_argument("--pridict1-library2", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args(argv)
    if not (args.deeppe or args.pridict1_library2 or args.all):
        parser.error("Pass --deeppe, --pridict1-library2, and/or --all")

    if args.all or args.deeppe:
        _write_json(DATA_ROOT / "raw" / "deeppe" / "deeppe_genomic_loci.json", retrieve_deeppe_loci())
    if args.all or args.pridict1_library2:
        _write_json(
            DATA_ROOT / "raw" / "pridict1" / "pridict1_library2_genomic_loci.json",
            retrieve_pridict1_library2_loci(),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
