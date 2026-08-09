#!/usr/bin/env python3
"""Map query DNA sequences onto local chromosome FASTAs (exact + RC match)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional


_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def iter_fasta_records(path: Path) -> Iterator[tuple[str, str]]:
    name: Optional[str] = None
    chunks: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks).upper()
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if name is not None:
        yield name, "".join(chunks).upper()


def find_exact_hits(
    query: str,
    chrom_dir: Path,
    *,
    chrom_glob: str = "chr*.fa",
) -> list[dict]:
    """Return all exact forward/RC hits for ``query`` under ``chrom_dir``."""
    q = query.upper().replace("U", "T")
    rc = reverse_complement(q)
    hits: list[dict] = []
    for path in sorted(chrom_dir.glob(chrom_glob)):
        for header, seq in iter_fasta_records(path):
            chrom = header.replace("chr", "") if header.startswith("chr") else header
            start = 0
            while True:
                idx = seq.find(q, start)
                if idx < 0:
                    break
                hits.append(
                    {
                        "chrom": chrom,
                        "start_0": idx,
                        "end_0": idx + len(q),
                        "assembly_strand": 1,
                        "match": "exact",
                    }
                )
                start = idx + 1
            start = 0
            while True:
                idx = seq.find(rc, start)
                if idx < 0:
                    break
                hits.append(
                    {
                        "chrom": chrom,
                        "start_0": idx,
                        "end_0": idx + len(q),
                        "assembly_strand": -1,
                        "match": "exact_rc",
                    }
                )
                start = idx + 1
    return hits


def unique_hit(query: str, chrom_dir: Path) -> Optional[dict]:
    hits = find_exact_hits(query, chrom_dir)
    if len(hits) == 1:
        return hits[0]
    # Prefer primary chromosomes if multi-hit noise on alt contigs (shouldn't happen with chr*.fa).
    primary = [h for h in hits if h["chrom"] in set(list(map(str, range(1, 23))) + ["X", "Y", "M"])]
    if len(primary) == 1:
        return primary[0]
    return None
