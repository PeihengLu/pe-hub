"""pegRNA scaffold reference sequences and resolution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Stable catalog IDs (seeded into scaffold.id).
SCAFFOLD_ID_CONVENTIONAL = 1
SCAFFOLD_ID_OPTIMIZED = 2
SCAFFOLD_ID_MINSEPIE_18NT = 3
SCAFFOLD_ID_MINSEPIE_CODON_VARIANT = 4

# Literature canonical gRNA scaffolds (5'→3', RNA written as DNA alphabet).
SCAFFOLD_CONVENTIONAL = (
    "GTTTTAGAGCTAGAAATAGCAAGTTAAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGC"
)
SCAFFOLD_OPTIMIZED = (
    "GTTTCAGAGCTATGCTGGAAACAGCATAGCAAGTTGAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGC"
)
# Koeppel et al. 2023 (MinSePIE) — custom variants in ST6_pegRNAs.
SCAFFOLD_MINSEPIE_18NT = (
    "GTTTAAGAGCTATGCTGGAAACAGCATAGCAAGTTTAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGC"
)
SCAFFOLD_MINSEPIE_CODON_VARIANT = (
    "GTTTAAGAGCTAAGCTGGAAACAGCATAGCAAGTTTAAATAAGGCTAGTCCGTTATCAACTCGAAAGAGTGGCACCGAGTCGGTGC"
)

# Catalog scaffold_id per MinSePIE exported dataset (Koeppel et al. 2023 library types).
MINSEPIE_DATASET_SCAFFOLD_ID: dict[str, int] = {
    "library-insert-set12": SCAFFOLD_ID_CONVENTIONAL,
    "library-insert-18nt": SCAFFOLD_ID_MINSEPIE_18NT,
    "library-insert-codon-variant": SCAFFOLD_ID_MINSEPIE_CODON_VARIANT,
    "library-insert-codon-hek3": SCAFFOLD_ID_MINSEPIE_CODON_VARIANT,
    "library-insert-piggybac": SCAFFOLD_ID_CONVENTIONAL,
}


@dataclass(frozen=True)
class PegRNAScaffold:
    scaffold_id: int
    name: str
    sequence: str
    description: str = ""


PEGRNA_SCAFFOLDS: tuple[PegRNAScaffold, ...] = (
    PegRNAScaffold(
        scaffold_id=SCAFFOLD_ID_CONVENTIONAL,
        name="Conventional",
        sequence=SCAFFOLD_CONVENTIONAL,
        description="Original SpCas9 pegRNA scaffold (Anzalone et al. 2019).",
    ),
    PegRNAScaffold(
        scaffold_id=SCAFFOLD_ID_OPTIMIZED,
        name="Optimized",
        sequence=SCAFFOLD_OPTIMIZED,
        description="Engineered scaffold (Chen et al. 2021); default in PRIDICT pegRNA design.",
    ),
    PegRNAScaffold(
        scaffold_id=SCAFFOLD_ID_MINSEPIE_18NT,
        name="MinSePIE 18nt insertion",
        sequence=SCAFFOLD_MINSEPIE_18NT,
        description="Custom scaffold for 18-nt insertion libraries (Koeppel et al. 2023 ST6).",
    ),
    PegRNAScaffold(
        scaffold_id=SCAFFOLD_ID_MINSEPIE_CODON_VARIANT,
        name="MinSePIE codon variant",
        sequence=SCAFFOLD_MINSEPIE_CODON_VARIANT,
        description="Custom scaffold for barnacle codon-variant libraries (Koeppel et al. 2023 ST6).",
    ),
)

_SCAFFOLD_BY_ID: dict[int, PegRNAScaffold] = {s.scaffold_id: s for s in PEGRNA_SCAFFOLDS}
_SEQUENCE_TO_ID: dict[str, int] = {s.sequence.upper(): s.scaffold_id for s in PEGRNA_SCAFFOLDS}


def scaffold_id_from_sequence(sequence: str) -> Optional[int]:
    """Map an explicit scaffold sequence to a known scaffold_id, if recognized."""
    normalized = str(sequence).strip().upper()
    return _SEQUENCE_TO_ID.get(normalized)


def scaffold_id_from_deepprime_label(label: str) -> int:
    """Map DeepPrime Summary 'Scaffold' column value to scaffold_id."""
    normalized = str(label).strip().lower()
    if normalized == "conventional":
        return SCAFFOLD_ID_CONVENTIONAL
    if normalized == "optimized":
        return SCAFFOLD_ID_OPTIMIZED
    raise ValueError(f"Unknown DeepPrime scaffold label: {label!r}")


def default_scaffold_for_pridict() -> int:
    """PRIDICT pegRNA design uses the optimized scaffold for PE2-NGG libraries."""
    return SCAFFOLD_ID_OPTIMIZED


def scaffold_id_for_minsepie_sequence(sequence: str) -> int:
    """Resolve MinSePIE ST6 scaffold sequence to scaffold_id."""
    resolved = scaffold_id_from_sequence(sequence)
    if resolved is not None:
        return resolved
    raise ValueError(f"Unrecognized MinSePIE scaffold sequence: {sequence[:40]}...")


def get_scaffold(scaffold_id: int) -> PegRNAScaffold:
    if scaffold_id not in _SCAFFOLD_BY_ID:
        raise KeyError(f"Unknown scaffold_id: {scaffold_id}")
    return _SCAFFOLD_BY_ID[scaffold_id]
