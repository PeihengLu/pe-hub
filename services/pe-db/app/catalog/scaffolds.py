"""pegRNA scaffold reference sequences and resolution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Literature / vendor canonical gRNA scaffolds (5'→3', RNA written as DNA alphabet).
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


@dataclass(frozen=True)
class PegRNAScaffold:
    scaffold_id: str
    name: str
    sequence: str
    description: str = ""


PEGRNA_SCAFFOLDS: tuple[PegRNAScaffold, ...] = (
    PegRNAScaffold(
        scaffold_id="conventional",
        name="Conventional",
        sequence=SCAFFOLD_CONVENTIONAL,
        description="Original SpCas9 pegRNA scaffold (Anzalone et al. 2019).",
    ),
    PegRNAScaffold(
        scaffold_id="optimized",
        name="Optimized",
        sequence=SCAFFOLD_OPTIMIZED,
        description="Engineered scaffold (Chen et al. 2021); default in PRIDICT pegRNA design.",
    ),
    PegRNAScaffold(
        scaffold_id="minsepie_18nt",
        name="MinSePIE 18nt insertion",
        sequence=SCAFFOLD_MINSEPIE_18NT,
        description="Custom scaffold for 18-nt insertion libraries (Koeppel et al. 2023 ST6).",
    ),
    PegRNAScaffold(
        scaffold_id="minsepie_codon_variant",
        name="MinSePIE codon variant",
        sequence=SCAFFOLD_MINSEPIE_CODON_VARIANT,
        description="Custom scaffold for barnacle codon-variant libraries (Koeppel et al. 2023 ST6).",
    ),
)

_SCAFFOLD_BY_ID: dict[str, PegRNAScaffold] = {s.scaffold_id: s for s in PEGRNA_SCAFFOLDS}
_SEQUENCE_TO_ID: dict[str, str] = {s.sequence.upper(): s.scaffold_id for s in PEGRNA_SCAFFOLDS}


def scaffold_id_from_sequence(sequence: str) -> Optional[str]:
    """Map an explicit scaffold sequence to a known scaffold_id, if recognized."""
    normalized = str(sequence).strip().upper()
    return _SEQUENCE_TO_ID.get(normalized)


def scaffold_id_from_deepprime_label(label: str) -> str:
    """Map DeepPrime Summary 'Scaffold' column value to scaffold_id."""
    normalized = str(label).strip().lower()
    if normalized == "conventional":
        return "conventional"
    if normalized == "optimized":
        return "optimized"
    raise ValueError(f"Unknown DeepPrime scaffold label: {label!r}")


def default_scaffold_for_pridict() -> str:
    """PRIDICT pegRNA design uses the optimized scaffold for PE2-NGG libraries."""
    return "optimized"


def scaffold_id_for_minsepie_sequence(sequence: str) -> str:
    """Resolve MinSePIE ST6 scaffold sequence to scaffold_id."""
    resolved = scaffold_id_from_sequence(sequence)
    if resolved is not None:
        return resolved
    raise ValueError(f"Unrecognized MinSePIE scaffold sequence: {sequence[:40]}...")


def get_scaffold(scaffold_id: str) -> PegRNAScaffold:
    if scaffold_id not in _SCAFFOLD_BY_ID:
        raise KeyError(f"Unknown scaffold_id: {scaffold_id}")
    return _SCAFFOLD_BY_ID[scaffold_id]
