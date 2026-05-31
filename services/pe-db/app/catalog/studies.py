"""Static study and dataset metadata for the PE Database catalog.

Should be refactored to be more extensible if the study count grows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass(frozen=True)
class StudyRecord:
    key: str
    display_name: str
    publication_date: Optional[date]
    authors: str


@dataclass(frozen=True)
class DatasetRecord:
    """Catalog metadata for a dataset within a study.

    Field semantics:
      - ``pegRNA_delivery_method`` / ``pe_delivery_method``: how each component
        reaches cells (often different, e.g. lentiviral pegRNA + transfected PE).
      - ``edit_scope``: intended vs off-target editing readout (on_target / off_target).
      - ``experimental_method``: where editing is measured (in_vitro / in_vivo).
      - ``target_context``: where the edit is measured — ``endogenous`` (native
        chromosomal locus) vs ``non_endogenous`` (synthetic target on plasmid or
        lentiviral self-targeting cassette; not the patient's genomic allele).
      - ``standardizable``: standardization support level for this dataset.
        ``True`` means full conversion to the shared standardized format.
        ``False`` means partial-support conversion (entry-level fields only).
    """

    study_key: str
    name: str
    description: str
    pegRNA_delivery_method: str
    pe_delivery_method: str
    edit_scope: str
    experimental_method: str
    target_context: str
    standardizable: bool = True


def _canonical_dataset_name(value: str) -> str:
    return str(value).strip().lower().replace("_", "-")


STUDY_REGISTRY: tuple[StudyRecord, ...] = (
    StudyRecord(
        key="deepprime",
        display_name="DeepPrime",
        publication_date=date(2023, 4, 20),
        authors="Yu et al.",
    ),
    StudyRecord(
        key="pridict1",
        display_name="PRIDICT1",
        publication_date=date(2023, 8, 1),
        authors="Mathis et al.",
    ),
    StudyRecord(
        key="pridict2",
        display_name="PRIDICT2",
        publication_date=date(2024, 6, 1),
        authors="Mathis et al.",
    ),
    StudyRecord(
        key="minsepie",
        display_name="MinsePIE",
        publication_date=date(2023, 2, 16),
        authors="Koeppel et al.",
    ),
    StudyRecord(
        key="deeppe",
        display_name="DeepPE",
        publication_date=date(2021, 2, 1),
        authors="Kim et al.",
    ),
)

_DEEPPE_HT_DESCRIPTION = (
    "High-throughput PE2 library 1 (G→C at RTT position +5); lentiviral pegRNA/target "
    "reporters in HEK293T (~43k pegRNAs; HT-training and HT-test splits)."
)

DATASET_REGISTRY: tuple[DatasetRecord, ...] = (
    DatasetRecord(
        study_key="deepprime",
        name="deepprime-clinvar",
        description=(
            "ClinVar-variant pegRNA library (~259k pairs); efficiencies measured on "
            "synthetic 74 bp target contexts (plasmid assay), not native genomic loci."
        ),
        pegRNA_delivery_method="plasmid_transfection",
        pe_delivery_method="plasmid_transfection",
        edit_scope="on_target",
        experimental_method="in_vitro",
        target_context="non_endogenous",
        standardizable=True,
    ),
    DatasetRecord(
        study_key="deepprime",
        name="deepprime-small",
        description=(
            "Multi cell line / PE-system fine-tuning subsets (DeepPrime-FT); "
            "synthetic target contexts."
        ),
        pegRNA_delivery_method="plasmid_transfection",
        pe_delivery_method="plasmid_transfection",
        edit_scope="on_target",
        experimental_method="in_vitro",
        target_context="non_endogenous",
        standardizable=True,
    ),
    DatasetRecord(
        study_key="deepprime",
        name="deepprime-off",
        description=(
            "Off-target mismatch library (DeepPrime-Off); synthetic mismatch reporters."
        ),
        pegRNA_delivery_method="plasmid_transfection",
        pe_delivery_method="plasmid_transfection",
        edit_scope="off_target",
        experimental_method="in_vitro",
        target_context="non_endogenous",
        standardizable=False,
    ),
    DatasetRecord(
        study_key="deepprime",
        name="deepprime-off-subpool",
        description="Off-target library sub-pools; synthetic mismatch reporters.",
        pegRNA_delivery_method="plasmid_transfection",
        pe_delivery_method="plasmid_transfection",
        edit_scope="off_target",
        experimental_method="in_vitro",
        target_context="non_endogenous",
        standardizable=False,
    ),
    DatasetRecord(
        study_key="pridict1",
        name="library1",
        description=(
            "PRIDICT1 self-targeting lentiviral HTS (~92k pegRNAs): each construct "
            "pairs pegRNA with a synthetic target cassette (not the endogenous pathogenic locus)."
        ),
        pegRNA_delivery_method="lentiviral",
        pe_delivery_method="plasmid_transfection",
        edit_scope="on_target",
        experimental_method="in_vitro",
        target_context="non_endogenous",
        standardizable=True,
    ),
    DatasetRecord(
        study_key="pridict1",
        name="library2",
        description=(
            "PRIDICT1 disease-block subscreen (~1.9k pegRNAs): pathogenic-variant "
            "pegRNAs in lentiviral self-targeting reporters across HEK293T, U2OS, "
            "and K562 (PE2 / PEmax; MMR-proficient and K562/U2OS/HEK MLH1−/− lines)."
        ),
        pegRNA_delivery_method="lentiviral",
        pe_delivery_method="plasmid_transfection",
        edit_scope="on_target",
        experimental_method="in_vitro",
        target_context="non_endogenous",
        standardizable=True,
    ),
    DatasetRecord(
        study_key="pridict1",
        name="library2-invivo",
        description=(
            "Disease-block subscreen pegRNAs validated in GFP+ sorted mouse liver "
            "(adenoviral pegRNA and PE2 delivery; endogenous liver loci)."
        ),
        pegRNA_delivery_method="adenovirus",
        pe_delivery_method="adenovirus",
        edit_scope="on_target",
        experimental_method="in_vivo",
        target_context="endogenous",
        standardizable=True,
    ),
    DatasetRecord(
        study_key="pridict1",
        name="endogenous",
        description=(
            "Arrayed pegRNA validation at ~45 native genomic loci in HEK293T and K562 "
            "(supplementary endogenous table; PE2 plasmid transfection)."
        ),
        pegRNA_delivery_method="plasmid_transfection",
        pe_delivery_method="plasmid_transfection",
        edit_scope="on_target",
        experimental_method="in_vitro",
        target_context="endogenous",
        standardizable=False,
    ),
    DatasetRecord(
        study_key="pridict2",
        name="library-diverse",
        description=(
            "PRIDICT2 diverse edit-type library (HEK293T, K562, K562 MLH1−/−); "
            "same self-targeting lentiviral reporter format as PRIDICT1."
        ),
        pegRNA_delivery_method="lentiviral",
        pe_delivery_method="plasmid_transfection",
        edit_scope="on_target",
        experimental_method="in_vitro",
        target_context="non_endogenous",
        standardizable=True,
    ),
    DatasetRecord(
        study_key="pridict2",
        name="library-diverse-invivo",
        description=(
            "Diverse-library pegRNAs validated at endogenous mouse liver loci "
            "(PRIDICT2 supplementary AdV column; distinct from in vitro reporter HTS)."
        ),
        pegRNA_delivery_method="adenovirus",
        pe_delivery_method="adenovirus",
        edit_scope="on_target",
        experimental_method="in_vivo",
        target_context="endogenous",
        standardizable=True,
    ),
    DatasetRecord(
        study_key="pridict2",
        name="trip-analysis",
        description=(
            "TRIP library editing survey at endogenous genomic integrations in K562 "
            "(supplementary table 12; Fig. 3 / Ext. Fig. 4–5; PE2)."
        ),
        pegRNA_delivery_method="lentiviral",
        pe_delivery_method="plasmid_transfection",
        edit_scope="on_target",
        experimental_method="in_vitro",
        target_context="endogenous",
        standardizable=False,
    ),
    DatasetRecord(
        study_key="minsepie",
        name="library-insert",
        description=(
            "Pooled insertion-efficiency library at endogenous loci (HEK293T, rc): "
            "lentiviral pegRNA library with PE2/PE3 or epegRNA by plasmid transfection."
        ),
        pegRNA_delivery_method="lentiviral",
        pe_delivery_method="plasmid_transfection",
        edit_scope="on_target",
        experimental_method="in_vitro",
        target_context="endogenous",
        standardizable=True,
    ),
    DatasetRecord(
        study_key="deeppe",
        name="deeppe-ht",
        description=_DEEPPE_HT_DESCRIPTION,
        pegRNA_delivery_method="lentiviral",
        pe_delivery_method="plasmid_transfection",
        edit_scope="on_target",
        experimental_method="in_vitro",
        target_context="non_endogenous",
        standardizable=True,
    ),
    DatasetRecord(
        study_key="deeppe",
        name="deeppe-type",
        description=(
            "DeepPE library 2 edit-type subset (lentiviral reporters in HEK293T; "
            "type-training and type-test splits)."
        ),
        pegRNA_delivery_method="lentiviral",
        pe_delivery_method="plasmid_transfection",
        edit_scope="on_target",
        experimental_method="in_vitro",
        target_context="non_endogenous",
        standardizable=True,
    ),
    DatasetRecord(
        study_key="deeppe",
        name="deeppe-position",
        description=(
            "DeepPE library 2 edit-position subset (lentiviral reporters in HEK293T; "
            "position-training and position-test splits)."
        ),
        pegRNA_delivery_method="lentiviral",
        pe_delivery_method="plasmid_transfection",
        edit_scope="on_target",
        experimental_method="in_vitro",
        target_context="non_endogenous",
        standardizable=True,
    ),
    DatasetRecord(
        study_key="deeppe",
        name="deeppe-endo",
        description=(
            "Arrayed pegRNA validation at endogenous genomic loci (PE2 plasmid "
            "transfection); editing_efficiency averaged over BR1–BR2 and TR1–TR2."
        ),
        pegRNA_delivery_method="plasmid_transfection",
        pe_delivery_method="plasmid_transfection",
        edit_scope="on_target",
        experimental_method="in_vitro",
        target_context="endogenous",
        standardizable=True,
    ),
    DatasetRecord(
        study_key="minsepie",
        name="library-insert-piggybac",
        description=(
            "Same lentiviral pegRNA library in HAP1 and HAP1 ΔMLH1 with dox-inducible "
            "PE2 integrated via PiggyBac (Koeppel et al. 2023)."
        ),
        pegRNA_delivery_method="lentiviral",
        pe_delivery_method="piggybac",
        edit_scope="on_target",
        experimental_method="in_vitro",
        target_context="endogenous",
        standardizable=True,
    ),
)

_STUDY_BY_KEY = {study.key: study for study in STUDY_REGISTRY}
_DATASET_BY_KEY = {
    (d.study_key, _canonical_dataset_name(d.name)): d for d in DATASET_REGISTRY
}


def get_study_record(study_key: str) -> Optional[StudyRecord]:
    return _STUDY_BY_KEY.get(study_key.strip().lower())


def get_dataset_record(study_key: str, dataset_name: str) -> Optional[DatasetRecord]:
    return _DATASET_BY_KEY.get(
        (study_key.strip().lower(), _canonical_dataset_name(dataset_name))
    )
