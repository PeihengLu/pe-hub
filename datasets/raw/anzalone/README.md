# Anzalone et al. 2019 (Easy-Prime reformatted)

Prime editing efficiencies at endogenous loci from Anzalone et al., *Nature* 2019
(doi:10.1038/s41586-019-1711-4), reanalyzed and reformatted by Li et al. (Easy-Prime;
doi:10.1186/s13059-021-02458-0).

Source files (from https://github.com/YichaoOU/easy_prime and Zenodo 5137926):

| File | Upstream path | Role |
|------|---------------|------|
| `anzalone_raw_table.csv` | `figure_rep/raw_table_XY_for_training.csv` | Per-replicate amplicon, pegRNA, PBS/RTT, efficiency |
| `anzalone_feature_matrix.csv` | `PE_data_collection/Anzalone_2019.feature_matrix.csv` | Replicate-averaged designs; PE2 vs PE3 via `nick_to_pegRNA` |

PE2 = rows with null `nick_to_pegRNA` (n=199); PE3 = non-null (n=278). This is the
external validation set used by Easy-Prime, PRIDICT, and DeepPrime (e.g. Yu et al.
Cell 2023 Fig. 3H). PE-DB exports DeepPrime-native 74 bp wide targets from the
Easy-Prime `reference_amplicon` (not a 47 bp DeepPE crop).

For OPED, standardization expands those PE-core rows onto 350 bp sense-oriented
hg38 windows (165 nt upstream of the spacer + 20 + 165 down; Liu et al. Nat. Mach.
Intell. 2023 Fig. 2i / Methods BLAT protocol). Cached windows live in
`anzalone_genomic_loci.json` (built by exact spacer match on
`datasets/reference/hg38_chroms`). The OPED converter then passes the **full**
standardized WT (no further 47 bp crop and no invented A/N flanks).
