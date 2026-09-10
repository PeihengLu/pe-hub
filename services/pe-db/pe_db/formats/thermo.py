"""Thermodynamic helpers shared by PRIDICT and DeepPrime converters."""
from __future__ import annotations

from Bio.Seq import Seq
from Bio.SeqUtils import MeltingTemp as mt

_viennarna = None


def _get_viennarna():
    global _viennarna
    if _viennarna is None:
        import RNA

        _viennarna = RNA
    return _viennarna


def _reverse_complement(seq: str) -> str:
    return str(Seq(seq).reverse_complement())


# DeepPrime biofeat.py: U is mapped to U, not A. Bio.Seq RC turns U→A and
# disagrees with Yu ClinVar tm1/tm4 by several degrees.
_DEEPPRIME_RC = str.maketrans("ACGTacgtnN", "TGCAtgcanN")


def _deepprime_reverse_complement(seq: str) -> str:
    return str(seq).translate(_DEEPPRIME_RC)[::-1]


def _gc_fraction_percent(seq: str) -> float:
    if not seq:
        return 0.0
    gc_count = seq.count("G") + seq.count("C")
    return 100.0 * gc_count / len(seq)


def _tm_nn_or_zero(seq: str, *, nn_table) -> float:
    """BioPython ``Tm_NN`` indexes ``seq[0]`` and crashes on empty oligo."""
    if not seq:
        return 0.0
    try:
        return float(mt.Tm_NN(seq=Seq(seq), nn_table=nn_table))
    except (ValueError, IndexError):
        return 0.0


def _rna_mfe_or_zero(fold, seq: str) -> float:
    if not seq:
        return 0.0
    return float(fold(seq)[1])


def _compute_pridict2_gc_features(pbs_seq: str, rt_seq: str) -> dict[str, float]:
    pbs = pbs_seq.upper()
    rt = rt_seq.upper()
    combined = pbs + rt
    n_gc_pbs = pbs.count("G") + pbs.count("C")
    n_gc_rt = rt.count("G") + rt.count("C")
    n_gc_combined = n_gc_pbs + n_gc_rt
    return {
        "nGCcnt1": float(n_gc_pbs),
        "nGCcnt2": float(n_gc_rt),
        "nGCcnt3": float(n_gc_combined),
        "fGCcont1": _gc_fraction_percent(pbs),
        "fGCcont2": _gc_fraction_percent(rt),
        "fGCcont3": _gc_fraction_percent(combined),
    }


def _compute_pridict2_tm_features(
    wt: str,
    pbs_seq: str,
    rt_seq: str,
    *,
    protospacer_r: int,
    edit_len: int,
    type_sub: bool,
    type_ins: bool,
    type_del: bool,
) -> dict[str, float]:
    pbs = pbs_seq.upper()
    rt = rt_seq.upper()
    n_nick = int(protospacer_r) - 3

    s_for_tm1 = _deepprime_reverse_complement(pbs.replace("A", "U"))
    s_for_tm2 = wt[n_nick:n_nick + len(rt)]

    if type_sub:
        s_for_tm2new = wt[n_nick:n_nick + len(rt)]
        s_tm3_anti = _reverse_complement(wt[n_nick:n_nick + len(rt)])
    elif type_ins:
        s_for_tm2new = wt[n_nick:n_nick + len(rt) - edit_len]
        s_tm3_anti = _reverse_complement(wt[n_nick:n_nick + len(rt) - edit_len])
    elif type_del:
        s_for_tm2new = wt[n_nick:n_nick + len(rt) + edit_len]
        s_tm3_anti = _reverse_complement(wt[n_nick:n_nick + len(rt) + edit_len])
    else:
        s_for_tm2new = s_for_tm2
        s_tm3_anti = _reverse_complement(s_for_tm2)

    s_for_tm3 = [rt, s_tm3_anti]
    s_for_tm4 = [_deepprime_reverse_complement(rt.replace("A", "U")), rt]

    tm1 = _tm_nn_or_zero(s_for_tm1, nn_table=mt.R_DNA_NN1)
    tm2 = _tm_nn_or_zero(s_for_tm2, nn_table=mt.DNA_NN3)
    tm2new = _tm_nn_or_zero(s_for_tm2new, nn_table=mt.DNA_NN3)

    tm3 = 0.0
    for s_seq1, s_seq2 in zip(s_for_tm3[0], s_for_tm3[1]):
        try:
            tm3 = float(mt.Tm_NN(seq=s_seq1, c_seq=s_seq2, nn_table=mt.DNA_NN3))
        except (ValueError, IndexError):
            continue

    tm4 = _tm_nn_or_zero(s_for_tm4[0], nn_table=mt.R_DNA_NN1)
    tmD = tm3 - tm2
    return {
        "Tm1": tm1,
        "Tm2": tm2,
        "Tm2new": tm2new,
        "Tm3": tm3,
        "Tm4": tm4,
        "TmD": tmD,
    }
