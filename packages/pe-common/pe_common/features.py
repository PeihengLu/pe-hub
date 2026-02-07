"""Biological feature calculation utilities

This module contains code for calculating biological features using
the raw sequence of a given prime editor guide RNA (pegRNA) sequence.

If the function name does not specify RNA/DNA, then both sequence
types are supported
"""

import os, sys
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np

# Calculating Minimum Free Energy (MFE)
import RNA

# Calculating melting temperature
from Bio.Seq import Seq  
from Bio.SeqUtils import MeltingTemp as mt 
import tensorflow as tf

from sequence_utils import onehot_encode
from constants import DEEPSPCAS9_MODEL_DIR, DEVICE
from deepspcas9 import _calculate_DeepSpCas9_score

# ---------- Lightweight helpers (fast, vectorized) ----------

# used by pridict2
def _wallace_tm_fast(seq: str) -> float:
    """Wallace rule: 2 * (A+T) + 4 * (G+C). Accepts RNA or DNA; treat U as T."""
    if not seq:
        return 0.0
    s = seq.upper().replace("U", "T")
    a = s.count("A")
    t = s.count("T")
    g = s.count("G")
    c = s.count("C")
    return 2.0 * (a + t) + 4.0 * (g + c)

def _gc_fraction_series(series: pd.Series) -> pd.Series:
    """Vectorized GC fraction for a pandas Series of sequences."""
    s = series.fillna("").astype(str).str.upper()
    # count G and C
    gc_count = s.str.count("G") + s.str.count("C")
    length = s.str.len().replace(0, np.nan)  # avoid divide-by-zero
    gc_fraction = (gc_count / length).fillna(0.0)
    return gc_fraction

# ---------- BioPython features ----------

def _tm_nn_biopython(seq: str, Na=50, Mg=1.5, dNTPs=0.0, DNA_conc=0.5e-6) -> float:
    """Calculate melting temperature using Biopython's nearest-neighbor method.
    Parameters:
        seq: DNA or RNA sequence (U treated as T)
        Na: Sodium ion concentration in mM
        Mg: Magnesium ion concentration in mM
        dNTPs: dNTP concentration in mM
        DNA_conc: DNA concentration in M (default 0.5 µM)
    Returns:
        Melting temperature in Celsius.
    """
    if not seq:
        return 0.0
    s = seq.upper().replace("U", "T")
    try:
        tm = mt.Tm_NN(
            Seq(s),
            nn_table=mt.DNA_NN4,
            Na=Na,
            Mg=Mg,
            dNTPs=dNTPs,
            dnac1=DNA_conc,
        )
        return float(tm)
    except Exception:
        return float("nan")

# ---------- parallel MFE (ViennaRNA) ----------

def _mfe_worker(seq: str) -> float:
    """Worker that calls RNA.fold. Put minimal imports here if using ProcessPoolExecutor."""
    # ViennaRNA's RNA.fold returns (structure, mfe)
    # ensure sequence is uppercase, and convert T -> U for RNA folding if needed
    if not seq:
        return 0.0
    s = seq.upper().replace("T", "U")
    try:
        struct, mfe = RNA.fold(s)
        return float(mfe)
    except Exception:
        # fallback if ViennaRNA can't fold (return 0 or np.nan depending on your preference)
        return float("nan")

def batch_rna_mfe(seqs: List[str], max_workers: Optional[int] = None) -> List[float]:
    """
    Compute MFE for a list of sequences in parallel using ProcessPoolExecutor.
    Returns a list aligned with seqs.
    """
    if len(seqs) == 0:
        return []
    # short-circuit single-thread performance
    try:
        workers = max_workers or max(1, (os.cpu_count() or 1) - 1)
        results: List[float] = [0.0] * len(seqs)
        with ProcessPoolExecutor(max_workers=workers) as exe:
            # submit tasks with their index so we can restore order
            future_to_idx = {exe.submit(_mfe_worker, seqs[i]): i for i in range(len(seqs))}
            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                try:
                    results[i] = future.result()
                except Exception:
                    results[i] = float("nan")
        return results
    except Exception:
        # fallback to serial if processpool not available or fails (e.g., Windows pickling issues)
        return [_mfe_worker(s) for s in seqs]

# ---------- Main batch function ----------

def batch_calculate_features(
    df: pd.DataFrame,
    pbs_col="pbs_sequence",
    rtt_col="rtt_sequence",
    lha_col="lha_sequence",
    rha_col="rha_sequence",
    target_col="target_sequence",
    pam_col="pam_location",
    mfe_workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    Input DataFrame must have columns for PBS, RTT, LHA, RHA, target sequence, and PAM location (1-based index).
    Returns a DataFrame with features appended:
      *_rna_mfe, *_tm_wallace, *_gc_content, target_deepspcas9
    """
    # copy to avoid mutating input
    df_in: pd.DataFrame = df.copy().reset_index(drop=True)

    # Ensure sequences are strings
    for col in (pbs_col, rtt_col, lha_col, rha_col, target_col):
        if col in df_in.columns:
            df_in[col] = df_in[col].fillna("").astype(str)
        else:
            df_in[col] = ""

    # 1) GC content (vectorized)
    df_in["pbs_gc_content"] = _gc_fraction_series(df_in[pbs_col])
    df_in["rtt_gc_content"] = _gc_fraction_series(df_in[rtt_col])
    df_in["lha_gc_content"] = _gc_fraction_series(df_in[lha_col])
    df_in["rha_gc_content"] = _gc_fraction_series(df_in[rha_col])
    df_in["target_gc_content"] = _gc_fraction_series(df_in[target_col])

    # 2) Wallace Tm (vectorized via Python list comprehension is fine; cheap)
    df_in["pbs_tm_wallace"] = df_in[pbs_col].apply(_wallace_tm_fast)
    df_in["rtt_tm_wallace"] = df_in[rtt_col].apply(_wallace_tm_fast)
    df_in["lha_tm_wallace"] = df_in[lha_col].apply(_wallace_tm_fast)
    df_in["rha_tm_wallace"] = df_in[rha_col].apply(_wallace_tm_fast)
    df_in["target_tm_wallace"] = df_in[target_col].apply(_wallace_tm_fast)

    # 3) RNA MFE (parallel)
    # Build lists for each region; you can run them in parallel per-region or combined for fewer executor churns
    # We'll compute MFE per region in parallel batches to limit memory/cpu pressure.
    df_in["pbs_rna_mfe"] = batch_rna_mfe(df_in[pbs_col].tolist(), max_workers=mfe_workers)
    df_in["rtt_rna_mfe"] = batch_rna_mfe(df_in[rtt_col].tolist(), max_workers=mfe_workers)
    df_in["lha_rna_mfe"] = batch_rna_mfe(df_in[lha_col].tolist(), max_workers=mfe_workers)
    df_in["rha_rna_mfe"] = batch_rna_mfe(df_in[rha_col].tolist(), max_workers=mfe_workers)
    df_in["target_rna_mfe"] = batch_rna_mfe(df_in[target_col].tolist(), max_workers=mfe_workers)

    # 4) Length of each region
    df_in["pbs_length"] = df_in[pbs_col].str.len().fillna
    df_in["rtt_length"] = df_in[rtt_col].str.len().fillna(0)
    df_in["lha_length"] = df_in[lha_col].str.len().fillna
    df_in["rha_length"] = df_in[rha_col].str.len().fillna(0)

    # ) DeepSpCas9 score for target (if pam_location valid)
    # The DeepSpCas9 expects a 30-nt string: [-4] + 20nt guide + PAM + +3bp
    def _extract_30nt(row: pd.Series) -> Optional[str]:
        t = row[target_col] or ""
        pam = row.get(pam_col, None)
        if pam is None or not isinstance(pam, (int, float)) or np.isnan(pam):
            return None
        pam = int(pam)
        # pam_location is 1-based in your code; slice in Python is 0-based
        start = pam - 17  # pam - 17  (as in original)
        end = pam + 13    # pam + 13 (exclusive index)
        if start < 0 or end > len(t) or (end - start) != 30:
            return None
        return t[start:end]

    df_in["_deepspcas9_target30"] = df_in.apply(_extract_30nt, axis=1)  # type: ignore[call-overload]

    # collect list of sequences where we can compute the score
    idx_to_seq = [(i, s) for i, s in enumerate(df_in["_deepspcas9_target30"].tolist()) if s]
    if idx_to_seq:
        seqs = [s for _, s in idx_to_seq]
        # compute scores in batch - using your function that loads the TF model once
        try:
            scores = _calculate_DeepSpCas9_score(seqs)  # returns list aligned to seqs
        except Exception as e:
            # if TF model fails for some reason, set NaN and continue
            scores = [float("nan")] * len(seqs)
        # assign back to proper indices
        for (i, _), sc in zip(idx_to_seq, scores):
            df_in.loc[i, "deepspcas9_score"] = sc
    else:
        df_in["deepspcas9_score"] = np.nan

    # cleanup intermediate column
    df_in = df_in.drop(columns=["_deepspcas9_target30"], errors="ignore")

    return df_in


# ---------- Prepare the data from standardized format ----------
def calculate_features_from_standardized_df(
    df: pd.DataFrame,
    mfe_workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    Wrapper to calculate features from a standardized pegRNA DataFrame.
    Expects columns: 'pbs_sequence', 'rtt_sequence', 'lha_sequence', 'rha_sequence', 'target_sequence', 'pam_location'
    """
    return batch_calculate_features(
        df,
        pbs_col="pbs_sequence",
        rtt_col="rtt_sequence",
        lha_col="lha_sequence",
        rha_col="rha_sequence",
        target_col="target_sequence",
        pam_col="pam_location",
        mfe_workers=mfe_workers,
    )