"""Biological feature calculation utilities

This module contains code for calculating biological features using 
the raw sequence of a given prime editor guide RNA (pegRNA) sequence.
"""
import pandas as pd
# Calculating Minimum Free Energy (MFE)
import RNA 
# Calculating melting temperature
from Bio.Seq import Seq
from Bio.SeqUtils import MeltingTemp as mt 


def calculate_mfe(sequence: str) -> float:
    """
    Calculate the Minimum Free Energy (MFE) of a given RNA sequence
    using ViennaRNA package.
    
    Args:
        sequence: RNA sequence string
        
    Returns:
        Minimum free energy value in kcal/mol
    """
    # Use ViennaRNA's fold function to calculate MFE
    structure, mfe = RNA.fold(sequence)
    return mfe


def calculate_mt_wallace(sequence: str) -> float:
    """
    Calculate the melting temperature of a given sequence using
    the Wallace method.
    
    Args:
        sequence: DNA/RNA sequence string
        
    Returns:
        Melting temperature in degrees Celsius
    """
    # Convert the sequence to a Bio.Seq object
    seq = Seq(sequence)
    # Calculate the melting temperature
    tm = mt.Tm_Wallace(seq)
    return tm


def calculate_gc_content(sequence: str) -> float:
    """
    Calculate the GC content of a DNA/RNA sequence.
    
    Args:
        sequence: DNA/RNA sequence string
        
    Returns:
        GC content as a fraction (0.0 to 1.0)
    """
    sequence = sequence.upper()
    gc_count = sequence.count('G') + sequence.count('C')
    total_count = len(sequence)
    
    if total_count == 0:
        return 0.0
    
    return gc_count / total_count
