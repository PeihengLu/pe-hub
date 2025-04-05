'''
This module contains code for calculating biological features using 
the raw sequence of a given prime editor guide RNA (pegRNA) sequence.
'''
import pandas as pd
# Calculating Minimum Free Energy (MFE)
import RNA 
# Calculating melting temperature
from Bio.Seq import Seq
from Bio.SeqUtils import MeltingTemp as mt 


def calculate_mfe(sequence: str) -> float:
    pass


def calculate_mt_wallace(sequence: str) -> float:
    """
    Calculate the melting temperature of a given sequence using
    the Wallace method.
    """
    # Convert the sequence to a Bio.Seq object
    seq = Seq(sequence)
    # Calculate the melting temperature
    tm = mt.Tm_Wallace(seq)
    return tm