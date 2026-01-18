# src/chop_and_restore_pridict.py
# -*- coding: utf-8 -*-

# chop the pridict data into three files for github
# restore the pridict data from three files after cloning
import pandas as pd
import os
import sys
import argparse
from pathlib import Path

# Use pe_common package for constants
from pe_common import DATA_ROOT

raw_data = 'raw'

def restore_pridict_data():
    """
    Restore the PRIDICT data from three parts after cloning.
    """
    # Load the three parts of the PRIDICT data
    part1 = pd.read_csv(DATA_ROOT / raw_data / 'pridict1' / 'pridict1_library1_part1.csv')
    part2 = pd.read_csv(DATA_ROOT / raw_data / 'pridict1' / 'pridict1_library1_part2.csv')
    part3 = pd.read_csv(DATA_ROOT / raw_data / 'pridict1' / 'pridict1_library1_part3.csv')

    # Concatenate the parts back into a single DataFrame
    restored_data = pd.concat([part1, part2, part3], ignore_index=True)

    # Save the restored data to a single file
    restored_data.to_csv(DATA_ROOT / raw_data / 'pridict1' / 'pridict1_library1.csv', index=False)

if __name__ == "__main__":
    restore_pridict_data()