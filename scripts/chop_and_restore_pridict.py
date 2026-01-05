# src/chop_and_restore_pridict.py
# -*- coding: utf-8 -*-

# chop the pridict data into three files for github
# restore the pridict data from three files after cloning
import pandas as pd
import os
import sys
import argparse
from pathlib import Path

# Add the src directory to the Python path
script_dir = Path(__file__).parent
src_dir = script_dir.parent
sys.path.insert(0, str(src_dir))

from src.constants import DATA_ROOT

endogenetic_data = 'endogenetic'
library_study = 'library-study'

def chop_pridict_data():
    """
    Chop the PRIDICT data into three files for GitHub.
    """
    # Load the original PRIDICT data
    pridict_data = pd.read_csv(DATA_ROOT / library_study / 'pridict1-org' / 'pridict1-org.csv')

    # Split the data into three parts
    part1 = pridict_data.iloc[:len(pridict_data)//3]
    part2 = pridict_data.iloc[len(pridict_data)//3:2*len(pridict_data)//3]
    part3 = pridict_data.iloc[2*len(pridict_data)//3:]

    # Save the parts to separate files
    part1.to_csv(DATA_ROOT / library_study / 'pridict1-org' / 'pridict1-library1_part1.csv', index=False)
    part2.to_csv(DATA_ROOT / library_study / 'pridict1-org' / 'pridict1-library1_part2.csv', index=False)
    part3.to_csv(DATA_ROOT / library_study / 'pridict1-org' / 'pridict1-library1_part3.csv', index=False)

def restore_pridict_data():
    """
    Restore the PRIDICT data from three parts after cloning.
    """
    # Load the three parts of the PRIDICT data
    part1 = pd.read_csv(DATA_ROOT / library_study / 'pridict1-org' / 'pridict1-library1_part1.csv')
    part2 = pd.read_csv(DATA_ROOT / library_study / 'pridict1-org' / 'pridict1-library1_part2.csv')
    part3 = pd.read_csv(DATA_ROOT / library_study / 'pridict1-org' / 'pridict1-library1_part3.csv')

    # Concatenate the parts back into a single DataFrame
    restored_data = pd.concat([part1, part2, part3], ignore_index=True)

    # Save the restored data to a single file
    restored_data.to_csv(DATA_ROOT / library_study / 'pridict1-org' / 'pridict1_library1-org.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chop or restore PRIDICT data.")
    parser.add_argument(
        '-a', '--action', choices=['chop', 'restore'], required=True, 
        help="Action to perform: 'chop' or 'restore'")

    args = parser.parse_args()

    if args.action == 'chop':
        chop_pridict_data()
        print("PRIDICT data has been chopped into three parts.")
    elif args.action == 'restore':
        restore_pridict_data()
        print("PRIDICT data has been restored from three parts.")