# read the deepprime excel file and split the sheets
# into multiple csv files
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constants import DATA_ROOT
from src.data import export_deepprime_all

if __name__ == "__main__":
    export_deepprime_all()