# read the deepprime excel file and split the sheets
# into multiple csv files
import pandas as pd
from pathlib import Path
import sys

# Use pe_common package and new service structure
try:
    from pe_common import DATA_ROOT
    # Import from new location
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from services.pe_db.app.data_prep.converter import DataConverter
except ImportError:
    # Fallback to old structure
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from src.constants import DATA_ROOT
    from datasets.data import export_deepprime_all

if __name__ == "__main__":
    export_deepprime_all()