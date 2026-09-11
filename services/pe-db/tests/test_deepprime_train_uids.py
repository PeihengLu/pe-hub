"""DeepPrime Excel train UIDs include mixed train/Test spacers."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "packages" / "pe-common"))

from pe_common.data_utils import compute_target_uid  # noqa: E402
from pe_db.studies.deepprime import _author_train_target_uids_from_deepprime_frame  # noqa: E402


def test_author_train_uids_include_spacers_that_also_have_test_rows():
    shared = "ACGTACGTACGTACGTACGT"
    only_test = "TGCATGCATGCATGCATGCA"
    df = pd.DataFrame(
        {
            "wt_sequence": [
                "AAAA" + shared + "N" * 50,
                "AAAA" + shared + "N" * 50,
                "CCCC" + only_test + "N" * 50,
            ],
            "fold": [1, "Test", "Test"],
        }
    )
    uids = _author_train_target_uids_from_deepprime_frame(df)
    assert uids == {compute_target_uid(shared)}
