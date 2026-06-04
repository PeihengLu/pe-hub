#!/usr/bin/env python3
"""Regenerate testdata/vendor_eval fixtures from a tiny standardized table."""
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PE_DB = REPO / "services" / "pe-db"
import sys

sys.path.insert(0, str(PE_DB))
sys.path.insert(0, str(REPO / "packages" / "pe-common"))

from app.utils.convert_data import (  # noqa: E402
    standardized_to_deepprime_dataframe,
    standardized_to_oped_dataframe,
    standardized_to_pridict_dataframe,
)


def _standardized_df() -> pd.DataFrame:
    seq = ("ACGT" * 25) + ("TGCA" * 25)
    return pd.DataFrame(
        {
            "wt_sequence": [seq, seq[::-1]],
            "mut_sequence": [
                seq[:100] + "A" + seq[101:],
                seq[::-1][:100] + "C" + seq[::-1][101:],
            ],
            "edit_len": [1, 1],
            "type_sub": [True, True],
            "type_ins": [False, False],
            "type_del": [False, False],
            "protospacer_location_l": [50, 52],
            "protospacer_location_r": [70, 72],
            "pbs_location_l": [80, 82],
            "pbs_location_r": [93, 95],
            "rtt_location_l": [93, 95],
            "rtt_location_r": [110, 112],
            "lha_location_r": [100, 102],
            "rha_location_l": [101, 103],
            "rha_location_r": [120, 122],
            "editing_efficiency": [0.3, 0.7],
            "spcas9_score": [0.5, 0.6],
        }
    )


def main() -> None:
    std = _standardized_df()
    out = REPO / "testdata" / "vendor_eval"
    out.mkdir(parents=True, exist_ok=True)
    std.to_csv(out / "standardized_small.csv", index=False)
    standardized_to_deepprime_dataframe(std).to_csv(out / "deepprime_small.csv", index=False)
    standardized_to_oped_dataframe(std).to_csv(out / "oped_native_small.csv", index=False)
    pr = standardized_to_pridict_dataframe(std)
    pr["averageedited"] = std["editing_efficiency"].values
    pr.to_csv(out / "pridict2_small.csv", index=False)
    print(f"Wrote fixtures under {out}")


if __name__ == "__main__":
    main()
