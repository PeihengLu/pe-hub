import pandas as pd
from pe_ensemble.models import optiprime_vendor_provenance as provenance


def test_pooled_clinvar_includes_deepprime_test_fold(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "_STANDARDIZED", tmp_path)
    for spec in provenance._DATASET_SPECS:
        folder = tmp_path / spec.study / spec.dataset
        folder.mkdir(parents=True)
        pd.DataFrame({"target_uid": [f"{spec.dataset}-train", f"{spec.dataset}-test"],
                      "original_fold": [0, -1]}).to_parquet(folder / "sheet.parquet")
    loci, _ = provenance._collect_loci()
    assert "deepprime_clinvar-test" in loci
    assert "deepprime_clinvar-train" in loci
