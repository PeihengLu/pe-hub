"""Tests for formatted model-format disk cache."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.converter import DataConverter
from app.formatted_cache import (
    clear_formatted_cache,
    formatted_cache_path,
    load_formatted_cache,
    save_formatted_cache,
)
from app.utils.convert_data import standardized_to_oped_dataframe


@pytest.fixture()
def datasets_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("PE_PROJECT_ROOT", str(tmp_path))
    root = tmp_path / "datasets"
    root.mkdir()
    return root


def _sample_standardized() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "wt_sequence": ["ACGT" * 30, "TGCA" * 30],
            "mut_sequence": ["ACGT" * 30, "TGCA" * 30],
            "edit_len": [1, 3],
            "type_sub": [True, False],
            "type_ins": [False, False],
            "type_del": [False, True],
            "protospacer_location_l": [40, 40],
            "protospacer_location_r": [60, 60],
            "pbs_location_l": [70, 70],
            "pbs_location_r": [83, 83],
            "rtt_location_l": [83, 83],
            "rtt_location_r": [100, 103],
            "lha_location_r": [50, 50],
            "rha_location_l": [100, 100],
            "rha_location_r": [103, 103],
            "editing_efficiency": [0.5, 0.7],
        }
    )


def test_formatted_cache_path_layout(datasets_dir: Path):
    path = formatted_cache_path(
        "oped",
        "PRIDICT1",
        "library-1",
        "HEK293T",
        "PE2",
        datasets_dir=datasets_dir,
    )
    assert path == datasets_dir / "formatted" / "oped" / "pridict1" / "library_1" / "hek293t-pe2.parquet"


def test_save_and_load_formatted_cache(datasets_dir: Path):
    df = standardized_to_oped_dataframe(_sample_standardized())
    save_formatted_cache(
        df,
        "oped",
        "deepprime",
        "clinvar",
        "hek293t",
        "pe2",
        datasets_dir=datasets_dir,
    )
    loaded = load_formatted_cache(
        "oped",
        "deepprime",
        "clinvar",
        "hek293t",
        "pe2",
        datasets_dir=datasets_dir,
        expected_rows=len(df),
    )
    assert loaded is not None
    assert list(loaded.columns) == list(df.columns)
    assert len(loaded) == len(df)


def test_load_formatted_cache_rejects_row_count_mismatch(datasets_dir: Path):
    df = standardized_to_oped_dataframe(_sample_standardized())
    save_formatted_cache(
        df,
        "oped",
        "deepprime",
        "clinvar",
        "hek293t",
        "pe2",
        datasets_dir=datasets_dir,
    )
    assert (
        load_formatted_cache(
            "oped",
            "deepprime",
            "clinvar",
            "hek293t",
            "pe2",
            datasets_dir=datasets_dir,
            expected_rows=len(df) + 1,
        )
        is None
    )


def test_load_or_convert_formatted_uses_cache(datasets_dir: Path, monkeypatch: pytest.MonkeyPatch):
    converter = DataConverter(datasets_dir)
    source = _sample_standardized()
    first = converter.load_or_convert_formatted(
        source,
        study="deepprime",
        dataset="clinvar",
        cell_line="hek293t",
        pe_system="pe2",
        target_format="oped",
    )
    cache_path = formatted_cache_path(
        "oped",
        "deepprime",
        "clinvar",
        "hek293t",
        "pe2",
        datasets_dir=datasets_dir,
    )
    assert cache_path.is_file()

    calls = {"count": 0}
    original = converter.convert_from_standardized

    def counting_convert(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(converter, "convert_from_standardized", counting_convert)
    second = converter.load_or_convert_formatted(
        source,
        study="deepprime",
        dataset="clinvar",
        cell_line="hek293t",
        pe_system="pe2",
        target_format="oped",
    )
    assert calls["count"] == 0
    pd.testing.assert_frame_equal(first, second)


def test_clear_formatted_cache_on_force_reexport(datasets_dir: Path):
    df = standardized_to_oped_dataframe(_sample_standardized())
    save_formatted_cache(
        df,
        "oped",
        "deepprime",
        "clinvar",
        "hek293t",
        "pe2",
        datasets_dir=datasets_dir,
    )
    cache_path = formatted_cache_path(
        "oped",
        "deepprime",
        "clinvar",
        "hek293t",
        "pe2",
        datasets_dir=datasets_dir,
    )
    assert cache_path.is_file()

    clear_formatted_cache(datasets_dir=datasets_dir)
    assert not cache_path.exists()
