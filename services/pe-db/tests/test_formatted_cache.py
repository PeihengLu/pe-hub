"""Tests for formatted model-format disk cache."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.converter import DataConverter
from app.formatted_cache import (
    FORMATTED_CACHE_REVISIONS,
    clear_formatted_cache,
    formatted_cache_path,
    formatted_cache_revision,
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
    sidecar = Path(str(formatted_cache_path(
        "oped",
        "deepprime",
        "clinvar",
        "hek293t",
        "pe2",
        datasets_dir=datasets_dir,
    )) + ".revision")
    assert sidecar.read_text(encoding="utf-8").strip() == str(formatted_cache_revision("oped"))


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


def test_load_formatted_cache_skips_stale_revision(datasets_dir: Path):
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
    sidecar = Path(str(cache_path) + ".revision")
    sidecar.write_text("1", encoding="utf-8")
    assert (
        load_formatted_cache(
            "oped",
            "deepprime",
            "clinvar",
            "hek293t",
            "pe2",
            datasets_dir=datasets_dir,
            expected_rows=len(df),
        )
        is None
    )


def test_load_formatted_cache_skips_pre_revision_when_format_bumped(
    datasets_dir: Path,
):
    """Existing OPED parquet has no sidecar; current revision is 2."""
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
    Path(str(cache_path) + ".revision").unlink()
    assert FORMATTED_CACHE_REVISIONS["oped"] > 1
    assert (
        load_formatted_cache(
            "oped",
            "deepprime",
            "clinvar",
            "hek293t",
            "pe2",
            datasets_dir=datasets_dir,
            expected_rows=len(df),
        )
        is None
    )


def test_load_formatted_cache_keeps_pre_revision_at_v1(
    datasets_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setitem(FORMATTED_CACHE_REVISIONS, "oped", 1)
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
    Path(str(formatted_cache_path(
        "oped",
        "deepprime",
        "clinvar",
        "hek293t",
        "pe2",
        datasets_dir=datasets_dir,
    )) + ".revision").unlink()
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
    assert len(loaded) == len(df)


def test_saving_one_oped_sheet_does_not_validate_sibling_stale_parquet(datasets_dir: Path):
    """Format-level .revision previously made every OPED parquet look current."""
    df = standardized_to_oped_dataframe(_sample_standardized())
    stale_path = formatted_cache_path(
        "oped",
        "deepprime",
        "clinvar",
        "hek293t",
        "pe2",
        datasets_dir=datasets_dir,
    )
    stale_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(stale_path, index=False)
    save_formatted_cache(
        df,
        "oped",
        "minsepie",
        "library-insert-18nt",
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
            expected_rows=len(df),
        )
        is None
    )
    loaded = load_formatted_cache(
        "oped",
        "minsepie",
        "library-insert-18nt",
        "hek293t",
        "pe2",
        datasets_dir=datasets_dir,
        expected_rows=len(df),
    )
    assert loaded is not None


def test_load_or_convert_formatted_restores_source_index(
    datasets_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    converter = DataConverter(datasets_dir)
    source = _sample_standardized()
    source.index = pd.Index([10, 20], name="row")
    converter.load_or_convert_formatted(
        source,
        study="deepprime",
        dataset="clinvar",
        cell_line="hek293t",
        pe_system="pe2",
        target_format="oped",
    )
    cached = converter.load_or_convert_formatted(
        source,
        study="deepprime",
        dataset="clinvar",
        cell_line="hek293t",
        pe_system="pe2",
        target_format="oped",
    )
    assert list(cached.index) == [10, 20]
    # Subsetting like the filter/merge path must keep alignment.
    subset = cached.loc[[20]]
    assert len(subset) == 1
    assert int(subset.index[0]) == 20


def test_convert_pending_sheets_via_formatted_cache_aligns_with_merge(
    datasets_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    from app.db.repository import _convert_pending_sheets_via_formatted_cache

    converter = DataConverter(datasets_dir)
    sheet_a = _sample_standardized()
    sheet_b = _sample_standardized()
    sheet_b["editing_efficiency"] = [0.1, 0.2]
    # Simulate an edit filter that drops the first row of sheet B.
    filtered_b = sheet_b.iloc[[1]]

    calls: list[tuple[str, str]] = []
    original = converter.load_or_convert_formatted

    def tracking_load(source, **kwargs):
        calls.append((kwargs["study"], kwargs["dataset"]))
        return original(source, **kwargs)

    monkeypatch.setattr(converter, "load_or_convert_formatted", tracking_load)

    pending = [
        (
            {
                "study": "deepprime",
                "dataset": "clinvar",
                "cell_line": "hek293t",
                "pe_system": "pe2",
            },
            sheet_a,
            sheet_a,
        ),
        (
            {
                "study": "pridict1",
                "dataset": "library1",
                "cell_line": "hek293t",
                "pe_system": "pe2",
            },
            sheet_b,
            filtered_b,
        ),
    ]
    merged_std = pd.concat([sheet_a, filtered_b], ignore_index=True)
    converted = _convert_pending_sheets_via_formatted_cache(
        converter,
        pending,
        target_format="oped",
    )
    assert calls == [
        ("deepprime", "clinvar"),
        ("pridict1", "library1"),
    ]
    assert len(converted) == len(merged_std) == 3

    # Second call should hit cache (no convert_from_standardized).
    convert_calls = {"count": 0}
    real_convert = converter.convert_from_standardized

    def counting_convert(*args, **kwargs):
        convert_calls["count"] += 1
        return real_convert(*args, **kwargs)

    monkeypatch.setattr(converter, "convert_from_standardized", counting_convert)
    monkeypatch.setattr(converter, "load_or_convert_formatted", original)
    again = _convert_pending_sheets_via_formatted_cache(
        converter,
        pending,
        target_format="oped",
    )
    assert convert_calls["count"] == 0
    assert len(again) == 3


def test_convert_pending_sheets_remaps_duplicate_seq_ids(
    datasets_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """Per-sheet caches both use seq_0..; merge must uniquify for PRIDICT2."""
    from app.db.repository import _convert_pending_sheets_via_formatted_cache

    converter = DataConverter(datasets_dir)
    sheet_a = _sample_standardized()
    sheet_b = _sample_standardized()
    pending = [
        (
            {
                "study": "deepprime",
                "dataset": "clinvar",
                "cell_line": "hek293t",
                "pe_system": "pe2",
            },
            sheet_a,
            sheet_a,
        ),
        (
            {
                "study": "pridict1",
                "dataset": "library1",
                "cell_line": "hek293t",
                "pe_system": "pe2",
            },
            sheet_b,
            sheet_b,
        ),
    ]
    # Force pridict2-shaped frames with colliding seq_ids without full convert.
    def fake_load(source, **kwargs):
        n = len(source)
        return pd.DataFrame(
            {
                "seq_id": [f"seq_{i}" for i in range(n)],
                "deepeditposition_lst": [f"[{i}]" for i in range(n)],
                "wide_initial_target": ["A" * 99] * n,
                "wide_mutated_target": ["T" * 99] * n,
            },
            index=source.index,
        )

    monkeypatch.setattr(converter, "load_or_convert_formatted", fake_load)
    converted = _convert_pending_sheets_via_formatted_cache(
        converter,
        pending,
        target_format="pridict2",
    )
    assert len(converted) == 4
    assert converted["seq_id"].is_unique
    assert list(converted["seq_id"]) == ["seq_0", "seq_1", "seq_2", "seq_3"]


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
