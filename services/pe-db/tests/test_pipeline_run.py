"""Export/standardize orchestration: force flags must rebuild even when files exist."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from pe_db.pipeline import run as pipeline_run


def test_export_skips_complete_study_unless_forced(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []

    monkeypatch.setattr(pipeline_run, "_ensure_studies_loaded", lambda: None)
    monkeypatch.setattr(
        pipeline_run,
        "iter_pipelines",
        lambda: (SimpleNamespace(key="deeppe"),),
    )
    monkeypatch.setattr(
        pipeline_run,
        "get_pipeline",
        lambda _name: SimpleNamespace(exporters=(lambda: called.append("export"),)),
    )
    monkeypatch.setattr(pipeline_run, "_missing_exported_datasets", lambda _name: [])
    monkeypatch.setattr(
        "pe_db.catalog.datasheets.index_exported_datasheets", lambda: None
    )
    monkeypatch.setattr("pe_db.formatted_cache.clear_formatted_cache", lambda **_k: None)

    pipeline_run.export_original_data(force_reexport=False)
    assert called == []

    pipeline_run.export_original_data(force_reexport=True)
    assert called == ["export"]


def test_force_reexport_also_restandardizes(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, bool] = {}

    monkeypatch.setattr(
        "pe_db.library.export_original_data",
        lambda study=None, force_reexport=False: calls.update(export=force_reexport),
    )
    monkeypatch.setattr(
        "pe_db.library.standardize_exported_data",
        lambda study=None, force=False: calls.update(standardize=force),
    )
    monkeypatch.setattr("pe_db.library.ensure_plugins_loaded", lambda: [])

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class _Repo:
        def list_datasheets(self):
            return []

    monkeypatch.setattr("pe_db.library.get_session", lambda: _Session())
    monkeypatch.setattr("pe_db.library.CatalogRepository", lambda _session: _Repo())

    from pe_db.library import run_export

    run_export(study="pridict2", force_reexport=True)
    assert calls == {"export": True, "standardize": True}
