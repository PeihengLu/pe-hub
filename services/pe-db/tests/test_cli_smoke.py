"""CLI smoke tests for every ``pedb`` subcommand.

Argument parsing and dispatch are exercised through ``pe_db.cli.main``. Catalog
commands use an isolated seeded database. Pipeline commands (init / export /
standardize / convert) are stubbed so the tests stay fast and do not rewrite
``datasets/``.
"""
from __future__ import annotations

import argparse
import json
from typing import Callable

import pytest

from pe_db.cli import build_parser, main

pytestmark = pytest.mark.smoke

CommandFn = Callable[[pytest.CaptureFixture[str]], None]


def _subcommand_paths(parser: argparse.ArgumentParser, prefix: str = "") -> set[str]:
    paths: set[str] = set()
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        for name, subparser in action.choices.items():
            path = f"{prefix}{name}"
            nested = _subcommand_paths(subparser, prefix=f"{path} ")
            if nested:
                paths.update(nested)
            else:
                paths.add(path)
    return paths


def test_cli_subcommand_inventory():
    actual = _subcommand_paths(build_parser())
    declared = set(CLI_CASES)
    missing = actual - declared
    extra = declared - actual
    assert not missing, f"Add PE-DB CLI smoke coverage for: {sorted(missing)}"
    assert not extra, f"Remove stale PE-DB CLI smoke cases: {sorted(extra)}"


@pytest.fixture
def stub_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pe_db.cli.run_init", lambda **kwargs: None)
    monkeypatch.setattr(
        "pe_db.cli.run_export",
        lambda **kwargs: {
            "status": "success",
            "study": kwargs.get("study") or "all",
            "datasheets_in_catalog": 0,
        },
    )
    monkeypatch.setattr(
        "pe_db.cli.run_standardize",
        lambda **kwargs: {"status": "success", "study": kwargs.get("study") or "all"},
    )
    monkeypatch.setattr(
        "pe_db.cli.run_convert_sheet",
        lambda **kwargs: {
            "status": "success",
            "records_converted": 0,
            "output_columns": [],
        },
    )


def _run(argv: list[str]) -> int:
    return int(main(argv))


def _json_stdout(capsys: pytest.CaptureFixture[str]) -> object:
    captured = capsys.readouterr()
    return json.loads(captured.out)


def _case_init(capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["init"]) == 0
    assert "initialization complete" in capsys.readouterr().out.lower()


def _case_seed(capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["seed"]) == 0
    assert "seeded" in capsys.readouterr().out.lower()


def _case_export(capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["export", "--study", "deepprime"]) == 0
    payload = _json_stdout(capsys)
    assert payload["status"] == "success"


def _case_standardize(capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["standardize", "--study", "deepprime"]) == 0
    payload = _json_stdout(capsys)
    assert payload["status"] == "success"


def _case_cache_clear(capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["cache-clear", "--dry-run"]) == 0
    payload = _json_stdout(capsys)
    assert "removed" in payload or "dry_run" in payload or "formatted" in payload


def _case_convert(capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(
        [
            "convert",
            "--study",
            "deepprime",
            "--dataset",
            "library2",
            "--cell-line",
            "HEK293T",
            "--pe-system",
            "PE2max",
        ]
    ) == 0
    payload = _json_stdout(capsys)
    assert payload["status"] == "success"


def _case_filter(capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["filter", "--study", "deepprime"]) == 0
    payload = _json_stdout(capsys)
    assert payload["status"] == "success"
    assert payload["format"] is None


def _case_studies(capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["studies"]) == 0
    names = {row["name"] for row in _json_stdout(capsys)}
    assert "deepprime" in names


def _case_datasets(capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["datasets", "--study", "deepprime"]) == 0
    rows = _json_stdout(capsys)
    assert isinstance(rows, list)
    assert rows


def _case_datasheets(capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["datasheets"]) == 0
    assert _json_stdout(capsys) == []


def _case_scaffolds(capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["scaffolds"]) == 0
    rows = _json_stdout(capsys)
    assert any(row["id"] == 1 for row in rows)


def _case_statistics(capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["statistics"]) == 0
    payload = _json_stdout(capsys)
    assert payload["total_entries"] == 0


def _case_formats(capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["formats"]) == 0
    names = capsys.readouterr().out.split()
    assert "std" in names
    assert "deepprime" in names


def _case_plugins_reload(capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["plugins", "reload"]) == 0
    payload = _json_stdout(capsys)
    assert payload["count"] == 0


CLI_CASES: dict[str, CommandFn] = {
    "init": _case_init,
    "seed": _case_seed,
    "export": _case_export,
    "standardize": _case_standardize,
    "cache-clear": _case_cache_clear,
    "convert": _case_convert,
    "filter": _case_filter,
    "studies": _case_studies,
    "datasets": _case_datasets,
    "datasheets": _case_datasheets,
    "scaffolds": _case_scaffolds,
    "statistics": _case_statistics,
    "formats": _case_formats,
    "plugins reload": _case_plugins_reload,
}


@pytest.mark.parametrize("command", sorted(CLI_CASES))
def test_cli_command_smoke(
    command: str,
    seeded_catalog,
    stub_pipeline,
    capsys: pytest.CaptureFixture[str],
):
    CLI_CASES[command](capsys)


def test_filter_format_without_split_is_error(
    seeded_catalog, stub_pipeline, capsys: pytest.CaptureFixture[str]
):
    assert _run(["filter", "--format", "deepprime"]) == 1
    err = capsys.readouterr().err
    assert "split_strategy" in err


def test_filter_writes_json_export(
    seeded_catalog, stub_pipeline, tmp_path, capsys: pytest.CaptureFixture[str]
):
    out = tmp_path / "export.json"
    assert _run(["filter", "--format", "std", "--split-strategy", "none", "--out", str(out)]) == 0
    capsys.readouterr()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "success"


def test_filter_writes_csv_export(
    seeded_catalog, stub_pipeline, tmp_path, capsys: pytest.CaptureFixture[str]
):
    out = tmp_path / "export.csv"
    assert _run(["filter", "--format", "std", "--split-strategy", "none", "--out", str(out)]) == 0
    capsys.readouterr()
    assert out.is_file()


def test_filter_rejects_unknown_out_extension(
    seeded_catalog, stub_pipeline, tmp_path, capsys: pytest.CaptureFixture[str]
):
    assert _run(["filter", "--out", str(tmp_path / "x.bin")]) == 1
    assert "Unsupported --out" in capsys.readouterr().err


def test_missing_command_exits():
    with pytest.raises(SystemExit):
        main([])
