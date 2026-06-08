"""Tests for the centralized weights registry."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.models import weights_registry


@pytest.fixture()
def weights_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "weights"
    monkeypatch.setenv("WEIGHTS_ROOT", str(root))
    return root


def test_generate_id_is_structured():
    weight_id = weights_registry.generate_id(
        "deepprime",
        metadata={"model_kwargs": {"cell_type": "HEK293T", "pe_system": "PE2max"}},
    )
    parts = weight_id.split("__")
    assert parts[0] == "deepprime"
    assert "hek293t" in parts[1]
    assert len(parts) == 4
    assert len(parts[3]) == 6


def test_register_list_resolve_round_trip(weights_root: Path):
    def populate(dest: Path) -> None:
        (dest / "weights.pt").write_text("stub", encoding="utf-8")

    weight_id = weights_registry.register(
        "oped",
        weight_id="test-oped-weights",
        label="Test OPED",
        source="trained",
        format_name="oped_state_dict",
        populate=populate,
    )
    assert weight_id == "test-oped-weights"

    entries = weights_registry.list_entries("oped")
    assert any(entry["id"] == weight_id for entry in entries)

    entry_dir = weights_registry.resolve_dir("oped", weight_id)
    manifest = json.loads((entry_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["label"] == "Test OPED"
    assert (entry_dir / "weights.pt").is_file()


def test_rebuild_index_from_manifests(weights_root: Path):
    entry_dir = weights_root / "deepprime" / "DeepPrime_base"
    entry_dir.mkdir(parents=True)
    (entry_dir / "model_0.pt").write_text("stub", encoding="utf-8")
    (entry_dir / "manifest.json").write_text(
        json.dumps(
            {
                "id": "DeepPrime_base",
                "model": "deepprime",
                "label": "DeepPrime base",
                "source": "vendor",
                "format": "deepprime_ensemble",
                "created_at": "2026-06-08T00:00:00Z",
                "files": ["model_0.pt"],
            }
        ),
        encoding="utf-8",
    )

    payload = weights_registry.rebuild_index()
    assert payload["count"] == 1
    assert weights_registry.list_weight_ids("deepprime") == ["DeepPrime_base"]
