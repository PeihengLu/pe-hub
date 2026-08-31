"""Tests for the centralized weights registry."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from app.models import deepprime_vendor_provenance
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


def test_load_training_loci_and_provenance_round_trip(weights_root: Path):
    loci = ["ps:cccc", "ps:aaaa", "ps:bbbb", "ps:aaaa", ""]

    class _StubWrapper:
        def save_to_registry(self, dest: Path) -> None:
            (dest / "weights.pt").write_text("stub", encoding="utf-8")

    metadata = {
        "training": {
            "dataset_name": "demo",
            "data_provenance": {
                "target_uid_fingerprint": weights_registry.loci_fingerprint(loci),
                "n_target_loci": 3,
                "has_original_test_split": True,
            },
        }
    }

    with patch.object(
        weights_registry.model_registry,
        "get",
        return_value=type("Spec", (), {"weight_format": "deepprime_ensemble"})(),
    ):
        weight_id = weights_registry.register_trained_model(
            "deepprime",
            _StubWrapper(),
            metadata=metadata,
            weight_id="trained-with-loci",
            label="Trained w/ loci",
            train_target_loci=loci,
        )

    entry_dir = weights_registry.resolve_dir("deepprime", weight_id)
    assert (entry_dir / weights_registry.TRAIN_LOCI_FILENAME).is_file()

    stored = weights_registry.load_training_loci("deepprime", weight_id)
    assert stored == {"ps:aaaa", "ps:bbbb", "ps:cccc"}

    provenance = weights_registry.load_training_provenance("deepprime", weight_id)
    assert provenance is not None
    assert provenance["n_target_loci"] == 3
    assert provenance["has_original_test_split"] is True


def test_load_training_loci_returns_none_without_sidecar(weights_root: Path):
    entry_dir = weights_root / "deepprime" / "no-loci"
    entry_dir.mkdir(parents=True)
    (entry_dir / "manifest.json").write_text(
        json.dumps(
            {
                "id": "no-loci",
                "model": "deepprime",
                "label": "no loci",
                "source": "vendor",
                "format": "deepprime_ensemble",
                "created_at": "2026-06-08T00:00:00Z",
                "files": [],
            }
        ),
        encoding="utf-8",
    )
    assert weights_registry.load_training_loci("deepprime", "no-loci") is None
    assert weights_registry.load_training_provenance("deepprime", "no-loci") is None


def test_sync_deepprime_vendor_provenance_backfills_manifest_and_loci(
    weights_root: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    deepprime_root = weights_root / "deepprime"
    for weight_id in ("DeepPrime_base", "DeepPrime_off", "DP_variant_demo"):
        entry_dir = deepprime_root / weight_id
        entry_dir.mkdir(parents=True)
        (entry_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "id": weight_id,
                    "model": "deepprime",
                    "label": weight_id,
                    "source": "vendor",
                    "format": "deepprime_ensemble",
                    "created_at": "2026-06-08T00:00:00Z",
                    "files": [],
                    "training": None,
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        deepprime_vendor_provenance,
        "_DEEPPRIME_WORKBOOK",
        weights_root / "deepprime-org.xlsx",
    )
    (weights_root / "deepprime-org.xlsx").write_text("stub", encoding="utf-8")
    monkeypatch.setattr(deepprime_vendor_provenance, "_read_summary", lambda: [])
    monkeypatch.setattr(
        deepprime_vendor_provenance,
        "_collect_dataset_loci",
        lambda _specs: (
            {
                ("deepprime_clinvar", "hek293t", "pe2"): {"ps:clin-a", "ps:clin-b"},
                ("deepprime_off", "hek293t", "pe2_off"): {"ps:off-a"},
                ("deepprime_small", "hek293t", "pe2"): {"ps:small-a"},
            },
            {
                ("deepprime_clinvar", "hek293t", "pe2"): True,
                ("deepprime_off", "hek293t", "pe2_off"): True,
                ("deepprime_small", "hek293t", "pe2"): True,
            },
        ),
    )
    monkeypatch.setattr(
        deepprime_vendor_provenance,
        "_FINE_TUNE_DATASHEET_BY_WEIGHT",
        {"DP_variant_demo": ("hek293t", "pe2")},
    )

    result = deepprime_vendor_provenance.sync_deepprime_vendor_provenance()
    assert result == {"updated_weights": 3}

    base_manifest = weights_registry.get_manifest("deepprime", "DeepPrime_base")
    assert base_manifest["training"]["filters"]["dataset"] == ["deepprime-clinvar"]
    assert base_manifest["training"]["data_provenance"]["has_original_test_split"] is True
    assert base_manifest["training"]["data_provenance"]["train_folds_only"] is True
    assert weights_registry.load_training_loci("deepprime", "DeepPrime_base") == {
        "ps:clin-a",
        "ps:clin-b",
    }

    off_manifest = weights_registry.get_manifest("deepprime", "DeepPrime_off")
    assert off_manifest["training"]["filters"]["dataset"] == ["deepprime-off"]
    assert weights_registry.load_training_loci("deepprime", "DeepPrime_off") == {"ps:off-a"}

    variant_manifest = weights_registry.get_manifest("deepprime", "DP_variant_demo")
    assert variant_manifest["training"]["filters"]["dataset"] == [
        "deepprime-clinvar",
        "deepprime-small",
    ]
    assert variant_manifest["training"]["filters"]["cell_line"] == ["hek293t"]
    assert variant_manifest["training"]["filters"]["pe_system"] == ["pe2"]
    assert weights_registry.load_training_loci("deepprime", "DP_variant_demo") == {
        "ps:clin-a",
        "ps:clin-b",
        "ps:small-a",
    }


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
    assert payload["tracked_count"] == 1
    assert payload["local_count"] == 0
    assert weights_registry.list_weight_ids("deepprime") == ["DeepPrime_base"]
    assert (weights_root / "registry.json").is_file()
    assert (weights_root / "local_registry.json").is_file()


def test_rebuild_index_splits_vendor_and_trained(weights_root: Path):
    vendor_dir = weights_root / "oped" / "pegRNA_vendor"
    vendor_dir.mkdir(parents=True)
    (vendor_dir / "manifest.json").write_text(
        json.dumps(
            {
                "id": "pegRNA_vendor",
                "model": "oped",
                "label": "vendor",
                "source": "vendor",
                "format": "oped_state_dict",
                "created_at": "2026-06-08T00:00:00Z",
                "files": [],
            }
        ),
        encoding="utf-8",
    )
    trained_dir = weights_root / "oped" / "oped__custom__20260611__abc123"
    trained_dir.mkdir(parents=True)
    (trained_dir / "manifest.json").write_text(
        json.dumps(
            {
                "id": "oped__custom__20260611__abc123",
                "model": "oped",
                "label": "trained",
                "source": "trained",
                "format": "oped_state_dict",
                "created_at": "2026-06-11T00:00:00Z",
                "files": [],
            }
        ),
        encoding="utf-8",
    )

    payload = weights_registry.rebuild_index()
    assert payload["tracked_count"] == 1
    assert payload["local_count"] == 1

    tracked = json.loads((weights_root / "registry.json").read_text(encoding="utf-8"))
    local = json.loads((weights_root / "local_registry.json").read_text(encoding="utf-8"))
    assert [e["id"] for e in tracked["entries"]] == ["pegRNA_vendor"]
    assert [e["id"] for e in local["entries"]] == ["oped__custom__20260611__abc123"]

    # Runtime listing merges both indexes (sorted by id).
    assert weights_registry.list_weight_ids("oped") == [
        "oped__custom__20260611__abc123",
        "pegRNA_vendor",
    ]


def test_is_git_tracked_source():
    assert weights_registry.is_git_tracked_source("vendor")
    assert weights_registry.is_git_tracked_source("plugin")
    assert not weights_registry.is_git_tracked_source("trained")
    assert not weights_registry.is_git_tracked_source(None)


def test_deepprime_sheet_target_uids_excludes_author_test_fold():
    frame = pd.DataFrame(
        {
            "wt_sequence": [
                "AAAA" + "ACGTACGTACGTACGTACGT" + "N" * 50,
                "CCCC" + "TGCATGCATGCATGCATGCA" + "N" * 50,
                "GGGG" + "ATATATATATATATATATAT" + "N" * 50,
            ],
            "fold": [0, "Test", 1],
        }
    )
    train_uids = deepprime_vendor_provenance._sheet_target_uids(frame, train_only=True)
    all_uids = deepprime_vendor_provenance._sheet_target_uids(frame, train_only=False)
    assert len(all_uids) == 3
    assert len(train_uids) == 2
    assert deepprime_vendor_provenance._sheet_has_author_test_split(frame) is True
