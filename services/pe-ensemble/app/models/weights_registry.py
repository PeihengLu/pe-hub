"""Central registry for pe-ensemble model weight sets.

All pretrained and service-trained weights live under ``WEIGHTS_ROOT`` (default:
``services/pe-ensemble/weights``). Each weight set is a directory with a
``manifest.json``; ``registry.json`` at the root is a fast aggregate index.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from .registry import model_registry

REGISTRY_FILENAME = "registry.json"
MANIFEST_FILENAME = "manifest.json"
TRAIN_LOCI_FILENAME = "train_target_loci.json"


def loci_fingerprint(loci: Iterable[str]) -> str:
    """Return a deterministic fingerprint over a set of target-locus IDs."""
    unique = sorted({str(value) for value in loci if value})
    digest = hashlib.sha1("\n".join(unique).encode("utf-8")).hexdigest()
    return f"sha1:{digest}"


def _known_models() -> tuple[str, ...]:
    return model_registry.names()


def _assert_known_model(model: str) -> None:
    known = _known_models()
    if known and model not in known:
        raise ValueError(f"Unknown model '{model}'. Supported: {known}")


def weights_root() -> Path:
    """Resolve the persistent weights directory."""
    env = os.getenv("WEIGHTS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return (Path(__file__).resolve().parents[2] / "weights").resolve()


def _entry_dir(model: str, weight_id: str) -> Path:
    return weights_root() / model / weight_id


def _registry_path() -> Path:
    return weights_root() / REGISTRY_FILENAME


@contextmanager
def _registry_lock():
    """Best-effort serialization for registry writes (atomic replace)."""
    yield


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return slug.strip("-") or "custom"


def _scope_from_metadata(metadata: Dict[str, Any]) -> str:
    """Build a scope slug from training filters / model kwargs."""
    parts: List[str] = []
    model_kwargs = metadata.get("model_kwargs") or {}
    training = metadata.get("training") or {}
    filters = training.get("filters") or {}

    for key in ("cell_line", "cell_type", "pe_system"):
        raw = model_kwargs.get(key) or filters.get(key)
        if raw is None:
            continue
        if isinstance(raw, list):
            parts.extend(str(v) for v in raw if v)
        else:
            parts.append(str(raw))

    if not parts:
        return "custom"
    return _slugify("-".join(parts))


def generate_id(model: str, metadata: Optional[Dict[str, Any]] = None, when: Optional[datetime] = None) -> str:
    """Generate a structured weight-set ID for a trained run."""
    metadata = metadata or {}
    when = when or datetime.now(timezone.utc)
    scope = _scope_from_metadata(metadata)
    date_part = when.strftime("%Y%m%d")
    shortid = uuid.uuid4().hex[:6]
    return f"{model}__{scope}__{date_part}__{shortid}"


def _read_manifest(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _write_manifest(entry_dir: Path, manifest: Dict[str, Any]) -> None:
    entry_dir.mkdir(parents=True, exist_ok=True)
    with open(entry_dir / MANIFEST_FILENAME, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _normalized_loci_list(train_target_loci: Optional[Iterable[str]]) -> Optional[list[str]]:
    if train_target_loci is None:
        return None
    return sorted({str(value) for value in train_target_loci if value})


def _write_training_loci(entry_dir: Path, loci: Optional[list[str]]) -> None:
    if loci is None:
        return
    with open(entry_dir / TRAIN_LOCI_FILENAME, "w", encoding="utf-8") as handle:
        json.dump({"target_uids": loci}, handle)
        handle.write("\n")


def _manifest_summary(manifest: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": manifest["id"],
        "model": manifest["model"],
        "label": manifest.get("label", manifest["id"]),
        "source": manifest.get("source", "trained"),
        "format": manifest.get("format"),
        "created_at": manifest.get("created_at"),
        "metrics": manifest.get("metrics"),
        "notes": manifest.get("notes"),
    }


def rebuild_index() -> Dict[str, Any]:
    """Rebuild ``registry.json`` by scanning per-entry manifests."""
    root = weights_root()
    root.mkdir(parents=True, exist_ok=True)
    entries: List[Dict[str, Any]] = []
    model_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    for model_dir in model_dirs:
        model = model_dir.name
        if not model_dir.is_dir():
            continue
        for entry_dir in sorted(model_dir.iterdir()):
            manifest_path = entry_dir / MANIFEST_FILENAME
            if not entry_dir.is_dir() or not manifest_path.is_file():
                continue
            manifest = _read_manifest(manifest_path)
            entries.append(_manifest_summary(manifest))

    payload = {
        "version": 1,
        "updated_at": _utc_now_iso(),
        "count": len(entries),
        "entries": entries,
    }
    registry_path = _registry_path()
    tmp_path = registry_path.with_suffix(".json.tmp")
    with _registry_lock():
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        tmp_path.replace(registry_path)
    return payload


def list_entries(model: str) -> List[Dict[str, Any]]:
    """List manifest summaries for a model (rebuilds index if missing)."""
    _assert_known_model(model)

    registry_file = _registry_path()
    if not registry_file.is_file():
        rebuild_index()
    else:
        with open(registry_file, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not payload.get("entries"):
            rebuild_index()

    with open(registry_file, encoding="utf-8") as handle:
        payload = json.load(handle)

    return [entry for entry in payload.get("entries", []) if entry.get("model") == model]


def list_weight_ids(model: str) -> List[str]:
    """Return weight IDs for backward-compatible callers."""
    return [entry["id"] for entry in list_entries(model)]


def resolve_dir(model: str, weight_id: str) -> Path:
    """Resolve a weight ID to its on-disk directory."""
    entry = _entry_dir(model, weight_id)
    if not entry.is_dir():
        available = list_weight_ids(model)
        raise ValueError(
            f"Unknown {model} weights '{weight_id}'. Available: {available}"
        )
    manifest_path = entry / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ValueError(f"Weight entry '{weight_id}' is missing {MANIFEST_FILENAME}.")
    return entry


def get_manifest(model: str, weight_id: str) -> Dict[str, Any]:
    return _read_manifest(resolve_dir(model, weight_id) / MANIFEST_FILENAME)


def register(
    model: str,
    *,
    weight_id: Optional[str] = None,
    label: Optional[str] = None,
    source: str = "trained",
    format_name: str,
    metadata: Optional[Dict[str, Any]] = None,
    populate: Callable[[Path], None],
    files: Optional[Iterable[str]] = None,
    rebuild: bool = True,
) -> str:
    """Create a new weight entry directory and register it.

    ``populate(dest_dir)`` must write weight files into the new entry directory.
    """
    _assert_known_model(model)

    metadata = dict(metadata or {})
    weight_id = weight_id or generate_id(model, metadata)
    entry_dir = _entry_dir(model, weight_id)
    if entry_dir.exists():
        raise FileExistsError(f"Weight entry already exists: {entry_dir}")

    entry_dir.mkdir(parents=True, exist_ok=False)
    try:
        populate(entry_dir)
        if files is None:
            files = sorted(
                p.name
                for p in entry_dir.iterdir()
                if p.is_file() and p.name != MANIFEST_FILENAME
            )
        manifest: Dict[str, Any] = {
            "id": weight_id,
            "model": model,
            "label": label or weight_id,
            "source": source,
            "format": format_name,
            "created_at": _utc_now_iso(),
            "files": list(files),
            "provenance": metadata.get("provenance") or {"vendor_origin": None},
            "training": metadata.get("training"),
            "metrics": metadata.get("metrics"),
            "notes": metadata.get("notes"),
        }
        _write_manifest(entry_dir, manifest)
    except Exception:
        shutil.rmtree(entry_dir, ignore_errors=True)
        raise

    if rebuild:
        with _registry_lock():
            rebuild_index()
    return weight_id


def register_from_directory(
    model: str,
    src_dir: Path,
    *,
    weight_id: str,
    label: str,
    source: str,
    format_name: str,
    metadata: Optional[Dict[str, Any]] = None,
    move: bool = False,
    rebuild: bool = True,
) -> str:
    """Register an existing directory of weight files."""
    src_dir = src_dir.resolve()
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    def populate(dest: Path) -> None:
        for item in src_dir.iterdir():
            if item.name == MANIFEST_FILENAME:
                continue
            target = dest / item.name
            if move:
                shutil.move(str(item), str(target))
            else:
                if item.is_dir():
                    shutil.copytree(item, target)
                else:
                    shutil.copy2(item, target)

    return register(
        model,
        weight_id=weight_id,
        label=label,
        source=source,
        format_name=format_name,
        metadata=metadata,
        populate=populate,
        rebuild=rebuild,
    )


def register_trained_model(
    model: str,
    wrapper: Any,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    notes: Optional[str] = None,
    weight_id: Optional[str] = None,
    label: Optional[str] = None,
    train_target_loci: Optional[Iterable[str]] = None,
) -> str:
    """Persist a trained in-memory model wrapper into the registry.

    When ``train_target_loci`` is provided, the universal target-locus IDs the
    model was trained on are written to a ``train_target_loci.json`` sidecar so
    that evaluation can detect data leakage between train and test data.
    """
    from .registry import model_registry

    metadata = dict(metadata or {})
    if metrics is not None:
        metadata["metrics"] = metrics
    if notes:
        metadata["notes"] = notes

    loci = _normalized_loci_list(train_target_loci)

    spec = model_registry.get(model)
    format_name = spec.weight_format

    def populate(dest: Path) -> None:
        wrapper.save_to_registry(dest)
        _write_training_loci(dest, loci)

    auto_label = label
    if auto_label is None:
        scope = _scope_from_metadata(metadata)
        auto_label = f"{model.title()} - {scope.replace('-', ' ')} ({datetime.now(timezone.utc):%Y-%m-%d})"

    return register(
        model,
        weight_id=weight_id,
        label=auto_label,
        source="trained",
        format_name=format_name,
        metadata=metadata,
        populate=populate,
    )


def write_training_provenance(
    model: str,
    weight_id: str,
    *,
    training: Dict[str, Any],
    train_target_loci: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Backfill or update manifest training metadata and loci sidecar."""
    entry_dir = resolve_dir(model, weight_id)
    manifest = get_manifest(model, weight_id)
    manifest["training"] = dict(training)
    _write_manifest(entry_dir, manifest)
    _write_training_loci(entry_dir, _normalized_loci_list(train_target_loci))
    return manifest


def load_training_loci(model: str, weight_id: str) -> Optional[set[str]]:
    """Return the universal target-locus IDs a weight set was trained on.

    Returns ``None`` when provenance is unavailable (e.g. vendor pretrained
    weights), which callers should treat as "training data unknown".
    """
    try:
        entry_dir = resolve_dir(model, weight_id)
    except ValueError:
        # PRIDICT2 vendor runs are multi-head (e.g. decoder_HEK.pkl), and the
        # UI selects weights using a `{base_run_id}__{cell_type}` suffix.
        # Training provenance sidecars are stored under the base run
        # directory, so normalize the suffix here.
        if model == "pridict2" and "__" in weight_id:
            base_id = weight_id.rsplit("__", 1)[0]
            if base_id and base_id != weight_id:
                try:
                    entry_dir = resolve_dir(model, base_id)
                except ValueError:
                    return None
            else:
                return None
        else:
            return None
    loci_path = entry_dir / TRAIN_LOCI_FILENAME
    if not loci_path.is_file():
        return None
    try:
        with open(loci_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    uids = payload.get("target_uids")
    if not isinstance(uids, list):
        return None
    return {str(value) for value in uids if value}


def load_training_provenance(model: str, weight_id: str) -> Optional[Dict[str, Any]]:
    """Return the recorded training ``data_provenance`` block, if any."""
    training = load_training_metadata(model, weight_id)
    if training is None:
        return None
    provenance = training.get("data_provenance")
    return provenance if isinstance(provenance, dict) else None


def load_training_metadata(model: str, weight_id: str) -> Optional[Dict[str, Any]]:
    """Return the full ``training`` block from a weight manifest, if recorded."""
    try:
        manifest = get_manifest(model, weight_id)
    except ValueError:
        return None
    training = manifest.get("training")
    return training if isinstance(training, dict) else None
