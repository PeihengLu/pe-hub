"""Shared plugin manifest and lifecycle helpers for PE-DB and PE Ensemble."""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .constants import PROJECT_ROOT

logger = logging.getLogger(__name__)

PLUGIN_STATE_FILENAME = ".state.json"
MANIFEST_YAML = "manifest.yaml"
MANIFEST_JSON = "manifest.json"

PLUGIN_NAME_PATTERN = re.compile(r"^[a-z0-9_]+$")
BUILTIN_MODEL_NAMES = frozenset({"deepprime", "oped", "pridict2"})
BUILTIN_FORMAT_NAMES = frozenset({"deepprime", "pridict", "pridict2", "oped"})

PluginStatus = str  # pending | active | rejected


class PluginError(ValueError):
    """Raised when a plugin manifest or state file is invalid."""


def plugins_root() -> Path:
    """Resolve the shared plugins directory (default: ``<repo>/plugins``)."""
    import os

    env = os.getenv("PLUGINS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return (PROJECT_ROOT / "plugins").resolve()


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_yaml_or_json(path: Path) -> Dict[str, Any]:
    text = _read_text(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise PluginError(
                "PyYAML is required to read manifest.yaml. "
                "Install pyyaml or use manifest.json."
            ) from exc
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise PluginError(f"Manifest must be a mapping: {path}")
    return payload


def find_manifest_path(plugin_dir: Path) -> Optional[Path]:
    yaml_path = plugin_dir / MANIFEST_YAML
    if yaml_path.is_file():
        return yaml_path
    json_path = plugin_dir / MANIFEST_JSON
    if json_path.is_file():
        return json_path
    return None


def read_plugin_state(plugin_dir: Path) -> Dict[str, Any]:
    state_path = plugin_dir / PLUGIN_STATE_FILENAME
    if not state_path.is_file():
        return {"status": "pending"}
    with open(state_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise PluginError(f"Invalid {PLUGIN_STATE_FILENAME} in {plugin_dir}")
    return payload


def write_plugin_state(plugin_dir: Path, state: Dict[str, Any]) -> None:
    state_path = plugin_dir / PLUGIN_STATE_FILENAME
    with open(state_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
        handle.write("\n")


def compute_plugin_file_hashes(plugin_dir: Path) -> Dict[str, str]:
    """SHA-256 hex digests for manifest, code, and weight files (relative paths)."""
    hashes: Dict[str, str] = {}
    for path in sorted(plugin_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name == PLUGIN_STATE_FILENAME:
            continue
        if path.name.startswith(".") and path.name not in {MANIFEST_YAML, MANIFEST_JSON}:
            continue
        rel = path.relative_to(plugin_dir).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[rel] = digest
    return hashes


@dataclass(frozen=True)
class PluginFormatSpec:
    module: str
    entrypoint: str
    required_std_columns: tuple[str, ...] = ()
    output_columns: tuple[str, ...] = ()
    label_column: Optional[str] = None


@dataclass(frozen=True)
class PluginModelSpec:
    module: str
    class_name: str
    pe_db_format: str
    weight_format: str
    hyperparameters: tuple[Dict[str, Any], ...] = ()
    constructor_kwargs: tuple[Dict[str, Any], ...] = ()


@dataclass(frozen=True)
class PluginWeightSpec:
    id: str
    files: tuple[str, ...] = ()
    notes: Optional[str] = None


@dataclass(frozen=True)
class PluginManifest:
    name: str
    version: str
    display_name: str
    description: str
    authors: tuple[str, ...] = ()
    format: Optional[PluginFormatSpec] = None
    model: Optional[PluginModelSpec] = None
    weights: tuple[PluginWeightSpec, ...] = ()
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def plugin_dir(self) -> Optional[Path]:
        return None  # set by loader wrapper


def _require_str(data: Mapping[str, Any], key: str, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PluginError(f"{context}: '{key}' must be a non-empty string")
    return value.strip()


def _optional_str_list(data: Mapping[str, Any], key: str) -> tuple[str, ...]:
    raw = data.get(key)
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise PluginError(f"Manifest field '{key}' must be a list of strings")
    return tuple(item.strip() for item in raw if item.strip())


def validate_plugin_name(name: str) -> str:
    key = name.strip().lower()
    if not PLUGIN_NAME_PATTERN.match(key):
        raise PluginError(
            f"Invalid plugin name '{name}'. Use lowercase letters, digits, and underscores."
        )
    if key in BUILTIN_MODEL_NAMES:
        raise PluginError(f"Plugin name '{key}' conflicts with a built-in model.")
    return key


def parse_manifest_data(data: Mapping[str, Any]) -> PluginManifest:
    """Parse a manifest mapping (from YAML/JSON) into a :class:`PluginManifest`."""
    name = validate_plugin_name(_require_str(data, "name", "manifest"))
    version = _require_str(data, "version", "manifest")
    display_name = _require_str(data, "display_name", "manifest")
    description = _require_str(data, "description", "manifest")
    authors = _optional_str_list(data, "authors")

    format_spec: Optional[PluginFormatSpec] = None
    format_raw = data.get("format")
    if format_raw is not None:
        if not isinstance(format_raw, dict):
            raise PluginError("manifest.format must be a mapping")
        format_spec = PluginFormatSpec(
            module=_require_str(format_raw, "module", "manifest.format"),
            entrypoint=_require_str(format_raw, "entrypoint", "manifest.format"),
            required_std_columns=_optional_str_list(format_raw, "required_std_columns"),
            output_columns=_optional_str_list(format_raw, "output_columns"),
            label_column=(
                str(format_raw["label_column"]).strip()
                if format_raw.get("label_column") is not None
                else None
            ),
        )

    model_spec: Optional[PluginModelSpec] = None
    model_raw = data.get("model")
    if model_raw is not None:
        if not isinstance(model_raw, dict):
            raise PluginError("manifest.model must be a mapping")
        hyperparameters: List[Dict[str, Any]] = []
        for item in model_raw.get("hyperparameters") or []:
            if isinstance(item, dict):
                hyperparameters.append(dict(item))
        constructor_kwargs: List[Dict[str, Any]] = []
        for item in model_raw.get("constructor_kwargs") or []:
            if isinstance(item, dict):
                constructor_kwargs.append(dict(item))
        model_spec = PluginModelSpec(
            module=_require_str(model_raw, "module", "manifest.model"),
            class_name=_require_str(model_raw, "class", "manifest.model"),
            pe_db_format=_require_str(model_raw, "pe_db_format", "manifest.model").lower(),
            weight_format=_require_str(model_raw, "weight_format", "manifest.model"),
            hyperparameters=tuple(hyperparameters),
            constructor_kwargs=tuple(constructor_kwargs),
        )

    weight_specs: List[PluginWeightSpec] = []
    for item in data.get("weights") or []:
        if not isinstance(item, dict):
            raise PluginError("manifest.weights entries must be mappings")
        weight_id = _require_str(item, "id", "manifest.weights")
        files = _optional_str_list(item, "files")
        notes = item.get("notes")
        weight_specs.append(
            PluginWeightSpec(
                id=weight_id,
                files=files,
                notes=str(notes).strip() if notes is not None else None,
            )
        )

    if model_spec is None:
        raise PluginError("manifest.model section is required")

    if format_spec is None and model_spec.pe_db_format in BUILTIN_FORMAT_NAMES:
        # Model consumes an existing PE-DB export format; no custom converter.
        pass
    elif format_spec is None:
        raise PluginError(
            "manifest.format is required when model.pe_db_format is not a built-in format"
        )

    return PluginManifest(
        name=name,
        version=version,
        display_name=display_name,
        description=description,
        authors=authors,
        format=format_spec,
        model=model_spec,
        weights=tuple(weight_specs),
        raw=dict(data),
    )


def parse_manifest(path: Path) -> PluginManifest:
    return parse_manifest_data(_load_yaml_or_json(path))


def parse_manifest_bytes(content: bytes) -> PluginManifest:
    """Parse manifest YAML/JSON bytes (e.g. from an upload)."""
    if not content.strip():
        raise PluginError("manifest file is empty")
    text = content.decode("utf-8")
    if text.lstrip().startswith("{"):
        payload = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise PluginError(
                "PyYAML is required to read manifest.yaml uploads. "
                "Install pyyaml or upload manifest.json."
            ) from exc
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise PluginError("manifest must be a YAML/JSON mapping")
    return parse_manifest_data(payload)


def load_manifest(plugin_dir: Path) -> PluginManifest:
    manifest_path = find_manifest_path(plugin_dir)
    if manifest_path is None:
        raise PluginError(f"No manifest.yaml or manifest.json in {plugin_dir}")
    manifest = parse_manifest(manifest_path)
    if manifest.name != plugin_dir.name.strip().lower():
        raise PluginError(
            f"Plugin directory '{plugin_dir.name}' must match manifest name '{manifest.name}'"
        )
    return manifest


def list_plugin_dirs(root: Optional[Path] = None) -> List[Path]:
    root = root or plugins_root()
    if not root.is_dir():
        return []
    dirs: List[Path] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and not child.name.startswith(".") and find_manifest_path(child):
            dirs.append(child)
    return dirs


def list_active_plugin_dirs(root: Optional[Path] = None) -> List[Path]:
    active: List[Path] = []
    for plugin_dir in list_plugin_dirs(root):
        state = read_plugin_state(plugin_dir)
        if state.get("status") == "active":
            active.append(plugin_dir)
    return active


def plugin_status(plugin_dir: Path) -> PluginStatus:
    return str(read_plugin_state(plugin_dir).get("status", "pending"))


def reject_plugin(plugin_dir: Path, report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    state = read_plugin_state(plugin_dir)
    state["status"] = "rejected"
    state["updated_at"] = _utc_now_iso()
    if report is not None:
        state["validation_report"] = report
    write_plugin_state(plugin_dir, state)
    return state


def set_plugin_pending(plugin_dir: Path) -> Dict[str, Any]:
    state = read_plugin_state(plugin_dir)
    state["status"] = "pending"
    state["updated_at"] = _utc_now_iso()
    state.pop("validation_report", None)
    write_plugin_state(plugin_dir, state)
    return state


def write_validation_report(plugin_dir: Path, report: Dict[str, Any]) -> Dict[str, Any]:
    state = read_plugin_state(plugin_dir)
    state["validation_report"] = report
    state["updated_at"] = _utc_now_iso()
    if report.get("passed"):
        state["status"] = "pending"
    else:
        state["status"] = "rejected"
    write_plugin_state(plugin_dir, state)
    return state


def build_manifest_yaml(payload: Dict[str, Any]) -> str:
    import yaml

    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


def activate_plugin(plugin_dir: Path) -> Dict[str, Any]:
    """Mark a plugin as active (used by tests and future validation gate)."""
    state = read_plugin_state(plugin_dir)
    state["status"] = "active"
    state["updated_at"] = _utc_now_iso()
    state["file_hashes"] = compute_plugin_file_hashes(plugin_dir)
    write_plugin_state(plugin_dir, state)
    return state


def load_module_from_file(module_name: str, file_path: Path) -> Any:
    import importlib.util

    file_path = file_path.resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"Plugin module not found: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise PluginError(f"Cannot import plugin module: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
