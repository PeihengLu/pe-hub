"""Plugin upload, validation, activation, and removal."""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from pe_common.plugin_validation import validate_plugin_directory
from pe_common.plugins import (
    BUILTIN_FORMAT_NAMES,
    PluginError,
    PluginManifest,
    activate_plugin,
    build_manifest_yaml,
    find_manifest_path,
    list_plugin_dirs,
    load_manifest,
    parse_manifest,
    parse_manifest_bytes,
    plugin_status,
    plugins_root,
    read_plugin_state,
    set_plugin_pending,
    validate_plugin_name,
    write_validation_report,
)

from ..plugin_loader import reload_active_plugins, unregister_plugin
from ..training.pe_db_access import PeDbAccessError, reload_pe_db_plugins

logger = logging.getLogger(__name__)

VALIDATION_LOG_FILENAME = "validation.log"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def _plugin_dir(name: str) -> Path:
    key = validate_plugin_name(name)
    return plugins_root() / key


def _parse_csv_list(value: Optional[str]) -> List[str]:
    if not value or not value.strip():
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_authors(value: Optional[str]) -> List[str]:
    if not value or not value.strip():
        return []
    stripped = value.strip()
    if stripped.startswith("["):
        parsed = json.loads(stripped)
        if not isinstance(parsed, list):
            raise PluginError("authors must be a JSON array of strings")
        return [str(item).strip() for item in parsed if str(item).strip()]
    return _parse_csv_list(value)


def _parse_json_list(value: Optional[str], field_name: str) -> List[Dict[str, Any]]:
    if not value or not value.strip():
        return []
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise PluginError(f"{field_name} must be a JSON array")
    return [dict(item) for item in parsed if isinstance(item, dict)]


def _summary_from_dir(plugin_dir: Path) -> Dict[str, Any]:
    state = read_plugin_state(plugin_dir)
    report = state.get("validation_report") or {}
    manifest_path = find_manifest_path(plugin_dir)
    version = ""
    display_name = plugin_dir.name
    if manifest_path is not None:
        try:
            manifest = load_manifest(plugin_dir)
            version = manifest.version
            display_name = manifest.display_name
        except PluginError:
            pass
    checks = report.get("checks") or []
    return {
        "name": plugin_dir.name,
        "status": state.get("status", "pending"),
        "version": version,
        "display_name": display_name,
        "updated_at": state.get("updated_at"),
        "validation_passed": report.get("passed"),
        "validated_at": report.get("validated_at"),
        "check_count": len(checks),
        "failed_checks": [
            check.get("id")
            for check in checks
            if isinstance(check, dict) and not check.get("passed")
        ],
    }


def list_plugins() -> List[Dict[str, Any]]:
    return [_summary_from_dir(plugin_dir) for plugin_dir in list_plugin_dirs()]


def get_plugin(name: str) -> Dict[str, Any]:
    plugin_dir = _plugin_dir(name)
    if not plugin_dir.is_dir():
        raise FileNotFoundError(f"Plugin not found: {name}")
    state = read_plugin_state(plugin_dir)
    manifest_path = find_manifest_path(plugin_dir)
    manifest_raw: Dict[str, Any] = {}
    if manifest_path is not None:
        manifest = load_manifest(plugin_dir)
        manifest_raw = dict(manifest.raw)
    log_path = plugin_dir / VALIDATION_LOG_FILENAME
    return {
        "name": plugin_dir.name,
        "status": state.get("status", "pending"),
        "updated_at": state.get("updated_at"),
        "file_hashes": state.get("file_hashes"),
        "manifest": manifest_raw,
        "validation_report": state.get("validation_report"),
        "validation_log_exists": log_path.is_file(),
    }


def _assert_upload_size(data: bytes, label: str) -> None:
    if len(data) > MAX_UPLOAD_BYTES:
        raise PluginError(f"{label} exceeds upload size limit ({MAX_UPLOAD_BYTES} bytes)")


def _write_bytes(path: Path, data: bytes) -> None:
    _assert_upload_size(data, path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _safe_zip_extract(zip_path: Path, dest_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            member_path = Path(member)
            if member.startswith("/") or ".." in member_path.parts:
                raise PluginError(f"Unsafe path in plugin zip: {member}")
        archive.extractall(dest_dir)


def _plugin_content_root(staging: Path) -> Path:
    """If the zip contains a single top-level plugin directory, use it as the root."""
    entries = [
        path
        for path in staging.iterdir()
        if path.name not in {".", ".."} and not path.name.startswith(".")
    ]
    if len(entries) == 1 and entries[0].is_dir():
        nested = entries[0]
        if find_manifest_path(nested) is not None or (nested / "wrapper.py").is_file():
            return nested
    return staging


def _extract_bundle_zip(bundle_zip_bytes: bytes) -> tuple[Path, Path]:
    """Extract a plugin zip to a temp directory; return (content_root, staging_dir)."""
    _assert_upload_size(bundle_zip_bytes, "bundle.zip")
    staging = Path(tempfile.mkdtemp(prefix="pe-plugin-upload-"))
    zip_path = staging / "bundle.zip"
    zip_path.write_bytes(bundle_zip_bytes)
    _safe_zip_extract(zip_path, staging)
    zip_path.unlink(missing_ok=True)
    return _plugin_content_root(staging), staging


def _copy_plugin_contents(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name.startswith(".") and item.name not in {".state.json"}:
            continue
        target = dest / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def _assert_plugin_artifacts(plugin_dir: Path, manifest: PluginManifest) -> None:
    if manifest.model is None:
        raise PluginError("manifest.model section is required")
    wrapper_path = plugin_dir / manifest.model.module
    if not wrapper_path.is_file():
        raise PluginError(f"Missing model module file: {manifest.model.module}")
    if manifest.format is not None:
        convert_path = plugin_dir / manifest.format.module
        if not convert_path.is_file():
            raise PluginError(f"Missing format module file: {manifest.format.module}")


def _resolve_plugin_key(name: Optional[str], manifest: PluginManifest) -> str:
    key = manifest.name
    if name and name.strip():
        requested = validate_plugin_name(name)
        if requested != key:
            raise PluginError(
                f"manifest name '{key}' must match upload name '{requested}'"
            )
    return key


def _read_manifest_from_dir(content_root: Path) -> Optional[PluginManifest]:
    manifest_path = find_manifest_path(content_root)
    if manifest_path is None:
        return None
    return parse_manifest(manifest_path)


def _build_manifest_dict(
    *,
    name: str,
    version: str,
    display_name: str,
    description: str,
    authors: Sequence[str],
    wrapper_class: str,
    convert_entrypoint: str,
    pe_db_format: str,
    weight_format: str,
    output_columns: Sequence[str],
    required_std_columns: Sequence[str],
    label_column: Optional[str],
    hyperparameters: Sequence[Dict[str, Any]],
    weights: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {
        "name": name,
        "version": version,
        "display_name": display_name,
        "description": description,
        "authors": list(authors),
        "model": {
            "module": "wrapper.py",
            "class": wrapper_class,
            "pe_db_format": pe_db_format,
            "weight_format": weight_format,
        },
    }
    if hyperparameters:
        manifest["model"]["hyperparameters"] = list(hyperparameters)
    if weights:
        manifest["weights"] = weights
    if pe_db_format not in BUILTIN_FORMAT_NAMES:
        format_block: Dict[str, Any] = {
            "module": "convert.py",
            "entrypoint": convert_entrypoint,
        }
        if required_std_columns:
            format_block["required_std_columns"] = list(required_std_columns)
        if output_columns:
            format_block["output_columns"] = list(output_columns)
        if label_column:
            format_block["label_column"] = label_column
        manifest["format"] = format_block
    return manifest


def upload_plugin_bundle(
    *,
    name: Optional[str] = None,
    version: str = "0.1.0",
    display_name: str = "",
    description: str = "",
    authors: Optional[str] = None,
    wrapper_class: str = "",
    convert_entrypoint: str = "convert",
    pe_db_format: Optional[str] = None,
    weight_format: str = "",
    output_columns: Optional[str] = None,
    required_std_columns: Optional[str] = None,
    label_column: Optional[str] = None,
    hyperparameters_json: Optional[str] = None,
    weights_json: Optional[str] = None,
    convert_bytes: Optional[bytes] = None,
    wrapper_bytes: Optional[bytes] = None,
    bundle_zip_bytes: Optional[bytes] = None,
    manifest_bytes: Optional[bytes] = None,
    weight_uploads: Optional[Sequence[Tuple[str, bytes]]] = None,
    replace_existing: bool = False,
) -> Dict[str, Any]:
    """Write a new plugin directory in ``pending`` state.

    Bundle-first upload: provide ``bundle_zip`` and/or ``manifest_bytes`` with code
    files. Plugin name and metadata come from ``manifest.yaml``. Form fields are
    only used when no manifest is supplied.
    """
    staging_dir: Optional[Path] = None
    content_root: Optional[Path] = None
    plugin_dir: Optional[Path] = None

    try:
        if bundle_zip_bytes is not None:
            content_root, staging_dir = _extract_bundle_zip(bundle_zip_bytes)

        manifest: Optional[PluginManifest] = None
        manifest_write_bytes: Optional[bytes] = None

        if manifest_bytes is not None:
            manifest = parse_manifest_bytes(manifest_bytes)
            manifest_write_bytes = manifest_bytes
        elif content_root is not None:
            manifest = _read_manifest_from_dir(content_root)
            if manifest is None:
                raise PluginError(
                    "bundle zip must include manifest.yaml or manifest.json "
                    "(or upload manifest_file separately)"
                )

        using_form_manifest = manifest is None
        if using_form_manifest:
            if not name or not name.strip():
                raise PluginError(
                    "Plugin name is required when manifest.yaml is not provided"
                )
            if not wrapper_class.strip():
                raise PluginError("wrapper_class is required when manifest.yaml is not provided")
            if not weight_format.strip():
                raise PluginError("weight_format is required when manifest.yaml is not provided")
            key = validate_plugin_name(name)
        else:
            assert manifest is not None
            key = _resolve_plugin_key(name, manifest)

        plugin_dir = plugins_root() / key
        root = plugins_root()
        root.mkdir(parents=True, exist_ok=True)

        if plugin_dir.is_dir():
            status = plugin_status(plugin_dir)
            if status == "active" and not replace_existing:
                raise PluginError(
                    f"Plugin '{key}' is active. Deactivate/delete it before uploading again."
                )
            if status == "pending" and not replace_existing:
                raise PluginError(
                    f"Plugin '{key}' already exists. Delete it or pass replace_existing."
                )
            shutil.rmtree(plugin_dir)

        plugin_dir.mkdir(parents=True, exist_ok=False)

        if content_root is not None:
            _copy_plugin_contents(content_root, plugin_dir)
        else:
            if convert_bytes is None or wrapper_bytes is None:
                raise PluginError("convert.py and wrapper.py are required when no zip is provided")
            _write_bytes(plugin_dir / "convert.py", convert_bytes)
            _write_bytes(plugin_dir / "wrapper.py", wrapper_bytes)

        if using_form_manifest:
            resolved_pe_db_format = (pe_db_format or key).strip().lower()
            manifest_dict = _build_manifest_dict(
                name=key,
                version=version.strip(),
                display_name=display_name.strip() or key,
                description=description.strip() or "Plugin model",
                authors=_parse_authors(authors),
                wrapper_class=wrapper_class.strip(),
                convert_entrypoint=convert_entrypoint.strip() or "convert",
                pe_db_format=resolved_pe_db_format,
                weight_format=weight_format.strip(),
                output_columns=_parse_csv_list(output_columns),
                required_std_columns=_parse_csv_list(required_std_columns),
                label_column=label_column.strip() if label_column else None,
                hyperparameters=_parse_json_list(hyperparameters_json, "hyperparameters"),
                weights=_parse_json_list(weights_json, "weights"),
            )
            manifest_yaml = build_manifest_yaml(manifest_dict)
            (plugin_dir / "manifest.yaml").write_text(manifest_yaml, encoding="utf-8")
        elif manifest_write_bytes is not None:
            (plugin_dir / "manifest.yaml").write_bytes(manifest_write_bytes)

        if weight_uploads:
            for weight_id, payload in weight_uploads:
                weight_key = weight_id.strip()
                if not weight_key:
                    continue
                weight_dir = plugin_dir / "weights" / weight_key
                weight_dir.mkdir(parents=True, exist_ok=True)
                _write_bytes(weight_dir / "weights.txt", payload)

        final_manifest = load_manifest(plugin_dir)
        _assert_plugin_artifacts(plugin_dir, final_manifest)

        set_plugin_pending(plugin_dir)
        state = read_plugin_state(plugin_dir)
        return {
            "name": key,
            "status": state.get("status"),
            "message": "Plugin uploaded as pending",
        }
    except Exception:
        if plugin_dir is not None and plugin_dir.is_dir():
            shutil.rmtree(plugin_dir, ignore_errors=True)
        raise
    finally:
        if staging_dir is not None:
            shutil.rmtree(staging_dir, ignore_errors=True)


def _append_validation_log(plugin_dir: Path, message: str) -> None:
    log_path = plugin_dir / VALIDATION_LOG_FILENAME
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def _log_validation(
    plugin_dir: Path,
    message: str,
    log_line: Optional[Callable[[str], None]] = None,
) -> None:
    _append_validation_log(plugin_dir, message)
    if log_line is not None:
        log_line(message)


def execute_validation(
    name: str,
    *,
    log_line: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run the validation harness and persist results (blocking worker entrypoint)."""
    plugin_dir = _plugin_dir(name)
    if not plugin_dir.is_dir():
        raise FileNotFoundError(f"Plugin not found: {name}")

    log_path = plugin_dir / VALIDATION_LOG_FILENAME
    log_path.write_text("", encoding="utf-8")
    _log_validation(plugin_dir, f"Validation started for {plugin_dir.name}", log_line)

    try:
        report = validate_plugin_directory(plugin_dir)
        _log_validation(
            plugin_dir,
            f"Validation finished: passed={report.passed} ({len(report.checks)} checks)",
            log_line,
        )
        for check in report.checks:
            _log_validation(
                plugin_dir,
                f"[{'PASS' if check.passed else 'FAIL'}] {check.id}: {check.detail}",
                log_line,
            )
    except Exception as exc:
        _log_validation(plugin_dir, f"Validation crashed: {exc}", log_line)
        report_dict = {
            "plugin_name": plugin_dir.name,
            "passed": False,
            "validated_at": None,
            "checks": [
                {
                    "id": "validation_runner",
                    "passed": False,
                    "detail": str(exc),
                    "duration_ms": 0.0,
                }
            ],
        }
        state = write_validation_report(plugin_dir, report_dict)
        return {
            "plugin_name": plugin_dir.name,
            "validation_report": report_dict,
            "status": state.get("status"),
        }

    report_dict = report.to_dict()
    state = write_validation_report(plugin_dir, report_dict)
    return {
        "plugin_name": plugin_dir.name,
        "validation_report": report_dict,
        "status": state.get("status"),
    }


def run_validation(name: str) -> Dict[str, Any]:
    """Synchronous validation (tests and direct calls)."""
    result = execute_validation(name)
    return {
        "validation_report": result["validation_report"],
        "status": result["status"],
    }


def queue_validation(name: str) -> Dict[str, Any]:
    """Queue an asynchronous validation job for a plugin."""
    plugin_dir = _plugin_dir(name)
    if not plugin_dir.is_dir():
        raise FileNotFoundError(f"Plugin not found: {name}")

    from .scheduler import get_validation_scheduler
    from .validation_jobs import get_job

    job_id = get_validation_scheduler().submit(name)
    manifest = get_job(job_id)
    return {
        "job_id": job_id,
        "plugin_name": manifest["plugin_name"],
        "status": manifest["status"],
        "message": "Validation job queued",
    }


def read_validation_log(name: str, offset: int = 0) -> Tuple[str, int]:
    plugin_dir = _plugin_dir(name)
    if not plugin_dir.is_dir():
        raise FileNotFoundError(f"Plugin not found: {name}")
    log_path = plugin_dir / VALIDATION_LOG_FILENAME
    if not log_path.is_file():
        return "", offset
    data = log_path.read_bytes()
    if offset > len(data):
        offset = len(data)
    chunk = data[offset:].decode("utf-8", errors="replace")
    return chunk, len(data)


def notify_pe_db_plugin_reload() -> Dict[str, Any]:
    try:
        return reload_pe_db_plugins()
    except PeDbAccessError as exc:
        raise PluginError(str(exc)) from exc


def activate_plugin_bundle(name: str) -> Dict[str, Any]:
    plugin_dir = _plugin_dir(name)
    if not plugin_dir.is_dir():
        raise FileNotFoundError(f"Plugin not found: {name}")

    state = read_plugin_state(plugin_dir)
    report = state.get("validation_report") or {}
    if not report.get("passed"):
        raise PluginError(
            "Plugin validation has not passed. Run validate before activate."
        )

    activate_plugin(plugin_dir)
    loaded = reload_active_plugins()
    pe_db_result: Dict[str, Any] = {}
    try:
        pe_db_result = notify_pe_db_plugin_reload()
    except PluginError as exc:
        logger.warning("PE-DB plugin reload failed after activation: %s", exc)
        pe_db_result = {"error": str(exc)}

    final_state = read_plugin_state(plugin_dir)
    return {
        "name": plugin_dir.name,
        "status": final_state.get("status"),
        "ensemble_loaded": loaded,
        "pe_db_reload": pe_db_result,
    }


def delete_plugin_bundle(name: str) -> Dict[str, Any]:
    plugin_dir = _plugin_dir(name)
    if not plugin_dir.is_dir():
        raise FileNotFoundError(f"Plugin not found: {name}")

    status = plugin_status(plugin_dir)
    if status == "active":
        unregister_plugin(plugin_dir.name)
        try:
            notify_pe_db_plugin_reload()
        except PluginError as exc:
            logger.warning("PE-DB plugin reload failed after delete: %s", exc)

    shutil.rmtree(plugin_dir)
    return {"name": plugin_dir.name, "deleted": True}
