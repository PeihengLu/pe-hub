"""Load active model plugins into the PE Database format registry."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional, Set

import pandas as pd

from pe_common.plugins import (
    BUILTIN_FORMAT_NAMES,
    PluginError,
    PluginManifest,
    list_active_plugin_dirs,
    load_manifest,
    load_module_from_file,
    plugins_root,
)

from .format_registry import is_format_registered, register_format, unregister_format

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str], None]

_loaded_plugins: Set[str] = set()
_quarantined_plugins: Set[str] = set()
_plugin_format_names: Set[str] = set()


def loaded_plugin_names() -> tuple[str, ...]:
    return tuple(sorted(_loaded_plugins))


def quarantined_plugin_names() -> tuple[str, ...]:
    return tuple(sorted(_quarantined_plugins))


def _call_converter(
    fn: Callable[..., pd.DataFrame],
    df: pd.DataFrame,
    progress_callback: Optional[ProgressCallback] = None,
) -> pd.DataFrame:
    if progress_callback is not None:
        try:
            return fn(df, progress_callback=progress_callback)
        except TypeError:
            pass
    return fn(df)


def _wrap_plugin_converter(
    manifest: PluginManifest,
    convert_fn: Callable[..., pd.DataFrame],
) -> Callable[..., pd.DataFrame]:
    format_spec = manifest.format
    if format_spec is None:
        raise PluginError(f"Plugin '{manifest.name}' has no format spec")

    def wrapped(
        df: pd.DataFrame,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> pd.DataFrame:
        if format_spec.required_std_columns:
            missing = [
                col
                for col in format_spec.required_std_columns
                if col not in df.columns
            ]
            if missing:
                raise ValueError(
                    f"Plugin '{manifest.name}' converter missing standardized columns: "
                    f"{missing}"
                )
        converted = _call_converter(convert_fn, df, progress_callback)
        if len(converted) != len(df):
            raise ValueError(
                f"Plugin '{manifest.name}' converter changed row count "
                f"({len(df)} -> {len(converted)})"
            )
        if not converted.index.equals(df.index):
            converted = converted.copy()
            converted.index = df.index
        if format_spec.output_columns:
            missing_out = [
                col for col in format_spec.output_columns if col not in converted.columns
            ]
            if missing_out:
                raise ValueError(
                    f"Plugin '{manifest.name}' converter missing output columns: "
                    f"{missing_out}"
                )
        return converted

    return wrapped


def _register_plugin_format(plugin_dir: Path, manifest: PluginManifest) -> None:
    if manifest.format is None:
        logger.info(
            "Plugin '%s' uses built-in PE-DB format '%s'; skipping converter registration",
            manifest.name,
            manifest.model.pe_db_format,
        )
        return

    format_name = manifest.name
    if format_name in BUILTIN_FORMAT_NAMES:
        raise PluginError(
            f"Plugin format name '{format_name}' conflicts with a built-in format"
        )
    if is_format_registered(format_name):
        logger.info(
            "Format '%s' already registered; skipping plugin converter",
            format_name,
        )
        return

    module_path = plugin_dir / manifest.format.module
    module = load_module_from_file(
        f"pe_db_plugin_{manifest.name}_convert",
        module_path,
    )
    entrypoint = getattr(module, manifest.format.entrypoint, None)
    if entrypoint is None or not callable(entrypoint):
        raise PluginError(
            f"Plugin '{manifest.name}' entrypoint "
            f"'{manifest.format.entrypoint}' not found in {manifest.format.module}"
        )

    register_format(format_name, _wrap_plugin_converter(manifest, entrypoint))
    _plugin_format_names.add(format_name)
    logger.info("Registered plugin format '%s' from %s", format_name, plugin_dir)


def load_active_plugins(root: Optional[Path] = None) -> List[str]:
    """Scan ``PLUGINS_ROOT`` and register active plugin converters."""
    global _loaded_plugins, _quarantined_plugins
    root = root or plugins_root()
    loaded: List[str] = []

    for plugin_dir in list_active_plugin_dirs(root):
        name = plugin_dir.name
        if name in _loaded_plugins:
            loaded.append(name)
            continue
        try:
            manifest = load_manifest(plugin_dir)
            _register_plugin_format(plugin_dir, manifest)
            _loaded_plugins.add(manifest.name)
            _quarantined_plugins.discard(manifest.name)
            loaded.append(manifest.name)
        except Exception as exc:
            _quarantined_plugins.add(name)
            logger.error(
                "Failed to load PE-DB plugin '%s' from %s: %s",
                name,
                plugin_dir,
                exc,
                exc_info=True,
            )

    return loaded


def unregister_plugin(name: str) -> None:
    """Remove a plugin format from the registry if it was loaded from plugins."""
    global _loaded_plugins, _quarantined_plugins
    key = name.strip().lower()
    if key in _plugin_format_names and is_format_registered(key):
        unregister_format(key)
    _plugin_format_names.discard(key)
    _loaded_plugins.discard(key)
    _quarantined_plugins.discard(key)


def reload_active_plugins(root: Optional[Path] = None) -> List[str]:
    """Unregister plugin formats then reload all active plugins."""
    root = root or plugins_root()
    for name in list(_plugin_format_names):
        unregister_plugin(name)
    return load_active_plugins(root)
