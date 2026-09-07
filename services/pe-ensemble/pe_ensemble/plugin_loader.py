"""Load active model plugins into the PE Ensemble model registry."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Set

from pe_common.model_interface import BasePEModel
from pe_common.plugins import (
    PluginError,
    PluginManifest,
    list_active_plugin_dirs,
    load_manifest,
    load_module_from_file,
    plugins_root,
)

from .models.registry import ModelSpec, model_registry
from .models import weights_registry

logger = logging.getLogger(__name__)

_loaded_plugins: Set[str] = set()
_quarantined_plugins: Set[str] = set()


def loaded_plugin_names() -> tuple[str, ...]:
    return tuple(sorted(_loaded_plugins))


def quarantined_plugin_names() -> tuple[str, ...]:
    return tuple(sorted(_quarantined_plugins))


def _hyperparameter_architecture_builder(
    hyperparameters: tuple[Dict[str, Any], ...],
) -> Callable[[Mapping[str, Any]], Dict[str, Any]]:
    keys = [str(item.get("name", "")).strip() for item in hyperparameters]
    keys = [key for key in keys if key]

    def builder(values: Mapping[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key in keys:
            if key in values and values[key] is not None:
                out[key] = values[key]
        return out

    return builder


def _import_wrapper_class(plugin_dir: Path, manifest: PluginManifest) -> type:
    if manifest.model is None:
        raise PluginError(f"Plugin '{manifest.name}' is missing manifest.model")
    module_path = plugin_dir / manifest.model.module
    module = load_module_from_file(
        f"pe_ensemble_plugin_{manifest.name}_wrapper",
        module_path,
    )
    wrapper_class = getattr(module, manifest.model.class_name, None)
    if wrapper_class is None:
        raise PluginError(
            f"Plugin '{manifest.name}' class '{manifest.model.class_name}' "
            f"not found in {manifest.model.module}"
        )
    if not isinstance(wrapper_class, type) or not issubclass(wrapper_class, BasePEModel):
        raise PluginError(
            f"Plugin '{manifest.name}' class '{manifest.model.class_name}' "
            "must subclass BasePEModel"
        )
    return wrapper_class


def _register_plugin_weights(plugin_dir: Path, manifest: PluginManifest) -> None:
    if manifest.model is None:
        return
    model_name = manifest.name
    weight_format = manifest.model.weight_format

    for weight_spec in manifest.weights:
        existing = set(weights_registry.list_weight_ids(model_name))
        if weight_spec.id in existing:
            logger.info(
                "Plugin weight '%s' for model '%s' already registered; skipping",
                weight_spec.id,
                model_name,
            )
            continue

        if weight_spec.files:
            # Copy listed files into a temp staging dir under plugin weights/
            src_dir = plugin_dir / "weights" / weight_spec.id
        else:
            src_dir = plugin_dir / "weights" / weight_spec.id

        if not src_dir.is_dir():
            raise FileNotFoundError(
                f"Plugin weight directory not found: {src_dir}"
            )

        weights_registry.register_from_directory(
            model_name,
            src_dir,
            weight_id=weight_spec.id,
            label=weight_spec.id,
            source="plugin",
            format_name=weight_format,
            metadata={
                "provenance": {"plugin": manifest.name, "version": manifest.version},
                "notes": weight_spec.notes,
            },
            rebuild=False,
        )

    if manifest.weights:
        weights_registry.rebuild_index()


def _register_plugin_model(plugin_dir: Path, manifest: PluginManifest) -> None:
    if manifest.model is None:
        raise PluginError(f"Plugin '{manifest.name}' is missing manifest.model")

    if model_registry.is_registered(manifest.name):
        existing = model_registry.get(manifest.name)
        if existing.source == "builtin":
            raise PluginError(
                f"Plugin name '{manifest.name}' conflicts with a built-in model"
            )
        logger.info("Plugin model '%s' already registered; refreshing weights", manifest.name)
        _register_plugin_weights(plugin_dir, manifest)
        return

    wrapper_class = _import_wrapper_class(plugin_dir, manifest)
    hyperparameters = manifest.model.hyperparameters
    architecture_builder = _hyperparameter_architecture_builder(hyperparameters)

    model_registry.register(
        ModelSpec(
            name=manifest.name,
            wrapper_class=wrapper_class,
            display_name=manifest.display_name,
            description=manifest.description,
            model_type="plugin",
            pe_db_format=manifest.model.pe_db_format,
            weight_format=manifest.model.weight_format,
            architecture_builder=architecture_builder,
            source="plugin",
        )
    )
    _register_plugin_weights(plugin_dir, manifest)
    logger.info("Registered plugin model '%s' from %s", manifest.name, plugin_dir)


def load_active_plugins(root: Optional[Path] = None) -> List[str]:
    """Scan ``PLUGINS_ROOT`` and register active plugin models and weights."""
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
            _register_plugin_model(plugin_dir, manifest)
            _loaded_plugins.add(manifest.name)
            _quarantined_plugins.discard(manifest.name)
            loaded.append(manifest.name)
        except Exception as exc:
            _quarantined_plugins.add(name)
            logger.error(
                "Failed to load PE Ensemble plugin '%s' from %s: %s",
                name,
                plugin_dir,
                exc,
                exc_info=True,
            )

    return loaded


def unregister_plugin(name: str) -> None:
    """Remove a plugin model from the registry if it was loaded from plugins."""
    global _loaded_plugins, _quarantined_plugins
    key = name.strip().lower()
    if model_registry.is_registered(key):
        spec = model_registry.get(key)
        if spec.source == "plugin":
            model_registry.unregister(key)
    _loaded_plugins.discard(key)
    _quarantined_plugins.discard(key)


def reload_active_plugins(root: Optional[Path] = None) -> List[str]:
    """Unregister plugin models then reload all active plugins."""
    root = root or plugins_root()
    for name in list(_loaded_plugins):
        unregister_plugin(name)
    return load_active_plugins(root)
