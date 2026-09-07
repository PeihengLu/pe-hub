"""Load and resolve dataset-specific training hyperparameter presets.

Shipped YAML under ``config/training_presets`` holds shared model defaults
(tracked in git). Optuna writes dataset hits under
``config/training_presets_local`` (gitignored). Resolution merges local over
shipped. Pass ``preset_root`` to use a single root (tests / explicit override).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .config import local_presets_root, shipped_presets_root
from .dataset_key import candidate_preset_keys, filters_from_request
from .model_baselines import model_baseline_hyperparameters
from .model_architecture import apply_fine_tune_defaults

# Scheduler keys are user-controlled during training and excluded from Optuna presets.
SCHEDULER_KEYS = frozenset({"scheduler", "scheduler_kwargs"})


@dataclass(frozen=True)
class ResolvedHyperparameters:
    hyperparameters: Dict[str, Any]
    preset_key: Optional[str]
    preset_source: str


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency declared in pyproject
        raise RuntimeError(
            "PyYAML is required for training presets. Install pe-ensemble with pyyaml."
        ) from exc
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Preset file must be a mapping: {path}")
    return data


def preset_path_for_model(model_name: str, *, root: Optional[Path] = None) -> Path:
    base = root if root is not None else local_presets_root()
    return base / f"{model_name.strip().lower()}.yaml"


def load_preset_bundle(
    model_name: str,
    *,
    root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load preset YAML for a model. Missing files yield an empty bundle."""
    path = preset_path_for_model(model_name, root=root)
    if not path.is_file():
        return {}
    return _load_yaml(path)


def _dataset_entry_hyperparameters(entry: Any) -> Dict[str, Any]:
    if not isinstance(entry, dict):
        return {}
    hyperparameters = entry.get("hyperparameters")
    if isinstance(hyperparameters, dict):
        return dict(hyperparameters)
    # Allow shorthand: datasets.<key>.lr: 1e-4 at top level of entry
    return {
        key: value
        for key, value in entry.items()
        if key != "provenance" and not str(key).startswith("_")
    }


def lookup_dataset_preset(
    bundle: Mapping[str, Any],
    *,
    study: Any = None,
    dataset: Any = None,
    cell_line: Any = None,
    pe_system: Any = None,
) -> tuple[Optional[str], Dict[str, Any]]:
    datasets = bundle.get("datasets") or {}
    if not isinstance(datasets, dict):
        return None, {}
    for key in candidate_preset_keys(
        study=study,
        dataset=dataset,
        cell_line=cell_line,
        pe_system=pe_system,
    ):
        entry = datasets.get(key)
        if entry is None:
            continue
        hyperparameters = _dataset_entry_hyperparameters(entry)
        if hyperparameters:
            return key, hyperparameters
    return None, {}


def _strip_scheduler_keys(values: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in values.items() if key not in SCHEDULER_KEYS}


def _bundle_defaults(bundle: Mapping[str, Any]) -> Dict[str, Any]:
    defaults = bundle.get("defaults")
    return dict(defaults) if isinstance(defaults, dict) else {}


def merge_hyperparameter_layers(
    *layers: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for layer in layers:
        if not layer:
            continue
        for key, value in layer.items():
            if value is None:
                continue
            merged[key] = value
    return merged


def resolve_hyperparameters(
    model_name: str,
    *,
    study: Any = None,
    dataset: Any = None,
    cell_line: Any = None,
    pe_system: Any = None,
    user_overrides: Optional[Mapping[str, Any]] = None,
    mode: str = "merge",
    preset_root: Optional[Path] = None,
    shipped_root: Optional[Path] = None,
    local_root: Optional[Path] = None,
) -> ResolvedHyperparameters:
    """Merge baseline, shipped + local YAML presets, and user overrides.

    When ``preset_root`` is set, only that root is used (single-file / test mode).
    Otherwise shipped defaults/datasets load first, then local overlays them.

    Scheduler keys are never loaded from presets. Users may set them via overrides.
    """
    name = model_name.strip().lower()
    baseline = model_baseline_hyperparameters(name)

    if mode == "replace":
        merged = merge_hyperparameter_layers(baseline, user_overrides)
        return ResolvedHyperparameters(
            hyperparameters=apply_fine_tune_defaults(merged),
            preset_key=None,
            preset_source="replace",
        )

    lookup_kwargs = {
        "study": study,
        "dataset": dataset,
        "cell_line": cell_line,
        "pe_system": pe_system,
    }

    if preset_root is not None:
        bundle = load_preset_bundle(name, root=preset_root)
        model_defaults = _bundle_defaults(bundle)
        preset_key, dataset_preset = lookup_dataset_preset(bundle, **lookup_kwargs)
        layers = [
            baseline,
            _strip_scheduler_keys(model_defaults),
            _strip_scheduler_keys(dataset_preset),
            user_overrides,
        ]
        source = "baseline"
        if model_defaults:
            source = "model_defaults"
        if preset_key:
            source = f"preset:{preset_key}"
        if user_overrides:
            source = f"{source}+user"
        return ResolvedHyperparameters(
            hyperparameters=apply_fine_tune_defaults(
                merge_hyperparameter_layers(*layers)
            ),
            preset_key=preset_key,
            preset_source=source,
        )

    shipped_bundle = load_preset_bundle(
        name, root=shipped_root if shipped_root is not None else shipped_presets_root()
    )
    local_bundle = load_preset_bundle(
        name, root=local_root if local_root is not None else local_presets_root()
    )
    shipped_defaults = _bundle_defaults(shipped_bundle)
    local_defaults = _bundle_defaults(local_bundle)
    shipped_key, shipped_dataset = lookup_dataset_preset(shipped_bundle, **lookup_kwargs)
    local_key, local_dataset = lookup_dataset_preset(local_bundle, **lookup_kwargs)

    layers = [
        baseline,
        _strip_scheduler_keys(shipped_defaults),
        _strip_scheduler_keys(local_defaults),
        _strip_scheduler_keys(shipped_dataset),
        _strip_scheduler_keys(local_dataset),
        user_overrides,
    ]
    merged = merge_hyperparameter_layers(*layers)

    preset_key = local_key or shipped_key
    source = "baseline"
    if shipped_defaults or local_defaults:
        source = "model_defaults"
    if shipped_key and not local_key:
        source = f"preset:{shipped_key}"
    if local_key:
        source = f"local_preset:{local_key}"
    if user_overrides:
        source = f"{source}+user"

    return ResolvedHyperparameters(
        hyperparameters=apply_fine_tune_defaults(merged),
        preset_key=preset_key,
        preset_source=source,
    )


def resolve_hyperparameters_for_request(
    request: Any,
    *,
    preset_root: Optional[Path] = None,
) -> ResolvedHyperparameters:
    """Resolve hyperparameters from a :class:`TrainingRequest`."""
    filters = filters_from_request(request)
    mode = str(getattr(request, "hyperparameter_mode", "merge") or "merge")
    return resolve_hyperparameters(
        request.model_name,
        study=filters["study"],
        dataset=filters["dataset"],
        cell_line=filters["cell_line"],
        pe_system=filters["pe_system"],
        user_overrides=request.hyperparameters,
        mode=mode,
        preset_root=preset_root,
    )


def write_dataset_preset(
    path: Path,
    *,
    model_name: str,
    dataset_key: str,
    hyperparameters: Mapping[str, Any],
    provenance: Optional[Mapping[str, Any]] = None,
) -> None:
    """Insert or update one dataset entry in a preset YAML file."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to write training presets.") from exc

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.is_file():
        bundle = _load_yaml(path)
    else:
        bundle = {
            "schema_version": 1,
            "model": model_name.strip().lower(),
            "defaults": {},
            "datasets": {},
        }

    if "datasets" not in bundle or not isinstance(bundle["datasets"], dict):
        bundle["datasets"] = {}

    entry: Dict[str, Any] = {
        "hyperparameters": _strip_scheduler_keys(dict(hyperparameters)),
    }
    if provenance:
        entry["provenance"] = dict(provenance)

    bundle["model"] = model_name.strip().lower()
    bundle["datasets"][dataset_key] = entry

    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(bundle, handle, sort_keys=False, default_flow_style=False)
