"""Plugin validation harness (correctness gate before activation)."""
from __future__ import annotations

import math
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from pe_common.constants import PROJECT_ROOT
from pe_common.model_interface import BasePEModel
from pe_common.plugins import (
    PluginError,
    PluginManifest,
    list_plugin_dirs,
    load_manifest,
    load_module_from_file,
    plugins_root,
    validate_plugin_name,
)

DEFAULT_SMOKE_TIMEOUT_SECONDS = 120.0
EVAL_METRIC_KEYS = ("pearson", "spearman")


@dataclass
class ValidationCheckResult:
    id: str
    passed: bool
    detail: str
    duration_ms: float


@dataclass
class ValidationReport:
    plugin_name: str
    passed: bool
    checks: List[ValidationCheckResult]
    validated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plugin_name": self.plugin_name,
            "passed": self.passed,
            "validated_at": self.validated_at,
            "checks": [asdict(check) for check in self.checks],
        }


@dataclass
class _ValidationState:
    convert_fn: Optional[Callable[..., pd.DataFrame]] = None
    wrapper_class: Optional[type] = None
    saved_artifact: Optional[Path] = None


def _smoke_timeout_seconds() -> float:
    raw = os.getenv("VALIDATION_SMOKE_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return DEFAULT_SMOKE_TIMEOUT_SECONDS
    try:
        return max(1.0, float(raw))
    except ValueError:
        return DEFAULT_SMOKE_TIMEOUT_SECONDS


def _standardized_fixture_path() -> Path:
    return (PROJECT_ROOT / "testdata" / "vendor_eval" / "standardized_small.csv").resolve()


def _load_standardized_fixture() -> pd.DataFrame:
    fixture_path = _standardized_fixture_path()
    if fixture_path.is_file():
        df = pd.read_csv(fixture_path)
        if df.empty:
            raise ValueError(f"standardized fixture is empty: {fixture_path}")
        return df
    return _inline_standardized_fixture()


def _inline_standardized_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "wt_sequence": ["ACGTACGT", "TGCATGCA"],
            "mut_sequence": ["ACGTACGT", "TGCATGCA"],
            "editing_efficiency": [0.2, 0.5],
            "edit_len": [1, 3],
            "type_sub": [True, False],
            "type_ins": [False, False],
            "type_del": [False, True],
            "protospacer_location_l": [1, 1],
            "protospacer_location_r": [4, 4],
            "pbs_location_l": [5, 5],
            "pbs_location_r": [6, 6],
            "rtt_location_l": [6, 6],
            "rtt_location_r": [7, 7],
            "lha_location_r": [7, 7],
        }
    )


def _ensure_std_columns(
    std_df: pd.DataFrame,
    required_columns: tuple[str, ...],
) -> pd.DataFrame:
    if not required_columns:
        return std_df
    missing = [col for col in required_columns if col not in std_df.columns]
    if not missing:
        return std_df
    filled = std_df.copy()
    for col in missing:
        filled[col] = 0
    return filled


def _label_column(manifest: PluginManifest) -> str:
    if manifest.format is not None and manifest.format.label_column:
        return manifest.format.label_column
    return "Efficiency"


def _build_train_df(
    manifest: PluginManifest,
    convert_fn: Optional[Callable[..., pd.DataFrame]],
) -> pd.DataFrame:
    std_df = _load_standardized_fixture().head(2)
    if manifest.format is not None and convert_fn is not None:
        std_df = _ensure_std_columns(std_df, manifest.format.required_std_columns)
        return convert_fn(std_df)
    label = _label_column(manifest)
    return pd.DataFrame({"feature": [1.0, 2.0], label: [0.1, 0.2]})


def _build_eval_df(
    manifest: PluginManifest,
    convert_fn: Optional[Callable[..., pd.DataFrame]],
) -> pd.DataFrame:
    std_df = _load_standardized_fixture().head(2)
    if manifest.format is not None and convert_fn is not None:
        std_df = _ensure_std_columns(std_df, manifest.format.required_std_columns)
        return convert_fn(std_df)
    label = _label_column(manifest)
    return pd.DataFrame({"feature": [1.0, 3.0], label: [0.1, 0.3]})


def _find_weight_artifact(directory: Path) -> Path:
    directory = directory.resolve()
    preferred = directory / "weights.txt"
    if preferred.is_file():
        return preferred
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name not in {".state.json", "manifest.yaml", "manifest.json"}:
            return path
    raise FileNotFoundError(f"No weight artifact found under {directory}")


def _resolve_cpu_device() -> Any:
    try:
        from pe_common.devices import resolve_device

        return resolve_device("cpu")
    except Exception:
        return None


def _create_wrapper(wrapper_class: type) -> BasePEModel:
    device = _resolve_cpu_device()
    if device is not None:
        try:
            return wrapper_class(device=device)
        except TypeError:
            pass
    return wrapper_class()


def _patch_load_weights(
    wrapper: BasePEModel,
    plugin_dir: Path,
    fallback_artifact: Optional[Path],
) -> None:
    def patched_load_weights(name: str) -> None:
        weight_dir = plugin_dir / "weights" / name
        artifact = weight_dir / "weights.txt"
        if artifact.is_file():
            wrapper.load_model(str(artifact))
            return
        if fallback_artifact is not None and fallback_artifact.is_file():
            wrapper.load_model(str(fallback_artifact))
            return
        raise FileNotFoundError(
            f"Validation could not resolve weights '{name}' in {plugin_dir} "
            f"or fallback artifact {fallback_artifact}"
        )

    if hasattr(wrapper, "load_weights_by_name"):
        wrapper.load_weights_by_name = patched_load_weights  # type: ignore[method-assign]


def _assert_eval_metrics(metrics: Dict[str, Any]) -> str:
    if not isinstance(metrics, dict):
        raise TypeError(f"evaluate() must return a dict, got {type(metrics).__name__}")
    n_samples = metrics.get("n_samples", 2)
    for key in EVAL_METRIC_KEYS:
        if key not in metrics:
            raise KeyError(f"evaluate() missing metric '{key}'")
        value = metrics[key]
        if not isinstance(value, (int, float)):
            raise TypeError(f"metric '{key}' must be numeric")
        if n_samples >= 2 and not math.isfinite(float(value)):
            raise ValueError(f"metric '{key}' is not finite: {value}")
    return f"pearson={metrics['pearson']}, spearman={metrics['spearman']}"


def _ensure_saved_artifact(
    state: _ValidationState,
    manifest: PluginManifest,
    plugin_dir: Path,
) -> Path:
    if state.saved_artifact is not None and state.saved_artifact.is_file():
        return state.saved_artifact

    wrapper = _create_wrapper(state.wrapper_class)
    train_df = _build_train_df(manifest, state.convert_fn)
    wrapper.train(train_df, hyperparameters={"epochs": 1})
    with tempfile.TemporaryDirectory(prefix="pe_validate_") as tmp:
        dest = Path(tmp)
        fmt = wrapper.save_to_registry(dest)
        if not fmt:
            raise ValueError("save_to_registry returned empty format name")
        artifact = _find_weight_artifact(dest)
        persistent = plugin_dir / "_validation_artifact"
        persistent.parent.mkdir(parents=True, exist_ok=True)
        persistent.write_bytes(artifact.read_bytes())
        state.saved_artifact = persistent
        return persistent


def _run_check(
    checks: List[ValidationCheckResult],
    check_id: str,
    fn: Callable[[], Any],
    *,
    timeout: Optional[float] = None,
) -> bool:
    start = time.perf_counter()

    def invoke() -> Any:
        return fn()

    try:
        if timeout is not None:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(invoke)
                detail = future.result(timeout=timeout)
        else:
            detail = invoke()
        passed = True
        if detail is None:
            detail = "ok"
        elif not isinstance(detail, str):
            detail = str(detail)
    except FuturesTimeoutError:
        passed = False
        detail = f"timed out after {timeout}s"
    except Exception as exc:  # noqa: BLE001
        passed = False
        detail = str(exc)

    duration_ms = (time.perf_counter() - start) * 1000
    checks.append(
        ValidationCheckResult(
            id=check_id,
            passed=passed,
            detail=detail,
            duration_ms=round(duration_ms, 2),
        )
    )
    return passed


def validate_plugin_directory(plugin_dir: Path) -> ValidationReport:
    """Run validation checks for a plugin directory."""
    from datetime import datetime, timezone

    plugin_dir = plugin_dir.resolve()
    manifest = load_manifest(plugin_dir)
    checks: List[ValidationCheckResult] = []
    state = _ValidationState()
    smoke_timeout = _smoke_timeout_seconds()

    def check_manifest_schema():
        validate_plugin_name(manifest.name)
        if manifest.model is None:
            raise PluginError("manifest.model is required")
        root = plugins_root()
        for other_dir in list_plugin_dirs(root):
            if other_dir.name == manifest.name and other_dir.resolve() != plugin_dir:
                raise PluginError(
                    f"duplicate plugin name '{manifest.name}' at {other_dir}"
                )
        fixture = _standardized_fixture_path()
        fixture_note = (
            f"fixture={fixture.name}" if fixture.is_file() else "fixture=inline"
        )
        return f"version {manifest.version}, {fixture_note}"

    _run_check(checks, "manifest_schema", check_manifest_schema)

    if manifest.format is not None:
        def check_import_convert():
            module = load_module_from_file(
                f"pe_validate_{manifest.name}_convert",
                plugin_dir / manifest.format.module,
            )
            convert_fn = getattr(module, manifest.format.entrypoint, None)
            if convert_fn is None or not callable(convert_fn):
                raise PluginError(
                    f"Missing entrypoint '{manifest.format.entrypoint}' "
                    f"in {manifest.format.module}"
                )
            state.convert_fn = convert_fn
            return "convert module imported"

        _run_check(checks, "import_convert", check_import_convert)

        if state.convert_fn is not None:
            def check_conversion_roundtrip():
                std_df = _load_standardized_fixture()
                std_df = _ensure_std_columns(
                    std_df,
                    manifest.format.required_std_columns if manifest.format else (),
                )
                out = state.convert_fn(std_df)
                if len(out) != len(std_df):
                    raise ValueError(
                        f"row count changed: {len(std_df)} -> {len(out)}"
                    )
                if not out.index.equals(std_df.index):
                    raise ValueError("index not preserved after convert()")
                if manifest.format and manifest.format.output_columns:
                    missing_out = [
                        col
                        for col in manifest.format.output_columns
                        if col not in out.columns
                    ]
                    if missing_out:
                        raise ValueError(f"missing output columns: {missing_out}")
                    empty_cols = [
                        col
                        for col in manifest.format.output_columns
                        if out[col].isna().all()
                    ]
                    if empty_cols:
                        raise ValueError(f"output columns are all empty: {empty_cols}")
                return f"{len(out)} rows converted"

            _run_check(checks, "conversion_roundtrip", check_conversion_roundtrip)

    def check_import_wrapper():
        module = load_module_from_file(
            f"pe_validate_{manifest.name}_wrapper",
            plugin_dir / manifest.model.module,
        )
        wrapper_class = getattr(module, manifest.model.class_name, None)
        if wrapper_class is None or not isinstance(wrapper_class, type):
            raise PluginError(f"Missing wrapper class '{manifest.model.class_name}'")
        state.wrapper_class = wrapper_class
        return "wrapper module imported"

    _run_check(checks, "import_wrapper", check_import_wrapper)

    if state.wrapper_class is not None:
        def check_interface_compliance():
            wrapper_class = state.wrapper_class
            if not issubclass(wrapper_class, BasePEModel):
                raise TypeError("wrapper must subclass BasePEModel")
            for method_name in (
                "load_model",
                "prepare_data",
                "predict",
                "train",
                "evaluate",
                "save_model",
            ):
                if not hasattr(wrapper_class, method_name):
                    raise AttributeError(f"missing method {method_name}")
                method = getattr(wrapper_class, method_name)
                if getattr(method, "__isabstractmethod__", False):
                    raise AttributeError(f"{method_name} is still abstract")
            if "save_to_registry" not in wrapper_class.__dict__:
                raise AttributeError("save_to_registry not implemented on wrapper class")
            return wrapper_class.__name__

        _run_check(checks, "interface_compliance", check_interface_compliance)

        def check_train_smoke():
            wrapper = _create_wrapper(state.wrapper_class)
            train_df = _build_train_df(manifest, state.convert_fn)
            result = wrapper.train(train_df, hyperparameters={"epochs": 1})
            if not isinstance(result, dict):
                raise TypeError("train() must return a dict")
            return f"train returned {len(result)} keys"

        _run_check(
            checks,
            "train_smoke",
            check_train_smoke,
            timeout=smoke_timeout,
        )

        def check_save_to_registry():
            wrapper = _create_wrapper(state.wrapper_class)
            train_df = _build_train_df(manifest, state.convert_fn)
            wrapper.train(train_df, hyperparameters={"epochs": 1})
            with tempfile.TemporaryDirectory(prefix="pe_validate_") as tmp:
                dest = Path(tmp)
                fmt = wrapper.save_to_registry(dest)
                if not fmt:
                    raise ValueError("save_to_registry returned empty format name")
                artifact = _find_weight_artifact(dest)
                persistent = plugin_dir / "_validation_artifact"
                persistent.write_bytes(artifact.read_bytes())
                state.saved_artifact = persistent
                if not persistent.is_file():
                    raise ValueError("saved artifact is not loadable")
                reload_wrapper = _create_wrapper(state.wrapper_class)
                reload_wrapper.load_model(str(persistent))
                return f"format={fmt}"

        _run_check(
            checks,
            "save_to_registry",
            check_save_to_registry,
            timeout=smoke_timeout,
        )

        def check_eval_smoke():
            wrapper = _create_wrapper(state.wrapper_class)
            fallback = _ensure_saved_artifact(state, manifest, plugin_dir)
            _patch_load_weights(wrapper, plugin_dir, fallback)
            test_df = _build_eval_df(manifest, state.convert_fn)
            weight_id = "validation_smoke"
            weights_dir = plugin_dir / "weights"
            if weights_dir.is_dir():
                manifest_ids = [spec.id for spec in manifest.weights]
                if manifest_ids:
                    weight_id = manifest_ids[0]
                else:
                    first_dir = next(
                        (p.name for p in sorted(weights_dir.iterdir()) if p.is_dir()),
                        weight_id,
                    )
                    weight_id = first_dir
            metrics = wrapper.evaluate(test_df, weight_id)
            return _assert_eval_metrics(metrics)

        _run_check(
            checks,
            "eval_smoke",
            check_eval_smoke,
            timeout=smoke_timeout,
        )

        def check_predict_smoke():
            wrapper = _create_wrapper(state.wrapper_class)
            data_df = _build_eval_df(manifest, state.convert_fn)
            prepared = wrapper.prepare_data(data_df)
            preds = wrapper.predict(prepared)
            if len(preds) != len(data_df):
                raise ValueError(f"predict length {len(preds)} != {len(data_df)}")
            for idx, value in enumerate(preds):
                if not isinstance(value, (int, float)):
                    raise TypeError(f"prediction {idx} is not numeric")
            return f"{len(preds)} predictions"

        _run_check(
            checks,
            "predict_smoke",
            check_predict_smoke,
            timeout=smoke_timeout,
        )

    validated_at = (
        datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    passed = all(check.passed for check in checks)
    validation_artifact = plugin_dir / "_validation_artifact"
    if validation_artifact.is_file():
        validation_artifact.unlink()
    return ValidationReport(
        plugin_name=manifest.name,
        passed=passed,
        checks=checks,
        validated_at=validated_at,
    )
