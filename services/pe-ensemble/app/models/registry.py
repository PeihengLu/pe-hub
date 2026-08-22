"""Central registry for PE Ensemble model wrappers and metadata."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Type

from pe_common.model_interface import BasePEModel

WeightEntry = Dict[str, Any]
ListWeightEntriesFn = Callable[[], List[WeightEntry]]
ValidateWeightFn = Callable[[str], None]
ArchitectureBuilderFn = Callable[[Mapping[str, Any]], Dict[str, Any]]
ArchitectureCliFn = Callable[[Any], Dict[str, Any]]


def _repeat_int(value: int, count: int = 3) -> list[int]:
    return [int(value)] * count


def _present(values: Mapping[str, Any], key: str) -> bool:
    if key not in values:
        return False
    value = values[key]
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True


def _build_deepprime_architecture(values: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if _present(values, "hidden_size"):
        out["hidden_size"] = int(values["hidden_size"])
    if _present(values, "num_layers"):
        out["num_layers"] = int(values["num_layers"])
    return out


def _build_oped_architecture(values: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if _present(values, "embedding_size"):
        out["embedding_size"] = int(values["embedding_size"])
    if _present(values, "ffn_dim"):
        out["hidden_size"] = _repeat_int(int(values["ffn_dim"]))
    if _present(values, "encoder_layers"):
        out["num_encoder_layers"] = _repeat_int(int(values["encoder_layers"]))
    if _present(values, "nhead"):
        out["nhead"] = int(values["nhead"])
    if _present(values, "dropout"):
        out["drop_out"] = float(values["dropout"])
    return out


def _build_pridict2_architecture(values: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if _present(values, "embed_dim"):
        out["embed_dim"] = int(values["embed_dim"])
    if _present(values, "z_dim"):
        out["z_dim"] = int(values["z_dim"])
    if _present(values, "num_hidden_layers"):
        out["num_hidden_layers"] = int(values["num_hidden_layers"])
    if _present(values, "dropout"):
        out["p_dropout"] = float(values["dropout"])
    return out


def _deepprime_architecture_from_cli(args: Any) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    if getattr(args, "dp_hidden_size", None) is not None:
        values["hidden_size"] = args.dp_hidden_size
    if getattr(args, "dp_num_layers", None) is not None:
        values["num_layers"] = args.dp_num_layers
    return _build_deepprime_architecture(values)


def _oped_architecture_from_cli(args: Any) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    if getattr(args, "oped_embedding_size", None) is not None:
        values["embedding_size"] = args.oped_embedding_size
    if getattr(args, "oped_ffn_dim", None) is not None:
        values["ffn_dim"] = args.oped_ffn_dim
    if getattr(args, "oped_encoder_layers", None) is not None:
        values["encoder_layers"] = args.oped_encoder_layers
    if getattr(args, "oped_nhead", None) is not None:
        values["nhead"] = args.oped_nhead
    if getattr(args, "oped_dropout", None) is not None:
        values["dropout"] = args.oped_dropout
    return _build_oped_architecture(values)


def _pridict2_architecture_from_cli(args: Any) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    if getattr(args, "pridict2_embed_dim", None) is not None:
        values["embed_dim"] = args.pridict2_embed_dim
    if getattr(args, "pridict2_z_dim", None) is not None:
        values["z_dim"] = args.pridict2_z_dim
    if getattr(args, "pridict2_num_hidden_layers", None) is not None:
        values["num_hidden_layers"] = args.pridict2_num_hidden_layers
    if getattr(args, "pridict2_dropout", None) is not None:
        values["dropout"] = args.pridict2_dropout
    return _build_pridict2_architecture(values)


@dataclass(frozen=True)
class ModelSpec:
    """Metadata and hooks for one registered model."""

    name: str
    wrapper_class: Type[BasePEModel]
    display_name: str
    description: str
    model_type: str
    pe_db_format: str
    weight_format: str
    architecture_builder: ArchitectureBuilderFn = field(
        default=lambda _values: {}
    )
    architecture_from_cli: ArchitectureCliFn = field(
        default=lambda _args: {}
    )
    list_weight_entries: Optional[ListWeightEntriesFn] = None
    validate_weight: Optional[ValidateWeightFn] = None
    source: str = "builtin"


class ModelRegistry:
    """Maps model names to wrapper classes and shared metadata."""

    def __init__(self) -> None:
        self._specs: Dict[str, ModelSpec] = {}

    def register(self, spec: ModelSpec) -> None:
        key = spec.name.strip().lower()
        if not key:
            raise ValueError("Model name must not be empty.")
        existing = self._specs.get(key)
        if existing is not None and existing.source == "builtin":
            raise ValueError(f"Cannot replace built-in model '{key}'")
        self._specs[key] = spec

    def is_registered(self, name: str) -> bool:
        return name.strip().lower() in self._specs

    def get(self, name: str) -> ModelSpec:
        key = name.strip().lower()
        if key not in self._specs:
            supported = sorted(self._specs.keys())
            raise ValueError(
                f"Unknown model: {name}. Available models: {supported}"
            )
        return self._specs[key]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs.keys()))

    def model_format_map(self) -> Dict[str, str]:
        return {name: self._specs[name].pe_db_format for name in self.names()}

    def list_catalog_entries(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "type": spec.model_type,
                "status": "available",
                "display_name": spec.display_name,
                "source": spec.source,
            }
            for spec in self._specs.values()
        ]

    def unregister(self, name: str) -> None:
        key = name.strip().lower()
        existing = self._specs.get(key)
        if existing is None:
            return
        if existing.source == "builtin":
            raise ValueError(f"Cannot unregister built-in model '{key}'")
        del self._specs[key]

    def build_architecture(
        self,
        model_name: str,
        values: Mapping[str, Any],
    ) -> Dict[str, Any]:
        return self.get(model_name).architecture_builder(values)

    def architecture_from_cli(self, model_name: str, args: Any) -> Dict[str, Any]:
        return self.get(model_name).architecture_from_cli(args)

    def list_weight_entries(self, model_name: str) -> List[WeightEntry]:
        spec = self.get(model_name)
        if spec.list_weight_entries is not None:
            return spec.list_weight_entries()
        from . import weights_registry

        return weights_registry.list_entries(model_name)

    def validate_weight_selection(self, model_name: str, weight_id: str) -> None:
        spec = self.get(model_name)
        if spec.validate_weight is not None:
            spec.validate_weight(weight_id)
            return
        from . import weights_registry

        weights_registry.resolve_dir(model_name, weight_id)


model_registry = ModelRegistry()


def _register_builtin_models() -> None:
    from .deepprime_wrapper import DeepPrimeModelWrapper
    from .oped_wrapper import OPEDModelWrapper
    from .optiprime_wrapper import OptiPrimeModelWrapper
    from .pridict2_wrapper import PRIDICT2ModelWrapper

    def _validate_pridict2_weight(weight_id: str) -> None:
        PRIDICT2ModelWrapper.resolve_weight_selection(weight_id)

    model_registry.register(
        ModelSpec(
            name="deepprime",
            wrapper_class=DeepPrimeModelWrapper,
            display_name="DeepPrime",
            description="CNN model for PE efficiency prediction",
            model_type="neural_network",
            pe_db_format="deepprime",
            weight_format="deepprime_ensemble",
            architecture_builder=_build_deepprime_architecture,
            architecture_from_cli=_deepprime_architecture_from_cli,
        )
    )
    model_registry.register(
        ModelSpec(
            name="oped",
            wrapper_class=OPEDModelWrapper,
            display_name="OPED",
            description=(
                "Optimized Prime Editor prediction model using transformer architecture"
            ),
            model_type="neural_network",
            pe_db_format="oped",
            weight_format="oped_state_dict",
            architecture_builder=_build_oped_architecture,
            architecture_from_cli=_oped_architecture_from_cli,
        )
    )
    model_registry.register(
        ModelSpec(
            name="optiprime",
            wrapper_class=OptiPrimeModelWrapper,
            display_name="OptiPrime",
            description=(
                "Mechanistic ML model (ODE kinetics + HetFormer) for PE efficiency prediction"
            ),
            model_type="mechanistic_ml",
            pe_db_format="optiprime",
            weight_format="optiprime_ensemble",
        )
    )
    model_registry.register(
        ModelSpec(
            name="pridict2",
            wrapper_class=PRIDICT2ModelWrapper,
            display_name="PRIDICT2",
            description="Improved version of PRIDICT with transfer learning",
            model_type="pssm",
            pe_db_format="pridict2",
            weight_format="pridict2_run",
            architecture_builder=_build_pridict2_architecture,
            architecture_from_cli=_pridict2_architecture_from_cli,
            list_weight_entries=PRIDICT2ModelWrapper.list_available_weight_entries,
            validate_weight=_validate_pridict2_weight,
        )
    )


_register_builtin_models()
