"""Shared PE-DB filter and split field names.

CLI flags, FastAPI query params, JSON request bodies, and
``filter_from_params`` all speak this contract. Wire names must stay stable:
PE Hub and ensemble ``GET /data/filter`` send these keys to PE-DB
``GET /api/filter``. Rename a field by changing the tuples here (and the
matching TypeScript consts); do not hand-list the same names in FastAPI or
the web client.
"""
from __future__ import annotations

import argparse
import inspect
from typing import Any, Literal, Mapping, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic_core import PydanticUndefined

FILTER_LIST_FIELDS: tuple[str, ...] = (
    "study",
    "dataset",
    "cell_line",
    "pe_system",
    "edit_type",
    "edit_length",
    "edit_scope",
    "experimental_method",
    "target_context",
    "scaffold_name",
)

FILTER_RANGE_FIELDS: tuple[str, ...] = (
    "edit_efficiency_min",
    "edit_efficiency_max",
)

FILTER_LIST_DESCRIPTIONS: dict[str, str] = {
    "study": "Filter by study key (e.g. deepprime).",
    "dataset": "Filter by dataset name within the study.",
    "cell_line": "Filter by cell line (e.g. HEK293T).",
    "pe_system": "Filter by PE system (e.g. PE2max).",
    "edit_type": "Filter edits by type (sub, ins, del).",
    "edit_length": "Filter edits by length.",
    "edit_scope": "Filter by edit scope (on_target, off_target).",
    "experimental_method": "Filter by experimental method.",
    "target_context": "Filter by target context.",
    "scaffold_name": "Filter by pegRNA scaffold name.",
}

FILTER_RANGE_DESCRIPTIONS: dict[str, str] = {
    "edit_efficiency_min": "Minimum editing efficiency.",
    "edit_efficiency_max": "Maximum editing efficiency.",
}

SPLIT_STRATEGY_CHOICES: tuple[str, ...] = ("none", "holdout_2", "holdout_3", "cv")
SplitStrategyName = Literal["none", "holdout_2", "holdout_3", "cv"]

SPLIT_QUERY_FIELDS: tuple[str, ...] = (
    "split_strategy",
    "train_pct",
    "val_pct",
    "test_pct",
    "cv_folds",
    "use_original_fold",
    "original_fold_test_value",
    "split_random_state",
    "merge",
)

SPLIT_QUERY_DESCRIPTIONS: dict[str, str] = {
    "split_strategy": (
        "Required when format is set. Split assignment strategy: "
        "none, holdout_2, holdout_3, or cv."
    ),
    "train_pct": "Train fraction for holdout splits.",
    "val_pct": "Validation fraction for three-way holdout.",
    "test_pct": "Test fraction for holdout splits or an optional CV outer holdout.",
    "cv_folds": "Number of CV folds when split_strategy is cv.",
    "use_original_fold": "When true, use author original_fold assignments where available.",
    "original_fold_test_value": (
        "original_fold value treated as the test partition when use_original_fold is true "
        "(-1 for DeepPrime-style held-out test; 0–4 for PRIDICT2 CV test folds)."
    ),
    "split_random_state": "RNG seed for random split assignment.",
    "merge": (
        "When true, merge all matching datasheets before split assignment. "
        "Reassigns group_id by shared protospacer after merge."
    ),
}

SPLIT_BOOL_FIELDS = frozenset({"use_original_fold", "merge"})
SPLIT_INT_FIELDS = frozenset({"cv_folds", "split_random_state"})
SPLIT_FLOAT_FIELDS = frozenset(
    {"train_pct", "val_pct", "test_pct", "original_fold_test_value"}
)
SPLIT_DEFAULTS: dict[str, Any] = {
    "use_original_fold": False,
    "original_fold_test_value": -1.0,
    "split_random_state": 42,
    "merge": False,
}
SPLIT_OMIT_IF_NONE = frozenset({"train_pct", "val_pct", "test_pct", "cv_folds"})

FilterScalar = Union[str, int]
FilterValue = Union[FilterScalar, list[Any]]

_FILTER_FIELD_NAMES = frozenset(FILTER_LIST_FIELDS) | frozenset(FILTER_RANGE_FIELDS)


class _IgnoreExtraModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


def coerce_list_param(value: Any) -> Optional[list[Any]]:
    """Normalize a scalar-or-list filter value; empty lists become ``None``."""
    if value is None:
        return None
    if isinstance(value, list):
        return value or None
    return [value]


def add_filter_arguments(parser: argparse.ArgumentParser) -> None:
    """Append the catalog/edit filter flags used by ``pedb`` and ``peen``."""
    for name in FILTER_LIST_FIELDS:
        kwargs: dict[str, Any] = {"action": "append", "default": [], "dest": name}
        if name == "edit_length":
            kwargs["type"] = int
        parser.add_argument(f"--{name.replace('_', '-')}", **kwargs)
    for name in FILTER_RANGE_FIELDS:
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=None)


def add_split_arguments(
    parser: argparse.ArgumentParser,
    *,
    split_strategy_default: Optional[str] = None,
    use_original_fold: str = "store_true",
) -> None:
    """Append split-assignment flags used by ``pedb filter`` and ``peen``."""
    parser.add_argument(
        "--split-strategy",
        default=split_strategy_default,
        choices=list(SPLIT_STRATEGY_CHOICES),
    )
    parser.add_argument("--train-pct", type=float, default=None)
    parser.add_argument("--val-pct", type=float, default=None)
    parser.add_argument("--test-pct", type=float, default=None)
    parser.add_argument("--cv-folds", type=int, default=None)
    if use_original_fold == "boolean_optional":
        parser.add_argument(
            "--use-original-fold",
            action=argparse.BooleanOptionalAction,
            default=False,
            help=(
                "Use author original_fold where available "
                "(--no-use-original-fold to force random)"
            ),
        )
    else:
        parser.add_argument("--use-original-fold", action="store_true")
    parser.add_argument(
        "--original-fold-test-value",
        type=float,
        default=SPLIT_DEFAULTS["original_fold_test_value"],
    )
    parser.add_argument(
        "--split-random-state",
        type=int,
        default=SPLIT_DEFAULTS["split_random_state"],
    )
    parser.add_argument("--merge", action="store_true")


def filter_kwargs_from_mapping(source: Mapping[str, Any]) -> dict[str, Any]:
    """Extract filter fields from a mapping (argparse namespace via ``vars``)."""
    out: dict[str, Any] = {
        name: coerce_list_param(source.get(name)) for name in FILTER_LIST_FIELDS
    }
    for name in FILTER_RANGE_FIELDS:
        out[name] = source.get(name)
    return out


def filter_kwargs_from_namespace(args: argparse.Namespace) -> dict[str, Any]:
    return filter_kwargs_from_mapping(vars(args))


def _coerce_split_value(name: str, value: Any) -> Any:
    if value is None:
        if name in SPLIT_DEFAULTS:
            value = SPLIT_DEFAULTS[name]
        else:
            return None
    if name in SPLIT_BOOL_FIELDS:
        return bool(value)
    if name in SPLIT_INT_FIELDS:
        return int(value)
    if name in SPLIT_FLOAT_FIELDS:
        return float(value)
    return value


def split_kwargs_from_mapping(source: Mapping[str, Any]) -> dict[str, Any]:
    """Named split kwargs for ``filter_data`` / ``SplitQueryParams`` (with defaults)."""
    return {
        name: _coerce_split_value(name, source.get(name)) for name in SPLIT_QUERY_FIELDS
    }


def split_query_params_from_mapping(source: Mapping[str, Any]) -> dict[str, Any]:
    """PE-DB GET query dict: omit optional split fields that are unset."""
    out = split_kwargs_from_mapping(source)
    return {name: value for name, value in out.items() if not (name in SPLIT_OMIT_IF_NONE and value is None)}


def known_filter_kwargs(filters: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only catalog/edit filter keys; raise on unknown names."""
    unknown = set(filters) - _FILTER_FIELD_NAMES
    if unknown:
        raise TypeError(f"unexpected filter fields: {sorted(unknown)}")
    return dict(filters)


def _list_annotation(name: str) -> Any:
    return Optional[list[int]] if name == "edit_length" else Optional[list[str]]


def _catalog_filter_query_fields() -> dict[str, tuple[Any, Any]]:
    fields: dict[str, tuple[Any, Any]] = {
        name: (
            _list_annotation(name),
            Field(default=None, description=FILTER_LIST_DESCRIPTIONS[name]),
        )
        for name in FILTER_LIST_FIELDS
    }
    fields.update(
        {
            name: (
                Optional[float],
                Field(default=None, description=FILTER_RANGE_DESCRIPTIONS[name]),
            )
            for name in FILTER_RANGE_FIELDS
        }
    )
    return fields


def _split_export_query_fields() -> dict[str, tuple[Any, Any]]:
    annotations: dict[str, Any] = {
        "split_strategy": Optional[SplitStrategyName],
        "train_pct": Optional[float],
        "val_pct": Optional[float],
        "test_pct": Optional[float],
        "cv_folds": Optional[int],
        "use_original_fold": bool,
        "original_fold_test_value": float,
        "split_random_state": int,
        "merge": bool,
    }
    defaults: dict[str, Any] = {
        "split_strategy": None,
        "train_pct": None,
        "val_pct": None,
        "test_pct": None,
        "cv_folds": None,
        **SPLIT_DEFAULTS,
    }
    return {
        name: (
            annotations[name],
            Field(default=defaults[name], description=SPLIT_QUERY_DESCRIPTIONS[name]),
        )
        for name in SPLIT_QUERY_FIELDS
    }


def _catalog_filter_body_fields() -> dict[str, tuple[Any, Any]]:
    fields: dict[str, tuple[Any, Any]] = {
        name: (Optional[FilterValue], Field(default=None)) for name in FILTER_LIST_FIELDS
    }
    fields.update(
        {name: (Optional[float], Field(default=None)) for name in FILTER_RANGE_FIELDS}
    )
    return fields


CatalogFilterQuery = create_model(
    "CatalogFilterQuery",
    __base__=_IgnoreExtraModel,
    **_catalog_filter_query_fields(),
)

SplitExportQuery = create_model(
    "SplitExportQuery",
    __base__=_IgnoreExtraModel,
    **_split_export_query_fields(),
)

CatalogFilterBody = create_model(
    "CatalogFilterBody",
    **_catalog_filter_body_fields(),
)


def query_dependency(model: type[BaseModel]):
    """FastAPI ``Depends`` callable that flattens a model into query parameters.

    ``Annotated[Model, Query()]`` is not flattened on the FastAPI version this
    repo pins, so both HTTP apps use this signature instead of listing
    ``Query(...)`` by hand.
    """
    from fastapi import Query

    parameters: list[inspect.Parameter] = []
    for name, field in model.model_fields.items():
        default = field.default
        if default is PydanticUndefined:
            query_default = Query(..., description=field.description)
        else:
            query_default = Query(default, description=field.description)
        parameters.append(
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                default=query_default,
                annotation=field.annotation,
            )
        )

    def _parse(**kwargs: Any) -> BaseModel:
        return model(**kwargs)

    _parse.__name__ = f"{model.__name__}_query"
    _parse.__qualname__ = _parse.__name__
    _parse.__signature__ = inspect.Signature(parameters, return_annotation=model)
    _parse.__annotations__ = {parameter.name: parameter.annotation for parameter in parameters}
    _parse.__annotations__["return"] = model
    return _parse
