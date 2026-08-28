"""Optuna search spaces for dataset hyperparameter tuning.

Scheduler settings are intentionally excluded; users set those at train time.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional

from .hyperparameter_presets import SCHEDULER_KEYS


@dataclass(frozen=True)
class FloatParam:
    low: float
    high: float
    log: bool = False


@dataclass(frozen=True)
class IntParam:
    low: int
    high: int


@dataclass(frozen=True)
class CategoricalParam:
    choices: tuple[Any, ...]


SearchParam = FloatParam | IntParam | CategoricalParam


@dataclass(frozen=True)
class SearchSpaceSpec:
    metric: str
    direction: str  # maximize | minimize
    params: Dict[str, SearchParam] = field(default_factory=dict)
    fixed: Dict[str, Any] = field(default_factory=dict)

    def suggest(self, trial: Any) -> Dict[str, Any]:
        suggested: Dict[str, Any] = dict(self.fixed)
        for name, spec in self.params.items():
            if name in SCHEDULER_KEYS:
                continue
            if isinstance(spec, FloatParam):
                suggested[name] = trial.suggest_float(
                    name, spec.low, spec.high, log=spec.log
                )
            elif isinstance(spec, IntParam):
                suggested[name] = trial.suggest_int(name, spec.low, spec.high)
            elif isinstance(spec, CategoricalParam):
                suggested[name] = trial.suggest_categorical(name, list(spec.choices))
        return suggested


def _oped_architecture_suggestions(trial: Any, suggested: Dict[str, Any]) -> Dict[str, Any]:
    del trial  # Optuna trial unused; kept for a uniform post-processor signature.
    ffn_dim = suggested.get("ffn_dim")
    encoder_layers = suggested.get("encoder_layers")
    if ffn_dim is not None:
        suggested["hidden_size"] = [int(ffn_dim), int(ffn_dim), int(ffn_dim)]
        suggested.pop("ffn_dim", None)
    if encoder_layers is not None:
        layers = int(encoder_layers)
        suggested["num_encoder_layers"] = [layers, layers, layers]
        suggested.pop("encoder_layers", None)
    if "dropout" in suggested:
        suggested["drop_out"] = float(suggested.pop("dropout"))
    embedding_size = suggested.get("embedding_size")
    nhead = suggested.get("nhead")
    if embedding_size is not None and nhead is not None:
        embed = int(embedding_size)
        heads = int(nhead)
        if heads <= 0 or embed % heads != 0:
            raise ValueError(
                f"OPED requires embedding_size % nhead == 0 "
                f"(got embedding_size={embed}, nhead={heads})"
            )
    return suggested


_POST_PROCESSORS: Dict[str, Callable[[Any, Dict[str, Any]], Dict[str, Any]]] = {
    "oped": _oped_architecture_suggestions,
}


SEARCH_SPACES: Dict[str, SearchSpaceSpec] = {
    "deepprime": SearchSpaceSpec(
        metric="cv.neg_mean_best_val_loss",
        direction="maximize",
        fixed={"load_pretrained": False},
        params={
            "lr": FloatParam(1e-5, 1e-3, log=True),
            "weight_decay": FloatParam(1e-6, 1e-2, log=True),
            "batch_size": CategoricalParam((64, 128, 256)),
            "epochs": IntParam(3, 20),
            "hidden_size": CategoricalParam((64, 128, 256)),
            "num_layers": CategoricalParam((1, 2, 3)),
        },
    ),
    "oped": SearchSpaceSpec(
        metric="cv.mean_val_spearman",
        direction="maximize",
        fixed={"load_pretrained": False},
        params={
            "lr": FloatParam(1e-5, 1e-3, log=True),
            "weight_decay": FloatParam(1e-6, 1e-2, log=True),
            "batch_size": CategoricalParam((64, 128, 256)),
            "epoch_num": IntParam(20, 100),
            "embedding_size": CategoricalParam((32, 64, 128)),
            "ffn_dim": CategoricalParam((1024, 2048)),
            "encoder_layers": CategoricalParam((4, 6)),
            "nhead": CategoricalParam((4, 8)),
            "dropout": FloatParam(0.05, 0.3),
        },
    ),
    "pridict2": SearchSpaceSpec(
        metric="cv.mean_averageedited_spearman",
        direction="maximize",
        # assemb_opt/annot_embed/z_dim are hardcoded or derived in the wrapper.
        fixed={
            "load_pretrained": False,
            "loss_func": "MSEloss",
            "y_ref": ["averageedited"],
        },
        params={
            "lr": FloatParam(1e-5, 1e-3, log=True),
            "weight_decay": FloatParam(1e-6, 1e-2, log=True),
            "batch_size": CategoricalParam((128, 256, 512, 1024)),
            "num_epochs": IntParam(10, 30),
            "embed_dim": CategoricalParam((32, 64, 128)),
            "num_hidden_layers": CategoricalParam((1, 2, 3)),
            "p_dropout": FloatParam(0.05, 0.45),
        },
    ),
}


def get_search_space(model_name: str) -> SearchSpaceSpec:
    key = model_name.strip().lower()
    if key not in SEARCH_SPACES:
        supported = ", ".join(sorted(SEARCH_SPACES))
        raise ValueError(f"No search space for model '{model_name}'. Supported: {supported}")
    return SEARCH_SPACES[key]


def _search_space_canonical(space: SearchSpaceSpec) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for name, spec in sorted(space.params.items()):
        if isinstance(spec, FloatParam):
            params[name] = {
                "type": "float",
                "low": float(spec.low),
                "high": float(spec.high),
                "log": bool(spec.log),
            }
        elif isinstance(spec, IntParam):
            params[name] = {
                "type": "int",
                "low": int(spec.low),
                "high": int(spec.high),
            }
        elif isinstance(spec, CategoricalParam):
            params[name] = {
                "type": "categorical",
                "choices": list(spec.choices),
            }
    return {
        "metric": space.metric,
        "direction": space.direction,
        "fixed": dict(space.fixed),
        "params": params,
    }


def search_space_fingerprint(model_name: str) -> str:
    """Stable short id for the current Optuna search space (8 hex chars)."""
    space = get_search_space(model_name)
    payload = json.dumps(_search_space_canonical(space), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def search_space_study_suffix(model_name: str) -> str:
    return f"__sp_{search_space_fingerprint(model_name)}"


def resolve_study_name(base_name: str, model_name: str) -> str:
    """Append a search-space suffix so Optuna studies are not resumed across space changes."""
    suffix = search_space_study_suffix(model_name)
    cleaned = base_name.strip()
    if cleaned.endswith(suffix):
        return cleaned
    marker = "__sp_"
    if marker in cleaned:
        head, _, tail = cleaned.rpartition(marker)
        if len(tail) == 8 and all(ch in "0123456789abcdef" for ch in tail.lower()):
            cleaned = head
    return f"{cleaned}{suffix}"


def materialize_hyperparameters(
    model_name: str,
    raw_params: Mapping[str, Any],
) -> Dict[str, Any]:
    """Merge search-space fixed keys and apply model post-processors.

    Optuna ``best.params`` only contains suggested keys. Call this before writing
    presets or launching a final train so aliases (e.g. OPED ``ffn_dim``) and
    fixed flags (e.g. ``load_pretrained=False``) are applied.
    """
    space = get_search_space(model_name)
    suggested: Dict[str, Any] = dict(space.fixed)
    suggested.update(dict(raw_params))
    processor = _POST_PROCESSORS.get(model_name.strip().lower())
    if processor is not None:
        suggested = processor(None, suggested)
    return {key: value for key, value in suggested.items() if key not in SCHEDULER_KEYS}


def suggest_trial_hyperparameters(model_name: str, trial: Any) -> Dict[str, Any]:
    space = get_search_space(model_name)
    suggested = space.suggest(trial)
    processor = _POST_PROCESSORS.get(model_name.strip().lower())
    if processor is not None:
        suggested = processor(trial, suggested)
    return {key: value for key, value in suggested.items() if key not in SCHEDULER_KEYS}
