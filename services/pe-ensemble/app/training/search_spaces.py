"""Optuna search spaces for dataset hyperparameter tuning.

Scheduler settings are intentionally excluded; users set those at train time.
"""
from __future__ import annotations

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
    return suggested


_POST_PROCESSORS: Dict[str, Callable[[Any, Dict[str, Any]], Dict[str, Any]]] = {
    "oped": _oped_architecture_suggestions,
}


SEARCH_SPACES: Dict[str, SearchSpaceSpec] = {
    "deepprime": SearchSpaceSpec(
        metric="validation.neg_val_loss",
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
        metric="validation.spearman",
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
        metric="validation.averageedited_spearman",
        direction="maximize",
        fixed={"load_pretrained": False, "loss_func": "KLDloss"},
        params={
            "lr": FloatParam(1e-5, 1e-3, log=True),
            "weight_decay": FloatParam(1e-6, 1e-2, log=True),
            "batch_size": CategoricalParam((64, 128, 256)),
            "num_epochs": IntParam(15, 40),
            "embed_dim": CategoricalParam((32, 64, 128)),
            "z_dim": CategoricalParam((32, 64, 128)),
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


def suggest_trial_hyperparameters(model_name: str, trial: Any) -> Dict[str, Any]:
    space = get_search_space(model_name)
    suggested = space.suggest(trial)
    processor = _POST_PROCESSORS.get(model_name.strip().lower())
    if processor is not None:
        suggested = processor(trial, suggested)
    return {key: value for key, value in suggested.items() if key not in SCHEDULER_KEYS}
