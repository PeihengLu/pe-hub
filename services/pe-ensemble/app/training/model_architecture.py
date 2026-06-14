"""Model architecture (size) hyperparameters for training UI and CLI."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


def _repeat_int(value: int, count: int = 3) -> list[int]:
    return [int(value)] * count


def build_architecture_hyperparameters(
    model_name: str,
    values: Mapping[str, Any],
) -> Dict[str, Any]:
    """Map UI/CLI architecture fields to wrapper hyperparameter keys."""
    model = model_name.strip().lower()
    out: Dict[str, Any] = {}

    if model == "deepprime":
        if _present(values, "hidden_size"):
            out["hidden_size"] = int(values["hidden_size"])
        if _present(values, "num_layers"):
            out["num_layers"] = int(values["num_layers"])
        return out

    if model == "oped":
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

    if model == "pridict2":
        if _present(values, "embed_dim"):
            out["embed_dim"] = int(values["embed_dim"])
        if _present(values, "z_dim"):
            out["z_dim"] = int(values["z_dim"])
        if _present(values, "num_hidden_layers"):
            out["num_hidden_layers"] = int(values["num_hidden_layers"])
        if _present(values, "dropout"):
            out["p_dropout"] = float(values["dropout"])
        return out

    return out


def merge_training_hyperparameters(
    model_name: str,
    base: Optional[Dict[str, Any]],
    architecture: Mapping[str, Any],
) -> Dict[str, Any]:
    """Merge architecture fields into an existing hyperparameters dict."""
    merged = dict(base or {})
    if not merged.get("load_pretrained"):
        merged.update(build_architecture_hyperparameters(model_name, architecture))
    return apply_fine_tune_defaults(merged)


def apply_fine_tune_defaults(hyperparameters: Mapping[str, Any]) -> Dict[str, Any]:
    """When fine-tuning from pretrained weights, freeze the representation backbone."""
    merged = dict(hyperparameters)
    if merged.get("load_pretrained"):
        merged["freezing"] = True
    return merged


def architecture_from_cli_args(model_name: str, args: Any) -> Dict[str, Any]:
    """Read optional architecture CLI flags for the selected model."""
    model = model_name.strip().lower()
    values: Dict[str, Any] = {}

    if model == "deepprime":
        if getattr(args, "dp_hidden_size", None) is not None:
            values["hidden_size"] = args.dp_hidden_size
        if getattr(args, "dp_num_layers", None) is not None:
            values["num_layers"] = args.dp_num_layers
    elif model == "oped":
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
    elif model == "pridict2":
        if getattr(args, "pridict2_embed_dim", None) is not None:
            values["embed_dim"] = args.pridict2_embed_dim
        if getattr(args, "pridict2_z_dim", None) is not None:
            values["z_dim"] = args.pridict2_z_dim
        if getattr(args, "pridict2_num_hidden_layers", None) is not None:
            values["num_hidden_layers"] = args.pridict2_num_hidden_layers
        if getattr(args, "pridict2_dropout", None) is not None:
            values["dropout"] = args.pridict2_dropout

    return build_architecture_hyperparameters(model, values)


def _present(values: Mapping[str, Any], key: str) -> bool:
    if key not in values:
        return False
    value = values[key]
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True
