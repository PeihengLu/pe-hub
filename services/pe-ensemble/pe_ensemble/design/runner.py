"""Interactive pegRNA design: enumerate → convert → score → rank."""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from pe_common.design_candidates import enumerate_design_candidates
from pe_common.design_rules import apply_design_ruleset_mask
from pe_common.devices import AUTO_DEVICE, resolve_device, resolve_device_id
from pe_common.model_interface import BasePEModel

from ..ensemble.combine import combine_predictions
from ..ensemble.schemas import EnsembleMember
from ..models.model_factory import ModelFactory
from ..models.registry import model_registry
from ..training.config import is_supported_model, pe_db_url, use_pe_db_library
from .schemas import DesignRequest

logger = logging.getLogger(__name__)

ProgressLog = Callable[[str], None]

# Extra columns kept for the design UI (not part of PE-core conversion).
_DESIGN_META_COLUMNS = (
    "pbs_len",
    "rtt_len",
    "homology_len",
    "spacer",
    "pam",
    "nick",
)


class DesignError(RuntimeError):
    """Raised when design enumeration or scoring fails."""


def _null_log(message: str) -> None:
    logger.info(message)


def _pe_core_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Drop design-only metadata before PE-DB conversion."""
    drop = [col for col in _DESIGN_META_COLUMNS if col in df.columns]
    return df.drop(columns=drop) if drop else df


def _json_safe_value(value: Any) -> Any:
    """Make a scalar JSON-encodable (NaN/Inf → None)."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, np.floating):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _records_for_json(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """DataFrame rows as JSON-safe dicts (stdlib json rejects bare NaN/Inf)."""
    sanitized = df.replace([np.inf, -np.inf], np.nan)
    return [
        {column: _json_safe_value(value) for column, value in record.items()}
        for record in sanitized.to_dict(orient="records")
    ]


def _convert_standardized(
    std_df: pd.DataFrame,
    model_format: str,
    *,
    progress_log: ProgressLog,
) -> pd.DataFrame:
    records = _records_for_json(_pe_core_frame(std_df))
    progress_log(f"Converting {len(records)} candidates to format={model_format}")
    if use_pe_db_library():
        try:
            from pe_db.library import convert_standardized_records
        except ImportError as exc:
            raise DesignError(
                "In-process PE-DB convert requires pe-db. "
                "Install with: pip install -e services/pe-db"
            ) from exc
        payload = convert_standardized_records(records, format_=model_format)
    else:
        try:
            response = requests.post(
                f"{pe_db_url().rstrip('/')}/api/convert",
                json={"records": records, "format": model_format},
                timeout=(10, 600),
            )
        except (requests.RequestException, ValueError, TypeError) as exc:
            raise DesignError(f"PE-DB convert request failed: {exc}") from exc
        if response.status_code >= 400:
            raise DesignError(
                f"PE-DB convert failed ({response.status_code}): {response.text}"
            )
        payload = response.json()
    converted = pd.DataFrame(payload.get("records") or [])
    if len(converted) != len(std_df):
        raise DesignError(
            f"Format conversion changed row count: {len(std_df)} → {len(converted)}"
        )
    converted.index = std_df.index
    return converted


def _predict_member(
    member: EnsembleMember,
    native_df: pd.DataFrame,
    *,
    device,
    progress_log: ProgressLog,
) -> List[float]:
    model_name = member.model_name.strip().lower()
    if not is_supported_model(model_name):
        raise DesignError(f"Invalid model name: {member.model_name}")
    model = ModelFactory.create_model(model_name, device=device)
    model.load_weights_by_name(member.weights)
    progress_log(f"Scoring with {model_name}:{member.weights} ({len(native_df)} rows)")
    predict = getattr(model, "predict_on_frame", None)
    if predict is None:
        return BasePEModel.predict_on_frame(model, native_df)
    return predict(native_df)


def _design_result_rows(
    std_df: pd.DataFrame,
    scores: np.ndarray,
    *,
    member_scores: Optional[Dict[str, List[float]]] = None,
) -> List[Dict[str, Any]]:
    order = np.argsort(-scores, kind="mergesort")
    rows: List[Dict[str, Any]] = []
    for rank, index in enumerate(order, start=1):
        row = std_df.iloc[int(index)]
        item: Dict[str, Any] = {
            "rank": rank,
            "score": float(scores[int(index)]),
            "pbs_len": int(row["pbs_len"]) if "pbs_len" in std_df.columns else None,
            "rtt_len": int(row["rtt_len"]) if "rtt_len" in std_df.columns else None,
            "homology_len": (
                int(row["homology_len"]) if "homology_len" in std_df.columns else None
            ),
            "spacer": str(row["spacer"]) if "spacer" in std_df.columns else None,
            "pam": str(row["pam"]) if "pam" in std_df.columns else None,
            "nick": int(row["nick"]) if "nick" in std_df.columns else None,
            "protospacer_location_l": int(row["protospacer_location_l"]),
            "protospacer_location_r": int(row["protospacer_location_r"]),
            "pbs_location_l": int(row["pbs_location_l"]),
            "pbs_location_r": int(row["pbs_location_r"]),
            "rtt_location_l": int(row["rtt_location_l"]),
            "rtt_location_r": int(row["rtt_location_r"]),
            "type_sub": bool(row["type_sub"]),
            "type_ins": bool(row["type_ins"]),
            "type_del": bool(row["type_del"]),
            "edit_len": int(row["edit_len"]),
            "wt_sequence": str(row["wt_sequence"]),
            "mut_sequence": str(row["mut_sequence"]),
        }
        if member_scores:
            item["member_scores"] = {
                label: float(values[int(index)]) for label, values in member_scores.items()
            }
        rows.append(item)
    return rows


def execute_design(
    request: DesignRequest,
    *,
    progress_log: Optional[ProgressLog] = None,
) -> Dict[str, Any]:
    """Enumerate qualified pegRNAs, score with the selected model(s), and rank."""
    log = progress_log or _null_log
    resolved_device_id = resolve_device_id(request.device or AUTO_DEVICE)
    device = resolve_device(resolved_device_id)

    log(
        f"Enumerating pegRNA designs policy={request.design_policy} "
        f"max_edit_distance={request.max_edit_distance}"
    )
    try:
        std_df = enumerate_design_candidates(
            request.sequence,
            design_policy=request.design_policy,
            max_edit_distance=request.max_edit_distance,
        )
    except ValueError as exc:
        raise DesignError(str(exc)) from exc

    if std_df.empty:
        return {
            "design_policy": request.design_policy,
            "mode": request.mode,
            "n_candidates": 0,
            "n_qualified": 0,
            "designs": [],
            "device": resolved_device_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "No SpCas9 NGG sites reached the edit under the selected policy",
        }

    mask = apply_design_ruleset_mask(std_df, [request.design_policy])
    qualified = std_df.loc[mask].reset_index(drop=True)
    log(f"Enumerated {len(std_df)} candidates; {len(qualified)} pass {request.design_policy}")
    if qualified.empty:
        return {
            "design_policy": request.design_policy,
            "mode": request.mode,
            "n_candidates": int(len(std_df)),
            "n_qualified": 0,
            "designs": [],
            "device": resolved_device_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "Candidates were generated but none satisfied the design ruleset",
        }

    member_scores: Dict[str, List[float]] = {}
    if request.mode == "single":
        model_name = str(request.model_name).strip().lower()
        weights = str(request.weights).strip()
        try:
            model_format = model_registry.get(model_name).pe_db_format
        except ValueError as exc:
            raise DesignError(str(exc)) from exc
        native = _convert_standardized(qualified, model_format, progress_log=log)
        scores = np.asarray(
            _predict_member(
                EnsembleMember(model_name=model_name, weights=weights),
                native,
                device=device,
                progress_log=log,
            ),
            dtype=float,
        )
        scorer = {"model_name": model_name, "weights": weights}
    else:
        members = list(request.members or [])
        prediction_matrix_cols: List[np.ndarray] = []
        for member in members:
            model_name = member.model_name.strip().lower()
            try:
                model_format = model_registry.get(model_name).pe_db_format
            except ValueError as exc:
                raise DesignError(str(exc)) from exc
            native = _convert_standardized(qualified, model_format, progress_log=log)
            preds = _predict_member(member, native, device=device, progress_log=log)
            label = f"{model_name}:{member.weights}"
            member_scores[label] = [float(value) for value in preds]
            prediction_matrix_cols.append(np.asarray(preds, dtype=float))
        matrix = np.column_stack(prediction_matrix_cols)
        combine_options = dict(request.combine_options or {})
        if request.combine == "weighted_mean" and "weights" not in combine_options:
            explicit = [member.member_weight for member in members]
            if all(weight is not None for weight in explicit):
                combine_options["weights"] = [float(weight) for weight in explicit]
        scores = combine_predictions(
            matrix,
            method=request.combine,
            options=combine_options,
        )
        scorer = {
            "combine": request.combine,
            "combine_options": combine_options,
            "members": [
                {
                    "model_name": member.model_name,
                    "weights": member.weights,
                    "member_weight": member.member_weight,
                }
                for member in members
            ],
        }

    if len(scores) != len(qualified):
        raise DesignError(
            f"Prediction count {len(scores)} does not match candidates {len(qualified)}"
        )

    designs = _design_result_rows(
        qualified,
        scores,
        member_scores=member_scores or None,
    )
    if request.top_k is not None:
        designs = designs[: int(request.top_k)]

    return {
        "design_policy": request.design_policy,
        "mode": request.mode,
        "scorer": scorer,
        "n_candidates": int(len(std_df)),
        "n_qualified": int(len(qualified)),
        "n_returned": int(len(designs)),
        "designs": designs,
        "device": resolved_device_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
