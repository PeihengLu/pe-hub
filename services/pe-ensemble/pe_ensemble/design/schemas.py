"""Request schemas for interactive pegRNA design."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..ensemble.combine import COMBINE_METHODS, CombineMethod
from ..ensemble.schemas import EnsembleMember

DesignPolicy = Literal["optiprime", "anzalone"]
DesignMode = Literal["single", "ensemble"]


class DesignRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    sequence: str = Field(
        ...,
        min_length=1,
        description="Target-strand DNA with one (pre/after) edit annotation",
    )
    design_policy: DesignPolicy = "optiprime"
    mode: DesignMode = "single"
    model_name: Optional[str] = None
    weights: Optional[str] = None
    members: Optional[List[EnsembleMember]] = None
    combine: CombineMethod = "mean"
    combine_options: Dict[str, Any] = Field(default_factory=dict)
    device: Optional[str] = "auto"
    max_edit_distance: int = Field(default=40, ge=0, le=200)
    top_k: Optional[int] = Field(
        default=50,
        ge=1,
        description="Return at most this many ranked designs (None = all)",
    )

    @field_validator("design_policy")
    @classmethod
    def _normalize_policy(cls, value: str) -> str:
        key = value.strip().lower()
        if key in {"hsu", "optiprime"}:
            return "optiprime"
        if key == "anzalone":
            return "anzalone"
        raise ValueError("design_policy must be 'optiprime' or 'anzalone'")

    @field_validator("combine")
    @classmethod
    def _validate_combine(cls, value: str) -> str:
        if value not in COMBINE_METHODS:
            raise ValueError(f"combine must be one of: {', '.join(COMBINE_METHODS)}")
        return value

    @model_validator(mode="after")
    def _validate_mode(self) -> "DesignRequest":
        if self.mode == "single":
            if not self.model_name or not str(self.model_name).strip():
                raise ValueError("model_name is required when mode='single'")
            if not self.weights or not str(self.weights).strip():
                raise ValueError("weights is required when mode='single'")
        else:
            if not self.members or len(self.members) < 2:
                raise ValueError("mode='ensemble' requires at least two members")
            if self.combine == "weighted_mean":
                explicit = [member.member_weight for member in self.members]
                if any(weight is None for weight in explicit):
                    if "weights" not in self.combine_options:
                        raise ValueError(
                            "weighted_mean requires member_weight on each member "
                            "or a weights list in combine_options"
                        )
        return self
