"""Immutable records of agent decisions locked before an FPL deadline."""

import re
from datetime import datetime
from typing import Self
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from project_ted.planning import AgentProvider, GameweekPlan


class LockedGameweekDecision(BaseModel):
    """One successful provider plan preserved for later evaluation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    source_run_id: UUID
    provider: AgentProvider
    model_name: str = Field(min_length=1)
    plan: GameweekPlan
    transfer_cost_points: int = Field(
        default=0,
        ge=0,
    )
    deadline_at: datetime
    locked_at: datetime

    @field_validator("deadline_at", "locked_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("decision timestamps must include a timezone")

        return value

    @model_validator(mode="after")
    def validate_lock(self) -> Self:
        if re.fullmatch(r"\d{4}/\d{2}", self.plan.season) is None:
            raise ValueError("decision plan must use a canonical FPL season")

        if self.locked_at >= self.deadline_at:
            raise ValueError("decision must be locked before the FPL deadline")

        return self
