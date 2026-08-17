"""Contracts for a weekly FPL decision and its provider outcomes."""

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Self
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

PlayerId = Annotated[int, Field(gt=0)]


class AgentProvider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class RunStatus(StrEnum):
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"


class GameweekPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    season: str = Field(min_length=1)
    gameweek: int = Field(gt=0)
    squad: tuple[PlayerId, ...]
    starting_xi: tuple[PlayerId, ...]
    bench: tuple[PlayerId, ...]
    captain_id: PlayerId
    vice_captain_id: PlayerId
    rationale: str = Field(min_length=1)
    risks: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_selections(self) -> Self:
        selections = {
            "squad": self.squad,
            "starting_xi": self.starting_xi,
            "bench": self.bench,
        }

        for name, player_ids in selections.items():
            if len(set(player_ids)) != len(player_ids):
                raise ValueError(f"{name} player IDs must be unique")

        squad = set(self.squad)
        starting_xi = set(self.starting_xi)
        bench = set(self.bench)

        if not starting_xi <= squad:
            raise ValueError("starting_xi must contain only squad players")

        if bench != squad - starting_xi:
            raise ValueError("bench must contain every non-starting squad player exactly once")

        if self.captain_id not in starting_xi:
            raise ValueError("captain must be in the starting XI")

        if self.vice_captain_id not in starting_xi:
            raise ValueError("vice-captain must be in the starting XI")

        if self.captain_id == self.vice_captain_id:
            raise ValueError("captain and vice-captain must be different players")

        return self


class AgentOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: AgentProvider
    model: str = Field(min_length=1)
    plan: GameweekPlan | None = None
    error: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def require_plan_or_error(self) -> Self:
        has_plan = self.plan is not None
        has_error = self.error is not None

        if has_plan == has_error:
            raise ValueError("agent outcome must contain exactly one of plan or error")

        return self

    @property
    def succeeded(self) -> bool:
        """Return whether the provider produced a valid plan."""

        return self.plan is not None


class WeeklyRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: UUID
    season: str = Field(min_length=1)
    gameweek: int = Field(gt=0)
    created_at: datetime
    deadline_at: datetime
    outcomes: tuple[AgentOutcome, AgentOutcome]

    @field_validator("created_at", "deadline_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("run timestamps must include a timezone")

        return value

    @model_validator(mode="after")
    def validate_outcomes(self) -> Self:
        providers = {outcome.provider for outcome in self.outcomes}
        required_providers = {
            AgentProvider.OPENAI,
            AgentProvider.ANTHROPIC,
        }

        if providers != required_providers:
            raise ValueError("weekly run must contain one OpenAI and one Anthropic outcome")

        for outcome in self.outcomes:
            plan = outcome.plan

            if plan is not None and (plan.season != self.season or plan.gameweek != self.gameweek):
                raise ValueError("agent plans must match the weekly run season and gameweek")

        return self

    @property
    def status(self) -> RunStatus:
        successes = sum(outcome.succeeded for outcome in self.outcomes)

        if successes == 2:
            return RunStatus.SUCCEEDED

        if successes == 1:
            return RunStatus.PARTIAL

        return RunStatus.FAILED
