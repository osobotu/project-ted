"""Core types consumed by the FPL engine."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Position(StrEnum):
    """An FPL player's registered position."""

    GOALKEEPER = "goalkeeper"
    DEFENDER = "defender"
    MIDFIELDER = "midfielder"
    FORWARD = "forward"


class Player(BaseModel):
    """The player information required by the engine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    player_id: int = Field(gt=0, strict=True)
    team_id: int = Field(gt=0, strict=True)
    position: Position
    now_cost: int = Field(gt=0, strict=True)


class Pick(BaseModel):
    """One player held in a simulated FPL squad."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    player_id: int = Field(gt=0, strict=True)
    purchase_price: int = Field(gt=0, strict=True)
    squad_position: int = Field(gt=0, strict=True)
    is_captain: bool = False
    is_vice_captain: bool = False


class Squad(BaseModel):
    """A simulated FPL squad and its remaining bank."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    picks: tuple[Pick, ...]
    bank: int = Field(ge=0, strict=True)
