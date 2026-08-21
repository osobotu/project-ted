"""Immutable domain rules for FPL planning and optimization."""

from enum import StrEnum
from itertools import pairwise
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Position(StrEnum):
    """A playing position recognized by Fantasy Premier League."""

    GOALKEEPER = "GKP"
    DEFENDER = "DEF"
    MIDFIELDER = "MID"
    FORWARD = "FWD"


class Chip(StrEnum):
    """A chip that can modify one FPL gameweek decision."""

    WILDCARD = "wildcard"
    FREE_HIT = "free_hit"
    BENCH_BOOST = "bench_boost"
    TRIPLE_CAPTAIN = "triple_captain"


class PositionRule(_FrozenModel):
    """Squad and starting-lineup limits for one position."""

    position: Position
    squad_count: PositiveInt
    minimum_starters: NonNegativeInt
    maximum_starters: PositiveInt

    @model_validator(mode="after")
    def validate_starter_limits(self) -> Self:
        if self.minimum_starters > self.maximum_starters:
            raise ValueError("minimum starters must not exceed maximum starters")

        if self.maximum_starters > self.squad_count:
            raise ValueError("maximum starters must not exceed squad count")

        return self


class GameweekWindow(_FrozenModel):
    """An inclusive range of gameweeks in which a rule applies."""

    first_gameweek: PositiveInt
    last_gameweek: PositiveInt

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        if self.first_gameweek > self.last_gameweek:
            raise ValueError("first gameweek must not be after last gameweek")

        return self

    def contains(self, gameweek: int) -> bool:
        return self.first_gameweek <= gameweek <= self.last_gameweek


class ChipRule(_FrozenModel):
    """Availability and reuse constraints for one chip."""

    chip: Chip
    availability_windows: tuple[GameweekWindow, ...] = Field(min_length=1)
    uses_per_window: PositiveInt = 1
    minimum_gameweek_difference_between_uses: PositiveInt = 1

    @model_validator(mode="after")
    def validate_availability_windows(self) -> Self:
        ordered_windows = tuple(
            sorted(
                self.availability_windows,
                key=lambda window: window.first_gameweek,
            )
        )

        if ordered_windows != self.availability_windows:
            raise ValueError("chip availability windows must be ordered")

        for earlier, later in pairwise(self.availability_windows):
            if earlier.last_gameweek >= later.first_gameweek:
                raise ValueError("chip availability windows must not overlap")

        return self

    def is_available_in(self, gameweek: int) -> bool:
        """Return whether the chip belongs to an active season window.

        This checks the season schedule only. It does not account for whether
        the manager has already used the chip in that window.
        """

        return any(window.contains(gameweek) for window in self.availability_windows)


class SeasonPolicy(_FrozenModel):
    """Every rule needed to validate and optimize one FPL season."""

    season: str = Field(pattern=r"^\d{4}/\d{2}$")
    total_gameweeks: PositiveInt

    squad_size: PositiveInt
    starting_size: PositiveInt
    max_players_per_team: PositiveInt
    budget_tenths: PositiveInt
    positions: tuple[PositionRule, ...] = Field(min_length=1)

    free_transfers_per_gameweek: PositiveInt
    maximum_free_transfers: PositiveInt
    additional_transfer_cost_points: PositiveInt

    maximum_chips_per_gameweek: Literal[1]
    chips: tuple[ChipRule, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        self._validate_squad_rules()
        self._validate_chip_rules()
        return self

    def position_rule(self, position: Position) -> PositionRule:
        """Return the rule for one position."""

        return next(rule for rule in self.positions if rule.position is position)

    def chip_rule(self, chip: Chip) -> ChipRule:
        """Return the season rule for one chip."""

        return next(rule for rule in self.chips if rule.chip is chip)

    def _validate_squad_rules(self) -> None:
        configured_positions = tuple(rule.position for rule in self.positions)

        if len(configured_positions) != len(set(configured_positions)):
            raise ValueError("positions must not contain duplicates")

        squad_count = sum(rule.squad_count for rule in self.positions)
        if squad_count != self.squad_size:
            raise ValueError("position squad counts must equal the squad size")

        minimum_starters = sum(rule.minimum_starters for rule in self.positions)
        maximum_starters = sum(rule.maximum_starters for rule in self.positions)

        if not minimum_starters <= self.starting_size <= maximum_starters:
            raise ValueError("starting size must be possible under the position rules")

        if self.starting_size >= self.squad_size:
            raise ValueError("starting size must be smaller than squad size")

        if self.max_players_per_team > self.squad_size:
            raise ValueError("maximum players per team must not exceed squad size")

    def _validate_chip_rules(self) -> None:
        configured_chips = tuple(rule.chip for rule in self.chips)

        if len(configured_chips) != len(set(configured_chips)):
            raise ValueError("chips must not contain duplicates")

        for rule in self.chips:
            for window in rule.availability_windows:
                if window.last_gameweek > self.total_gameweeks:
                    raise ValueError("chip availability must be within the season")


_FIRST_HALF = GameweekWindow(
    first_gameweek=1,
    last_gameweek=19,
)
_FIRST_HALF_AFTER_INITIAL_SELECTION = GameweekWindow(
    first_gameweek=2,
    last_gameweek=19,
)
_SECOND_HALF = GameweekWindow(
    first_gameweek=20,
    last_gameweek=38,
)

_POLICY_2026_27 = SeasonPolicy(
    season="2026/27",
    total_gameweeks=38,
    squad_size=15,
    starting_size=11,
    max_players_per_team=3,
    budget_tenths=1000,
    positions=(
        PositionRule(
            position=Position.GOALKEEPER,
            squad_count=2,
            minimum_starters=1,
            maximum_starters=1,
        ),
        PositionRule(
            position=Position.DEFENDER,
            squad_count=5,
            minimum_starters=3,
            maximum_starters=5,
        ),
        PositionRule(
            position=Position.MIDFIELDER,
            squad_count=5,
            minimum_starters=2,
            maximum_starters=5,
        ),
        PositionRule(
            position=Position.FORWARD,
            squad_count=3,
            minimum_starters=1,
            maximum_starters=3,
        ),
    ),
    free_transfers_per_gameweek=1,
    maximum_free_transfers=5,
    additional_transfer_cost_points=4,
    maximum_chips_per_gameweek=1,
    chips=(
        ChipRule(
            chip=Chip.WILDCARD,
            availability_windows=(
                _FIRST_HALF_AFTER_INITIAL_SELECTION,
                _SECOND_HALF,
            ),
        ),
        ChipRule(
            chip=Chip.FREE_HIT,
            availability_windows=(
                _FIRST_HALF_AFTER_INITIAL_SELECTION,
                _SECOND_HALF,
            ),
            minimum_gameweek_difference_between_uses=2,
        ),
        ChipRule(
            chip=Chip.BENCH_BOOST,
            availability_windows=(
                _FIRST_HALF,
                _SECOND_HALF,
            ),
        ),
        ChipRule(
            chip=Chip.TRIPLE_CAPTAIN,
            availability_windows=(
                _FIRST_HALF,
                _SECOND_HALF,
            ),
        ),
    ),
)

_SEASON_POLICIES = {
    _POLICY_2026_27.season: _POLICY_2026_27,
}


class UnsupportedSeasonError(ValueError):
    """Report that Project Ted has no verified policy for a season."""


def season_policy_for(season: str) -> SeasonPolicy:
    """Return the verified immutable policy for a supported FPL season."""

    try:
        return _SEASON_POLICIES[season]
    except KeyError as error:
        raise UnsupportedSeasonError(
            f"No verified FPL policy exists for season {season}"
        ) from error
