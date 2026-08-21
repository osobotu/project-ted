"""Authoritative, human-confirmed FPL team state."""

from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from itertools import pairwise
from typing import Annotated, Protocol, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from project_ted.strategy import Chip, Position, SeasonPolicy

PlayerId = Annotated[int, Field(gt=0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


class OwnedPlayer(_FrozenModel):
    """Financial information that FPL maintains for an owned player."""

    player_id: PlayerId
    purchase_price_tenths: PositiveInt
    selling_price_tenths: PositiveInt


class ChipUsage(_FrozenModel):
    """A chip that was confirmed for a completed gameweek."""

    chip: Chip
    gameweek: PositiveInt


class TeamState(_FrozenModel):
    """The confirmed squad and resources available for one planning run.

    `planning_gameweek` is the deadline currently being planned. Chip usage
    therefore contains only gameweeks before `planning_gameweek`.
    """

    season: str = Field(pattern=r"^\d{4}/\d{2}$")
    planning_gameweek: PositiveInt
    squad: tuple[OwnedPlayer, ...] = Field(min_length=1)
    bank_tenths: NonNegativeInt
    free_transfers: NonNegativeInt
    used_chips: tuple[ChipUsage, ...] = ()
    version: PositiveInt
    confirmed_at: datetime

    @field_validator("confirmed_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("team-state confirmation time must include a timezone")

        return value

    @model_validator(mode="after")
    def validate_internal_consistency(self) -> Self:
        player_ids = self.player_ids

        if len(player_ids) != len(set(player_ids)):
            raise ValueError("team-state player IDs must be unique")

        chip_gameweeks = tuple(usage.gameweek for usage in self.used_chips)
        if len(chip_gameweeks) != len(set(chip_gameweeks)):
            raise ValueError("only one chip can be recorded for a gameweek")

        if any(usage.gameweek >= self.planning_gameweek for usage in self.used_chips):
            raise ValueError("chip usage must precede the planning gameweek")

        return self

    @property
    def player_ids(self) -> tuple[int, ...]:
        return tuple(player.player_id for player in self.squad)

    @property
    def squad_selling_value_tenths(self) -> int:
        return sum(player.selling_price_tenths for player in self.squad)

    @property
    def team_value_tenths(self) -> int:
        return self.squad_selling_value_tenths + self.bank_tenths


class TeamStatePlayer(Protocol):
    """The current player information needed to verify a team state."""

    @property
    def id(self) -> int: ...

    @property
    def team_id(self) -> int: ...

    @property
    def position(self) -> Position: ...

    @property
    def price_tenths(self) -> int: ...


class InvalidTeamStateError(ValueError):
    """Report every reason that confirmed state cannot be trusted."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = tuple(violations)
        message = "; ".join(self.violations)
        super().__init__(f"Invalid team state: {message}")


def calculate_selling_price(
    purchase_price_tenths: int,
    current_price_tenths: int,
) -> int:
    """Calculate FPL's selling price from purchase and current prices."""

    if purchase_price_tenths <= 0:
        raise ValueError("purchase price must be positive")

    if current_price_tenths <= 0:
        raise ValueError("current price must be positive")

    if current_price_tenths <= purchase_price_tenths:
        return current_price_tenths

    profit_tenths = current_price_tenths - purchase_price_tenths
    return purchase_price_tenths + profit_tenths // 2


def validate_team_state(
    state: TeamState,
    rules: SeasonPolicy,
    players: Iterable[TeamStatePlayer],
) -> TeamState:
    """Validate confirmed state against one season and player catalogue."""

    violations: list[str] = []

    _validate_state_header(
        state,
        rules,
        violations,
    )
    _validate_state_squad(
        state,
        rules,
        players,
        violations,
    )
    _validate_chip_history(
        state,
        rules,
        violations,
    )

    if violations:
        raise InvalidTeamStateError(violations)

    return state


def _validate_state_header(
    state: TeamState,
    rules: SeasonPolicy,
    violations: list[str],
) -> None:
    if state.season != rules.season:
        violations.append(
            f"state season {state.season} does not match policy season {rules.season}"
        )

    if state.planning_gameweek > rules.total_gameweeks:
        violations.append(f"planning gameweek must not exceed {rules.total_gameweeks}")

    if state.free_transfers > rules.maximum_free_transfers:
        violations.append(f"free transfers must not exceed {rules.maximum_free_transfers}")


def _validate_state_squad(
    state: TeamState,
    rules: SeasonPolicy,
    players: Iterable[TeamStatePlayer],
    violations: list[str],
) -> None:
    if len(state.squad) != rules.squad_size:
        violations.append(
            f"squad must contain {rules.squad_size} players; received {len(state.squad)}"
        )

    player_by_id = {player.id: player for player in players}
    unknown_player_ids = sorted(set(state.player_ids) - player_by_id.keys())

    if unknown_player_ids:
        formatted_ids = ", ".join(str(player_id) for player_id in unknown_player_ids)
        violations.append(f"unknown player IDs: {formatted_ids}")
        return

    squad_players = [player_by_id[player_id] for player_id in state.player_ids]

    _validate_position_counts(
        squad_players,
        rules,
        violations,
    )
    _validate_team_counts(
        squad_players,
        rules,
        violations,
    )
    _validate_selling_prices(
        state,
        player_by_id,
        violations,
    )


def _validate_position_counts(
    players: list[TeamStatePlayer],
    rules: SeasonPolicy,
    violations: list[str],
) -> None:
    position_counts = Counter(player.position for player in players)

    for rule in rules.positions:
        actual_count = position_counts[rule.position]

        if actual_count != rule.squad_count:
            violations.append(
                f"squad must contain {rule.squad_count} "
                f"{rule.position.value} players; "
                f"received {actual_count}"
            )


def _validate_team_counts(
    players: list[TeamStatePlayer],
    rules: SeasonPolicy,
    violations: list[str],
) -> None:
    team_counts = Counter(player.team_id for player in players)

    for team_id, player_count in sorted(team_counts.items()):
        if player_count > rules.max_players_per_team:
            violations.append(
                f"team {team_id} has {player_count} players; "
                f"maximum is {rules.max_players_per_team}"
            )


def _validate_selling_prices(
    state: TeamState,
    player_by_id: dict[int, TeamStatePlayer],
    violations: list[str],
) -> None:
    for owned_player in state.squad:
        current_player = player_by_id[owned_player.player_id]
        expected_price = calculate_selling_price(
            owned_player.purchase_price_tenths,
            current_player.price_tenths,
        )

        if owned_player.selling_price_tenths != expected_price:
            violations.append(
                f"player {owned_player.player_id} selling price must be "
                f"{expected_price}; received "
                f"{owned_player.selling_price_tenths}"
            )


def _validate_chip_history(
    state: TeamState,
    rules: SeasonPolicy,
    violations: list[str],
) -> None:
    rule_by_chip = {rule.chip: rule for rule in rules.chips}
    uses_by_window: Counter[tuple[Chip, int]] = Counter()
    gameweeks_by_chip: dict[Chip, list[int]] = {}

    for usage in state.used_chips:
        rule = rule_by_chip.get(usage.chip)

        if rule is None:
            violations.append(f"{usage.chip.value} is not configured for season {rules.season}")
            continue

        window = next(
            (
                candidate
                for candidate in rule.availability_windows
                if candidate.contains(usage.gameweek)
            ),
            None,
        )

        if window is None:
            violations.append(f"{usage.chip.value} is not available in gameweek {usage.gameweek}")
            continue

        window_key = (
            usage.chip,
            window.first_gameweek,
        )
        uses_by_window[window_key] += 1
        gameweeks_by_chip.setdefault(
            usage.chip,
            [],
        ).append(usage.gameweek)

        if uses_by_window[window_key] > rule.uses_per_window:
            violations.append(
                f"{usage.chip.value} can be used only {rule.uses_per_window} time per window"
            )

    for chip, gameweeks in gameweeks_by_chip.items():
        rule = rule_by_chip[chip]

        for earlier, later in pairwise(sorted(gameweeks)):
            difference = later - earlier

            if difference < rule.minimum_gameweek_difference_between_uses:
                violations.append(
                    f"{chip.value} uses must be at least "
                    f"{rule.minimum_gameweek_difference_between_uses} "
                    f"gameweeks apart"
                )
