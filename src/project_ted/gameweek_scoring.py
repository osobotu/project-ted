"""Evaluate locked teams using finalized official FPL player points."""

from collections import Counter
from datetime import datetime
from typing import Annotated, Self
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from project_ted.decision import LockedGameweekDecision
from project_ted.planning import AgentProvider, GameweekPlan
from project_ted.strategy import Chip, Position, SeasonPolicy

PlayerId = Annotated[int, Field(gt=0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


class OfficialPlayerPoints(_FrozenModel):
    """One player's finalized official total for a gameweek.

    `played` is true when FPL considers the player to have featured. The data
    adapter must account for appearances and cards rather than inferring this
    solely from the points total.
    """

    player_id: PlayerId
    position: Position
    points: int
    played: bool


class OfficialGameweekPoints(_FrozenModel):
    """Finalized official points available for one FPL gameweek."""

    season: str = Field(pattern=r"^\d{4}/\d{2}$")
    gameweek: PositiveInt
    finalized_at: datetime
    players: tuple[OfficialPlayerPoints, ...] = Field(min_length=1)

    @field_validator("finalized_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("official-points timestamp must include a timezone")

        return value

    @model_validator(mode="after")
    def require_unique_players(self) -> Self:
        player_ids = tuple(player.player_id for player in self.players)

        if len(player_ids) != len(set(player_ids)):
            raise ValueError("official player IDs must be unique")

        return self


class AutomaticSubstitution(_FrozenModel):
    """One automatic replacement applied after the gameweek."""

    player_out_id: PlayerId
    player_in_id: PlayerId

    @model_validator(mode="after")
    def require_different_players(self) -> Self:
        if self.player_out_id == self.player_in_id:
            raise ValueError("automatic substitution players must be different")

        return self


class PlayerContribution(_FrozenModel):
    """Points contributed by one player after captaincy is applied."""

    player_id: PlayerId
    points: int
    multiplier: int = Field(ge=1, le=3)
    total_points: int

    @model_validator(mode="after")
    def validate_total(self) -> Self:
        expected_total = self.points * self.multiplier

        if self.total_points != expected_total:
            raise ValueError("player contribution total does not match its multiplier")

        return self


class GameweekScore(_FrozenModel):
    """The reproducible result of evaluating one locked decision."""

    source_run_id: UUID
    provider: AgentProvider
    model_name: str = Field(min_length=1)
    season: str = Field(pattern=r"^\d{4}/\d{2}$")
    gameweek: PositiveInt
    chip: Chip | None
    effective_captain_id: PlayerId | None
    substitutions: tuple[AutomaticSubstitution, ...]
    contributions: tuple[PlayerContribution, ...] = Field(min_length=1)
    gross_points: int
    transfer_cost_points: NonNegativeInt
    total_points: int
    finalized_at: datetime

    @model_validator(mode="after")
    def validate_totals(self) -> Self:
        expected_gross_points = sum(
            contribution.total_points for contribution in self.contributions
        )

        if self.gross_points != expected_gross_points:
            raise ValueError("gross points do not match player contributions")

        if self.total_points != self.gross_points - self.transfer_cost_points:
            raise ValueError("total points do not include the transfer cost")

        return self


class GameweekScoringError(ValueError):
    """Report that a locked decision cannot be scored reliably."""


def evaluate_gameweek(
    decision: LockedGameweekDecision,
    official: OfficialGameweekPoints,
    policy: SeasonPolicy,
) -> GameweekScore:
    """Apply FPL lineup mechanics to finalized official player points."""

    points_by_player = _validate_scoring_inputs(
        decision,
        official,
        policy,
    )
    plan = decision.plan

    if plan.chip is Chip.BENCH_BOOST:
        counted_player_ids = plan.squad
        substitutions: tuple[AutomaticSubstitution, ...] = ()
    else:
        counted_player_ids, substitutions = _apply_automatic_substitutions(
            plan,
            points_by_player,
            policy,
        )

    effective_captain_id = _effective_captain(
        plan,
        points_by_player,
    )
    captain_multiplier = 3 if plan.chip is Chip.TRIPLE_CAPTAIN else 2

    contributions = tuple(
        _player_contribution(
            player_id,
            points_by_player,
            effective_captain_id=effective_captain_id,
            captain_multiplier=captain_multiplier,
        )
        for player_id in counted_player_ids
    )
    gross_points = sum(contribution.total_points for contribution in contributions)

    return GameweekScore(
        source_run_id=decision.source_run_id,
        provider=decision.provider,
        model_name=decision.model_name,
        season=plan.season,
        gameweek=plan.gameweek,
        chip=plan.chip,
        effective_captain_id=effective_captain_id,
        substitutions=substitutions,
        contributions=contributions,
        gross_points=gross_points,
        transfer_cost_points=decision.transfer_cost_points,
        total_points=(gross_points - decision.transfer_cost_points),
        finalized_at=official.finalized_at,
    )


def _validate_scoring_inputs(
    decision: LockedGameweekDecision,
    official: OfficialGameweekPoints,
    policy: SeasonPolicy,
) -> dict[int, OfficialPlayerPoints]:
    plan = decision.plan

    if official.season != plan.season or official.gameweek != plan.gameweek:
        raise GameweekScoringError("official points do not match the locked decision")

    if policy.season != plan.season:
        raise GameweekScoringError("season policy does not match the locked decision")

    if decision.transfer_cost_points % policy.additional_transfer_cost_points != 0:
        raise GameweekScoringError(
            f"transfer cost must be a multiple of {policy.additional_transfer_cost_points}"
        )

    if plan.chip in {Chip.WILDCARD, Chip.FREE_HIT} and decision.transfer_cost_points != 0:
        raise GameweekScoringError("wildcard and free-hit decisions cannot have transfer costs")

    if plan.chip is not None and not policy.chip_rule(plan.chip).is_available_in(plan.gameweek):
        raise GameweekScoringError(
            f"{plan.chip.value} is not available in gameweek {plan.gameweek}"
        )

    points_by_player = {player.player_id: player for player in official.players}
    missing_player_ids = sorted(set(plan.squad) - points_by_player.keys())

    if missing_player_ids:
        formatted_ids = ", ".join(str(player_id) for player_id in missing_player_ids)
        raise GameweekScoringError(f"missing official points for player IDs: {formatted_ids}")

    if len(plan.squad) != policy.squad_size:
        raise GameweekScoringError("locked squad size does not match season policy")

    squad_position_counts = Counter(
        points_by_player[player_id].position for player_id in plan.squad
    )

    for rule in policy.positions:
        if squad_position_counts[rule.position] != rule.squad_count:
            raise GameweekScoringError("locked squad positions do not match season policy")

    if not _has_valid_formation(
        plan.starting_xi,
        points_by_player,
        policy,
    ):
        raise GameweekScoringError("locked starting XI does not have a valid formation")

    return points_by_player


def _apply_automatic_substitutions(
    plan: GameweekPlan,
    points_by_player: dict[int, OfficialPlayerPoints],
    policy: SeasonPolicy,
) -> tuple[
    tuple[int, ...],
    tuple[AutomaticSubstitution, ...],
]:
    lineup = list(plan.starting_xi)
    substitutions: list[AutomaticSubstitution] = []

    _replace_goalkeeper(
        lineup,
        plan.bench,
        points_by_player,
        substitutions,
    )

    for substitute_id in plan.bench:
        substitute = points_by_player[substitute_id]

        if substitute.position is Position.GOALKEEPER or not substitute.played:
            continue

        replacement_index = _eligible_replacement_index(
            lineup,
            substitute_id,
            points_by_player,
            policy,
        )

        if replacement_index is None:
            continue

        replaced_player_id = lineup[replacement_index]
        lineup[replacement_index] = substitute_id
        substitutions.append(
            AutomaticSubstitution(
                player_out_id=replaced_player_id,
                player_in_id=substitute_id,
            )
        )

    return (
        tuple(lineup),
        tuple(substitutions),
    )


def _replace_goalkeeper(
    lineup: list[int],
    bench: tuple[int, ...],
    points_by_player: dict[int, OfficialPlayerPoints],
    substitutions: list[AutomaticSubstitution],
) -> None:
    starting_goalkeeper_index = next(
        index
        for index, player_id in enumerate(lineup)
        if (points_by_player[player_id].position is Position.GOALKEEPER)
    )
    starting_goalkeeper_id = lineup[starting_goalkeeper_index]

    if points_by_player[starting_goalkeeper_id].played:
        return

    bench_goalkeeper_id = next(
        (
            player_id
            for player_id in bench
            if (points_by_player[player_id].position is Position.GOALKEEPER)
        ),
        None,
    )

    if bench_goalkeeper_id is None or not points_by_player[bench_goalkeeper_id].played:
        return

    lineup[starting_goalkeeper_index] = bench_goalkeeper_id
    substitutions.append(
        AutomaticSubstitution(
            player_out_id=starting_goalkeeper_id,
            player_in_id=bench_goalkeeper_id,
        )
    )


def _eligible_replacement_index(
    lineup: list[int],
    substitute_id: int,
    points_by_player: dict[int, OfficialPlayerPoints],
    policy: SeasonPolicy,
) -> int | None:
    substitute_position = points_by_player[substitute_id].position
    missing_indexes = [
        index
        for index, player_id in enumerate(lineup)
        if (
            points_by_player[player_id].position is not Position.GOALKEEPER
            and not points_by_player[player_id].played
        )
    ]
    ordered_indexes = sorted(
        missing_indexes,
        key=lambda index: points_by_player[lineup[index]].position is not substitute_position,
    )

    for index in ordered_indexes:
        candidate_lineup = list(lineup)
        candidate_lineup[index] = substitute_id

        if _has_valid_formation(
            candidate_lineup,
            points_by_player,
            policy,
        ):
            return index

    return None


def _has_valid_formation(
    player_ids: tuple[int, ...] | list[int],
    points_by_player: dict[int, OfficialPlayerPoints],
    policy: SeasonPolicy,
) -> bool:
    if len(player_ids) != policy.starting_size:
        return False

    position_counts = Counter(points_by_player[player_id].position for player_id in player_ids)

    return all(
        rule.minimum_starters <= position_counts[rule.position] <= rule.maximum_starters
        for rule in policy.positions
    )


def _effective_captain(
    plan: GameweekPlan,
    points_by_player: dict[int, OfficialPlayerPoints],
) -> int | None:
    if points_by_player[plan.captain_id].played:
        return plan.captain_id

    if points_by_player[plan.vice_captain_id].played:
        return plan.vice_captain_id

    return None


def _player_contribution(
    player_id: int,
    points_by_player: dict[int, OfficialPlayerPoints],
    *,
    effective_captain_id: int | None,
    captain_multiplier: int,
) -> PlayerContribution:
    official = points_by_player[player_id]
    multiplier = captain_multiplier if player_id == effective_captain_id else 1

    return PlayerContribution(
        player_id=player_id,
        points=official.points,
        multiplier=multiplier,
        total_points=official.points * multiplier,
    )
