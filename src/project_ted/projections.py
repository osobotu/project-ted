"""Deterministic six-gameweek player projections."""

from collections import defaultdict
from collections.abc import Iterable
from decimal import ROUND_HALF_UP, Decimal
from typing import (
    Annotated,
    Final,
    Literal,
    Protocol,
    Self,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

PlayerId = Annotated[int, Field(gt=0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
AvailabilityPercent = Annotated[int, Field(ge=0, le=100)]

_PROJECTION_HORIZON: Final = 6
_MILLI_POINTS: Final = Decimal("1000")

_DIFFICULTY_MULTIPLIERS: Final = {
    1: Decimal("1.20"),
    2: Decimal("1.10"),
    3: Decimal("1.00"),
    4: Decimal("0.90"),
    5: Decimal("0.80"),
}


class ProjectionPlayer(Protocol):
    """Current player information required by the projection model."""

    @property
    def id(self) -> int: ...

    @property
    def team_id(self) -> int: ...

    @property
    def can_select(self) -> bool: ...

    @property
    def status(self) -> str: ...

    @property
    def chance_of_playing_next_round(
        self,
    ) -> int | None: ...

    @property
    def expected_points_next(
        self,
    ) -> float | None: ...

    @property
    def form(self) -> float: ...

    @property
    def points_per_game(self) -> float: ...

    @property
    def minutes(self) -> int: ...


class ProjectionFixture(Protocol):
    """Fixture information required by the projection model."""

    @property
    def gameweek(self) -> int | None: ...

    @property
    def home_team_id(self) -> int: ...

    @property
    def away_team_id(self) -> int: ...

    @property
    def home_difficulty(self) -> int | None: ...

    @property
    def away_difficulty(self) -> int | None: ...


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


class PlayerGameweekProjection(_FrozenModel):
    """One player's deterministic projection for one gameweek."""

    gameweek: PositiveInt
    fixture_count: NonNegativeInt
    fixture_difficulties: tuple[int, ...]
    projected_points_milli: NonNegativeInt

    @model_validator(mode="after")
    def validate_fixture_count(self) -> Self:
        if self.fixture_count != len(self.fixture_difficulties):
            raise ValueError("fixture count must match fixture difficulties")

        return self


class PlayerProjection(_FrozenModel):
    """A player's complete projection across the planning horizon."""

    player_id: PlayerId
    baseline_points_milli: NonNegativeInt
    availability_percent: AvailabilityPercent
    gameweeks: tuple[
        PlayerGameweekProjection,
        ...,
    ] = Field(min_length=1)
    total_points_milli: NonNegativeInt

    @model_validator(mode="after")
    def validate_total(self) -> Self:
        expected_total = sum(gameweek.projected_points_milli for gameweek in self.gameweeks)

        if self.total_points_milli != expected_total:
            raise ValueError("projection total must equal its gameweek projections")

        return self


class ProjectionSet(_FrozenModel):
    """All player projections generated for one planning run."""

    method: Literal["bootstrap-v1"] = "bootstrap-v1"
    season: str = Field(pattern=r"^\d{4}/\d{2}$")
    first_gameweek: PositiveInt
    last_gameweek: PositiveInt
    players: tuple[PlayerProjection, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_horizon(self) -> Self:
        if self.first_gameweek > self.last_gameweek:
            raise ValueError("first projection gameweek must not exceed the last")

        expected_gameweeks = tuple(
            range(
                self.first_gameweek,
                self.last_gameweek + 1,
            )
        )

        for player in self.players:
            actual_gameweeks = tuple(projection.gameweek for projection in player.gameweeks)

            if actual_gameweeks != expected_gameweeks:
                raise ValueError("every player must cover the projection horizon")

        return self


def build_player_projections(
    *,
    season: str,
    first_gameweek: int,
    total_gameweeks: int,
    players: Iterable[ProjectionPlayer],
    fixtures: Iterable[ProjectionFixture],
) -> ProjectionSet:
    """Build a stable six-gameweek projection from normalized FPL data."""

    if first_gameweek <= 0:
        raise ValueError("first gameweek must be positive")

    if total_gameweeks < first_gameweek:
        raise ValueError("total gameweeks must include the first gameweek")

    last_gameweek = min(
        first_gameweek + _PROJECTION_HORIZON - 1,
        total_gameweeks,
    )
    gameweeks = tuple(
        range(
            first_gameweek,
            last_gameweek + 1,
        )
    )
    ordered_players = tuple(
        sorted(
            players,
            key=lambda player: player.id,
        )
    )
    player_ids = tuple(player.id for player in ordered_players)

    if len(player_ids) != len(set(player_ids)):
        raise ValueError("projection player IDs must be unique")

    fixture_difficulties = _fixture_difficulties_by_team(
        fixtures,
        first_gameweek=first_gameweek,
        last_gameweek=last_gameweek,
    )
    projections = tuple(
        _project_player(
            player,
            gameweeks,
            fixture_difficulties,
        )
        for player in ordered_players
    )

    return ProjectionSet(
        season=season,
        first_gameweek=first_gameweek,
        last_gameweek=last_gameweek,
        players=projections,
    )


def _project_player(
    player: ProjectionPlayer,
    gameweeks: tuple[int, ...],
    fixture_difficulties: dict[
        tuple[int, int],
        tuple[int, ...],
    ],
) -> PlayerProjection:
    baseline = _baseline_points(player)
    availability_percent = _availability_percent(player)
    availability_multiplier = Decimal(availability_percent) / Decimal("100")

    gameweek_projections = tuple(
        _project_player_gameweek(
            player,
            gameweek,
            baseline,
            availability_multiplier,
            fixture_difficulties,
        )
        for gameweek in gameweeks
    )

    return PlayerProjection(
        player_id=player.id,
        baseline_points_milli=_to_milli_points(baseline),
        availability_percent=availability_percent,
        gameweeks=gameweek_projections,
        total_points_milli=sum(
            projection.projected_points_milli for projection in gameweek_projections
        ),
    )


def _project_player_gameweek(
    player: ProjectionPlayer,
    gameweek: int,
    baseline: Decimal,
    availability_multiplier: Decimal,
    fixture_difficulties: dict[
        tuple[int, int],
        tuple[int, ...],
    ],
) -> PlayerGameweekProjection:
    difficulties = fixture_difficulties.get(
        (
            player.team_id,
            gameweek,
        ),
        (),
    )
    projected_points = sum(
        (baseline * _DIFFICULTY_MULTIPLIERS[difficulty] for difficulty in difficulties),
        start=Decimal("0"),
    )
    projected_points *= availability_multiplier

    return PlayerGameweekProjection(
        gameweek=gameweek,
        fixture_count=len(difficulties),
        fixture_difficulties=difficulties,
        projected_points_milli=_to_milli_points(projected_points),
    )


def _baseline_points(
    player: ProjectionPlayer,
) -> Decimal:
    expected_next = _optional_decimal(player.expected_points_next)
    form = _decimal(player.form)
    points_per_game = _decimal(player.points_per_game)

    if player.minutes == 0 and expected_next is not None:
        return max(
            expected_next,
            Decimal("0"),
        )

    if expected_next is not None:
        baseline = (
            expected_next * Decimal("0.50")
            + form * Decimal("0.30")
            + points_per_game * Decimal("0.20")
        )
    else:
        baseline = form * Decimal("0.60") + points_per_game * Decimal("0.40")

    return max(
        baseline,
        Decimal("0"),
    )


def _availability_percent(
    player: ProjectionPlayer,
) -> int:
    if not player.can_select:
        return 0

    chance = player.chance_of_playing_next_round

    if chance is not None:
        return min(
            max(chance, 0),
            100,
        )

    if player.status == "a":
        return 100

    return 50


def _fixture_difficulties_by_team(
    fixtures: Iterable[ProjectionFixture],
    *,
    first_gameweek: int,
    last_gameweek: int,
) -> dict[tuple[int, int], tuple[int, ...]]:
    grouped: defaultdict[
        tuple[int, int],
        list[int],
    ] = defaultdict(list)

    for fixture in fixtures:
        gameweek = fixture.gameweek

        if gameweek is None or gameweek < first_gameweek or gameweek > last_gameweek:
            continue

        grouped[
            (
                fixture.home_team_id,
                gameweek,
            )
        ].append(_normalized_difficulty(fixture.home_difficulty))
        grouped[
            (
                fixture.away_team_id,
                gameweek,
            )
        ].append(_normalized_difficulty(fixture.away_difficulty))

    return {key: tuple(sorted(difficulties)) for key, difficulties in grouped.items()}


def _normalized_difficulty(
    difficulty: int | None,
) -> int:
    if difficulty in _DIFFICULTY_MULTIPLIERS:
        return difficulty

    return 3


def _decimal(value: float) -> Decimal:
    return Decimal(str(value))


def _optional_decimal(
    value: float | None,
) -> Decimal | None:
    if value is None:
        return None

    return _decimal(value)


def _to_milli_points(
    value: Decimal,
) -> int:
    scaled = value * _MILLI_POINTS

    return int(
        scaled.quantize(
            Decimal("1"),
            rounding=ROUND_HALF_UP,
        )
    )
