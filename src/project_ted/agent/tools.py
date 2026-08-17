"""Read-only agent tools backed by a frozen FPL catalog."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from project_ted.data.catalog import (
    CatalogPlayer,
    Fixture,
    SnapshotCatalog,
    Team,
)
from project_ted.engine.models import Position

type PlayerSort = Literal[
    "total_points",
    "form",
    "points_per_game",
    "selected_by_percent",
    "price",
]


class FindPlayersInput(BaseModel):
    """Arguments accepted by the find_players tool."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = ""
    position: Position | None = None
    team_id: int | None = Field(default=None, gt=0, strict=True)
    maximum_price_tenths: int | None = Field(
        default=None,
        gt=0,
        strict=True,
    )
    sort_by: PlayerSort = "total_points"
    limit: int = Field(default=20, ge=1, le=50, strict=True)


class ComparePlayersInput(BaseModel):
    """Arguments accepted by the compare_players tool."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    player_ids: tuple[int, ...] = Field(
        min_length=2,
        max_length=15,
    )

    @field_validator("player_ids")
    @classmethod
    def player_ids_must_be_unique(
        cls,
        value: tuple[int, ...],
    ) -> tuple[int, ...]:
        if len(value) != len(set(value)):
            raise ValueError("player_ids must be unique")

        return value


class GetFixturesInput(BaseModel):
    """Arguments accepted by the get_fixtures tool."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gameweek: int = Field(gt=0, strict=True)
    team_id: int | None = Field(default=None, gt=0, strict=True)


class GetPlayerNewsInput(BaseModel):
    """Arguments accepted by the get_player_news tool."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    player_ids: tuple[int, ...] | None = Field(
        default=None,
        min_length=1,
        max_length=15,
    )

    @field_validator("player_ids")
    @classmethod
    def player_ids_must_be_unique(
        cls,
        value: tuple[int, ...] | None,
    ) -> tuple[int, ...] | None:
        if value is not None and len(value) != len(set(value)):
            raise ValueError("player_ids must be unique")

        return value


class PlayerView(BaseModel):
    """Agent-facing player information."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    player_id: int
    name: str
    team_id: int
    team: str
    position: Position
    price_tenths: int
    price_millions: float
    status: str
    news: str
    chance_of_playing_next_round: int | None
    form: float
    points_per_game: float
    selected_by_percent: float
    total_points: int
    minutes: int


class FixtureView(BaseModel):
    """Agent-facing fixture information with readable team names."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fixture_id: int
    gameweek: int | None
    kickoff: datetime | None
    home_team_id: int
    home_team: str
    away_team_id: int
    away_team: str
    home_difficulty: int
    away_difficulty: int
    started: bool
    finished: bool
    home_score: int | None
    away_score: int | None


def find_players(
    catalog: SnapshotCatalog,
    arguments: FindPlayersInput,
) -> tuple[PlayerView, ...]:
    """Find and rank players from one frozen catalog."""

    teams = _teams_by_id(catalog)
    _validate_team_id(arguments.team_id, teams)

    query = arguments.query.strip().casefold()

    matches = [
        player
        for player in catalog.players
        if _player_matches(
            player,
            query=query,
            position=arguments.position,
            team_id=arguments.team_id,
            maximum_price_tenths=arguments.maximum_price_tenths,
        )
    ]

    matches.sort(
        key=lambda player: (
            -_sort_value(player, arguments.sort_by),
            player.web_name.casefold(),
            player.player_id,
        )
    )

    return tuple(_player_view(player, teams) for player in matches[: arguments.limit])


def compare_players(
    catalog: SnapshotCatalog,
    arguments: ComparePlayersInput,
) -> tuple[PlayerView, ...]:
    """Return selected players in the requested order."""

    players = _players_for_ids(catalog, arguments.player_ids)
    teams = _teams_by_id(catalog)

    return tuple(_player_view(player, teams) for player in players)


def get_fixtures(
    catalog: SnapshotCatalog,
    arguments: GetFixturesInput,
) -> tuple[FixtureView, ...]:
    """Return fixtures for a gameweek, optionally limited to one team."""

    teams = _teams_by_id(catalog)
    _validate_team_id(arguments.team_id, teams)

    fixtures = [
        fixture
        for fixture in catalog.fixtures
        if fixture.gameweek == arguments.gameweek
        and (
            arguments.team_id is None
            or arguments.team_id in {fixture.home_team_id, fixture.away_team_id}
        )
    ]

    fixtures.sort(
        key=lambda fixture: (
            fixture.kickoff or datetime.max.replace(tzinfo=UTC),
            fixture.fixture_id,
        )
    )

    return tuple(_fixture_view(fixture, teams) for fixture in fixtures)


def get_player_news(
    catalog: SnapshotCatalog,
    arguments: GetPlayerNewsInput,
) -> tuple[PlayerView, ...]:
    """Return news for requested players or every flagged player."""

    if arguments.player_ids is None:
        players = tuple(player for player in catalog.players if _has_relevant_news(player))
        players = tuple(
            sorted(
                players,
                key=lambda player: (
                    player.web_name.casefold(),
                    player.player_id,
                ),
            )
        )
    else:
        players = _players_for_ids(catalog, arguments.player_ids)

    teams = _teams_by_id(catalog)

    return tuple(_player_view(player, teams) for player in players)


def _teams_by_id(catalog: SnapshotCatalog) -> dict[int, Team]:
    return {team.team_id: team for team in catalog.teams}


def _players_for_ids(
    catalog: SnapshotCatalog,
    player_ids: tuple[int, ...],
) -> tuple[CatalogPlayer, ...]:
    players_by_id = {player.player_id: player for player in catalog.players}
    missing_ids = [player_id for player_id in player_ids if player_id not in players_by_id]

    if missing_ids:
        joined_ids = ", ".join(str(player_id) for player_id in missing_ids)
        raise ValueError(f"unknown player IDs: {joined_ids}")

    return tuple(players_by_id[player_id] for player_id in player_ids)


def _validate_team_id(
    team_id: int | None,
    teams: dict[int, Team],
) -> None:
    if team_id is not None and team_id not in teams:
        raise ValueError(f"unknown team ID: {team_id}")


def _player_matches(
    player: CatalogPlayer,
    *,
    query: str,
    position: Position | None,
    team_id: int | None,
    maximum_price_tenths: int | None,
) -> bool:
    searchable_name = (f"{player.first_name} {player.second_name} {player.web_name}").casefold()

    return (
        (not query or query in searchable_name)
        and (position is None or player.position is position)
        and (team_id is None or player.team_id == team_id)
        and (maximum_price_tenths is None or player.now_cost <= maximum_price_tenths)
    )


def _sort_value(
    player: CatalogPlayer,
    sort_by: PlayerSort,
) -> float:
    if sort_by == "total_points":
        return float(player.total_points)
    if sort_by == "form":
        return player.form
    if sort_by == "points_per_game":
        return player.points_per_game
    if sort_by == "selected_by_percent":
        return player.selected_by_percent

    return float(player.now_cost)


def _has_relevant_news(player: CatalogPlayer) -> bool:
    return (
        bool(player.news.strip())
        or player.status != "a"
        or player.chance_of_playing_next_round not in {None, 100}
    )


def _player_view(
    player: CatalogPlayer,
    teams: dict[int, Team],
) -> PlayerView:
    return PlayerView(
        player_id=player.player_id,
        name=player.web_name,
        team_id=player.team_id,
        team=teams[player.team_id].name,
        position=player.position,
        price_tenths=player.now_cost,
        price_millions=player.now_cost / 10,
        status=player.status,
        news=player.news,
        chance_of_playing_next_round=(player.chance_of_playing_next_round),
        form=player.form,
        points_per_game=player.points_per_game,
        selected_by_percent=player.selected_by_percent,
        total_points=player.total_points,
        minutes=player.minutes,
    )


def _fixture_view(
    fixture: Fixture,
    teams: dict[int, Team],
) -> FixtureView:
    return FixtureView(
        fixture_id=fixture.fixture_id,
        gameweek=fixture.gameweek,
        kickoff=fixture.kickoff,
        home_team_id=fixture.home_team_id,
        home_team=teams[fixture.home_team_id].name,
        away_team_id=fixture.away_team_id,
        away_team=teams[fixture.away_team_id].name,
        home_difficulty=fixture.home_difficulty,
        away_difficulty=fixture.away_difficulty,
        started=fixture.started,
        finished=fixture.finished,
        home_score=fixture.home_score,
        away_score=fixture.away_score,
    )
