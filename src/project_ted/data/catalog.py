"""Translate frozen FPL responses into agent and engine data."""

from dataclasses import dataclass
from datetime import datetime, timedelta

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
)

from project_ted.data.snapshots import RunSnapshot
from project_ted.engine.models import Player, Position

_POSITION_BY_ELEMENT_TYPE = {
    1: Position.GOALKEEPER,
    2: Position.DEFENDER,
    3: Position.MIDFIELDER,
    4: Position.FORWARD,
}


class CatalogDataError(ValueError):
    """Raised when frozen FPL data cannot form a valid catalogue."""


class CatalogPlayer(BaseModel):
    """Player information available to tools and agents."""

    model_config = ConfigDict(
        frozen=True,
        extra="ignore",
        populate_by_name=True,
    )

    player_id: int = Field(alias="id", gt=0, strict=True)
    first_name: str
    second_name: str
    web_name: str
    team_id: int = Field(alias="team", gt=0, strict=True)
    position: Position = Field(alias="element_type")
    now_cost: int = Field(gt=0, strict=True)
    status: str
    news: str
    chance_of_playing_next_round: int | None = Field(
        default=None,
        ge=0,
        le=100,
    )
    form: float
    points_per_game: float
    selected_by_percent: float
    total_points: int
    minutes: int = Field(ge=0)

    @field_validator("position", mode="before")
    @classmethod
    def translate_element_type(cls, value: object) -> Position:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("element_type must be an integer")

        try:
            return _POSITION_BY_ELEMENT_TYPE[value]
        except KeyError as error:
            raise ValueError(f"unknown element_type: {value}") from error

    def to_engine_player(self) -> Player:
        """Return the rule-relevant representation of this player."""

        return Player(
            player_id=self.player_id,
            team_id=self.team_id,
            position=self.position,
            now_cost=self.now_cost,
        )


class Team(BaseModel):
    """An FPL team available for player and fixture tools."""

    model_config = ConfigDict(
        frozen=True,
        extra="ignore",
        populate_by_name=True,
    )

    team_id: int = Field(alias="id", gt=0, strict=True)
    name: str
    short_name: str
    strength: int = Field(gt=0)


class Gameweek(BaseModel):
    """A gameweek and its current completion state."""

    model_config = ConfigDict(
        frozen=True,
        extra="ignore",
        populate_by_name=True,
    )

    gameweek: int = Field(alias="id", gt=0, strict=True)
    name: str
    deadline: datetime = Field(alias="deadline_time")
    finished: bool
    data_checked: bool
    is_current: bool
    is_next: bool

    @field_validator("deadline")
    @classmethod
    def deadline_must_be_utc(cls, value: datetime) -> datetime:
        if value.utcoffset() != timedelta(0):
            raise ValueError("deadline_time must be UTC")

        return value


class Fixture(BaseModel):
    """One scheduled Premier League fixture."""

    model_config = ConfigDict(
        frozen=True,
        extra="ignore",
        populate_by_name=True,
    )

    fixture_id: int = Field(alias="id", gt=0, strict=True)
    gameweek: int | None = Field(alias="event", default=None, ge=1)
    kickoff: datetime | None = Field(alias="kickoff_time", default=None)
    home_team_id: int = Field(alias="team_h", gt=0, strict=True)
    away_team_id: int = Field(alias="team_a", gt=0, strict=True)
    home_difficulty: int = Field(
        alias="team_h_difficulty",
        ge=1,
        le=5,
    )
    away_difficulty: int = Field(
        alias="team_a_difficulty",
        ge=1,
        le=5,
    )
    started: bool
    finished: bool
    home_score: int | None = Field(
        alias="team_h_score",
        default=None,
        ge=0,
    )
    away_score: int | None = Field(
        alias="team_a_score",
        default=None,
        ge=0,
    )

    @field_validator("kickoff")
    @classmethod
    def kickoff_must_be_utc(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        if value is not None and value.utcoffset() != timedelta(0):
            raise ValueError("kickoff_time must be UTC")

        return value


class _BootstrapPayload(BaseModel):
    """The bootstrap collections required by the catalogue."""

    model_config = ConfigDict(extra="ignore")

    elements: tuple[CatalogPlayer, ...] = Field(min_length=1)
    teams: tuple[Team, ...] = Field(min_length=1)
    events: tuple[Gameweek, ...] = Field(min_length=1)


_FIXTURES_ADAPTER = TypeAdapter(tuple[Fixture, ...])


@dataclass(frozen=True, slots=True)
class SnapshotCatalog:
    """Parsed FPL information from one frozen run snapshot."""

    snapshot_id: str
    players: tuple[CatalogPlayer, ...]
    teams: tuple[Team, ...]
    gameweeks: tuple[Gameweek, ...]
    fixtures: tuple[Fixture, ...]

    def engine_players(self) -> dict[int, Player]:
        """Return players indexed for engine validation."""

        return {player.player_id: player.to_engine_player() for player in self.players}


def catalog_from_snapshot(snapshot: RunSnapshot) -> SnapshotCatalog:
    """Parse and cross-check one frozen FPL snapshot."""

    try:
        bootstrap = _BootstrapPayload.model_validate_json(snapshot.bootstrap.payload)
        fixtures = _FIXTURES_ADAPTER.validate_json(snapshot.fixtures.payload)
    except ValidationError as error:
        raise CatalogDataError("snapshot contains invalid FPL data") from error

    player_ids = [player.player_id for player in bootstrap.elements]
    if len(player_ids) != len(set(player_ids)):
        raise CatalogDataError("snapshot contains duplicate player IDs")

    team_ids = {team.team_id for team in bootstrap.teams}
    if len(team_ids) != len(bootstrap.teams):
        raise CatalogDataError("snapshot contains duplicate team IDs")

    gameweek_ids = {gameweek.gameweek for gameweek in bootstrap.events}
    if len(gameweek_ids) != len(bootstrap.events):
        raise CatalogDataError("snapshot contains duplicate gameweek IDs")

    for player in bootstrap.elements:
        if player.team_id not in team_ids:
            raise CatalogDataError(
                f"player {player.player_id} references unknown team {player.team_id}"
            )

    for fixture in fixtures:
        if fixture.home_team_id not in team_ids:
            raise CatalogDataError(
                f"fixture {fixture.fixture_id} references unknown home team {fixture.home_team_id}"
            )

        if fixture.away_team_id not in team_ids:
            raise CatalogDataError(
                f"fixture {fixture.fixture_id} references unknown away team {fixture.away_team_id}"
            )

        if fixture.gameweek is not None and fixture.gameweek not in gameweek_ids:
            raise CatalogDataError(
                f"fixture {fixture.fixture_id} references unknown gameweek {fixture.gameweek}"
            )

    return SnapshotCatalog(
        snapshot_id=snapshot.snapshot_id,
        players=bootstrap.elements,
        teams=bootstrap.teams,
        gameweeks=bootstrap.events,
        fixtures=fixtures,
    )
