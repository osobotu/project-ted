"""Load current FPL information into a stable, provider-neutral context."""

import re
from collections import Counter
from datetime import UTC, datetime
from typing import Self

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, model_validator

from project_ted.planning import GameweekPlan
from project_ted.strategy import Position, SeasonPolicy, season_policy_for

_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
_TIMEOUT_SECONDS = 10.0


class FplDataError(RuntimeError):
    """Report that a reliable current planning context could not be created."""


class InvalidPlanError(ValueError):
    def __init__(self, violations: list[str]) -> None:
        self.violations = tuple(violations)
        message = "; ".join(self.violations)
        super().__init__(f"Invalid FPL plan: {message}")


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Gameweek(_FrozenModel):
    """The next FPL deadline for which agents must make a decision."""

    id: int
    name: str
    deadline_at: datetime


class Team(_FrozenModel):
    """A Premier League team referenced by players and fixtures."""

    id: int
    name: str
    short_name: str


class Player(_FrozenModel):
    """The current FPL information needed to research one player."""

    id: int
    name: str
    team_id: int
    position: Position
    price_tenths: int
    status: str
    chance_of_playing_next_round: int | None
    news: str
    can_select: bool
    total_points: int
    minutes: int
    starts: int
    form: float
    points_per_game: float
    selected_by_percent: float
    expected_points_next: float | None
    expected_goals: float
    expected_assists: float
    expected_goal_involvements: float
    transfers_in_event: int
    transfers_out_event: int


class Fixture(_FrozenModel):
    """A scheduled match and its FPL difficulty ratings."""

    id: int
    gameweek: int | None
    kickoff_at: datetime | None
    home_team_id: int
    away_team_id: int
    home_difficulty: int | None
    away_difficulty: int | None
    started: bool
    finished: bool


class PlanningContext(_FrozenModel):
    """All official FPL information shared by both agents for one run."""

    fetched_at: datetime
    season: str
    target_gameweek: Gameweek
    rules: SeasonPolicy
    teams: tuple[Team, ...]
    players: tuple[Player, ...]
    fixtures: tuple[Fixture, ...]

    @model_validator(mode="after")
    def validate_policy_context(self) -> Self:
        if self.season != self.rules.season:
            raise ValueError("planning context and season policy must identify the same season")

        if self.target_gameweek.id > self.rules.total_gameweeks:
            raise ValueError("target gameweek must be within the season policy")

        return self

    def validate_plan(self, plan: GameweekPlan) -> GameweekPlan:
        violations: list[str] = []

        if plan.season != self.season:
            violations.append(
                f"plan season {plan.season} does not match context season {self.season}"
            )

        if plan.gameweek != self.target_gameweek.id:
            violations.append(
                f"plan gameweek {plan.gameweek} does not match "
                f"target gameweek {self.target_gameweek.id}"
            )

        expected_bench_size = self.rules.squad_size - self.rules.starting_size

        if len(plan.squad) != self.rules.squad_size:
            violations.append(
                f"squad must contain {self.rules.squad_size} players; received {len(plan.squad)}"
            )

        if len(plan.starting_xi) != self.rules.starting_size:
            violations.append(
                f"starting XI must contain "
                f"{self.rules.starting_size} players; "
                f"received {len(plan.starting_xi)}"
            )

        if len(plan.bench) != expected_bench_size:
            violations.append(
                f"bench must contain {expected_bench_size} players; received {len(plan.bench)}"
            )

        player_by_id = {player.id: player for player in self.players}
        unknown_ids = sorted(set(plan.squad) - player_by_id.keys())

        if unknown_ids:
            formatted_ids = ", ".join(str(player_id) for player_id in unknown_ids)
            violations.append(f"unknown player IDs: {formatted_ids}")
        else:
            squad_players = [player_by_id[player_id] for player_id in plan.squad]
            starting_players = [player_by_id[player_id] for player_id in plan.starting_xi]

            unselectable_ids = sorted(
                player.id for player in squad_players if not player.can_select
            )
            if unselectable_ids:
                formatted_ids = ", ".join(str(player_id) for player_id in unselectable_ids)
                violations.append(f"unselectable player IDs: {formatted_ids}")

            squad_cost = sum(player.price_tenths for player in squad_players)
            if squad_cost > self.rules.budget_tenths:
                violations.append(
                    f"squad costs {squad_cost} but budget is {self.rules.budget_tenths}"
                )

            team_name_by_id = {team.id: team.name for team in self.teams}
            team_counts = Counter(player.team_id for player in squad_players)

            for team_id, player_count in sorted(team_counts.items()):
                if player_count > self.rules.max_players_per_team:
                    team_name = team_name_by_id.get(
                        team_id,
                        f"team {team_id}",
                    )
                    violations.append(
                        f"{team_name} has {player_count} players; "
                        f"maximum is "
                        f"{self.rules.max_players_per_team}"
                    )

            squad_position_counts = Counter(player.position for player in squad_players)
            starting_position_counts = Counter(player.position for player in starting_players)

            for rule in self.rules.positions:
                squad_count = squad_position_counts[rule.position]
                if squad_count != rule.squad_count:
                    violations.append(
                        f"squad must contain {rule.squad_count} "
                        f"{rule.position.value} players; "
                        f"received {squad_count}"
                    )

                starting_count = starting_position_counts[rule.position]
                if not (rule.minimum_starters <= starting_count <= rule.maximum_starters):
                    violations.append(
                        f"starting XI must contain between "
                        f"{rule.minimum_starters} and "
                        f"{rule.maximum_starters} "
                        f"{rule.position.value} players; "
                        f"received {starting_count}"
                    )

        if violations:
            raise InvalidPlanError(violations)

        return plan


class _RawEvent(BaseModel):
    id: int
    name: str
    deadline_time: datetime
    is_next: bool


class _RawTeam(BaseModel):
    id: int
    name: str
    short_name: str


class _RawPositionRule(BaseModel):
    id: int
    singular_name_short: Position
    squad_select: int
    squad_min_play: int
    squad_max_play: int


class _RawGameSettings(BaseModel):
    squad_squadsize: int
    squad_squadplay: int
    squad_team_limit: int
    squad_total_spend: int


class _RawGameConfigSettings(BaseModel):
    static_content_url: str


class _RawGameConfig(BaseModel):
    settings: _RawGameConfigSettings


class _RawPlayer(BaseModel):
    id: int
    web_name: str
    team: int
    element_type: int
    now_cost: int
    status: str
    chance_of_playing_next_round: int | None
    news: str
    can_select: bool
    total_points: int
    minutes: int
    starts: int
    form: float
    points_per_game: float
    selected_by_percent: float
    ep_next: float | None
    expected_goals: float
    expected_assists: float
    expected_goal_involvements: float
    transfers_in_event: int
    transfers_out_event: int


class _RawBootstrap(BaseModel):
    events: list[_RawEvent]
    teams: list[_RawTeam]
    element_types: list[_RawPositionRule]
    game_settings: _RawGameSettings
    game_config: _RawGameConfig
    elements: list[_RawPlayer]


class _RawFixture(BaseModel):
    id: int
    event: int | None
    kickoff_time: datetime | None
    team_h: int
    team_a: int
    team_h_difficulty: int | None
    team_a_difficulty: int | None
    started: bool
    finished: bool


_fixture_adapter = TypeAdapter(list[_RawFixture])


def fetch_planning_context() -> PlanningContext:
    """Fetch and normalize the official information needed for one agent run."""

    try:
        with httpx.Client(
            timeout=_TIMEOUT_SECONDS,
            headers={"User-Agent": "project-ted/0.1"},
        ) as client:
            bootstrap_response = client.get(_BOOTSTRAP_URL)
            bootstrap_response.raise_for_status()

            fixtures_response = client.get(_FIXTURES_URL)
            fixtures_response.raise_for_status()

        bootstrap = _RawBootstrap.model_validate(bootstrap_response.json())
        fixtures = _fixture_adapter.validate_python(fixtures_response.json())

        return _build_context(
            bootstrap,
            fixtures,
            fetched_at=datetime.now(UTC),
        )
    except (
        httpx.HTTPError,
        ValidationError,
        ValueError,
        KeyError,
    ) as error:
        raise FplDataError("Could not load current FPL data") from error


def _build_context(
    bootstrap: _RawBootstrap,
    fixtures: list[_RawFixture],
    *,
    fetched_at: datetime,
) -> PlanningContext:
    next_events = [event for event in bootstrap.events if event.is_next]

    if len(next_events) != 1:
        raise ValueError("FPL must identify exactly one next gameweek")

    season = _extract_season(bootstrap.game_config.settings.static_content_url)
    rules = season_policy_for(season)
    _verify_bootstrap_rules(bootstrap, rules)

    target_event = next_events[0]
    position_by_id = {
        position.id: position.singular_name_short for position in bootstrap.element_types
    }

    return PlanningContext(
        fetched_at=fetched_at,
        season=season,
        target_gameweek=Gameweek(
            id=target_event.id,
            name=target_event.name,
            deadline_at=target_event.deadline_time,
        ),
        rules=rules,
        teams=tuple(
            Team(
                id=team.id,
                name=team.name,
                short_name=team.short_name,
            )
            for team in bootstrap.teams
        ),
        players=tuple(
            Player(
                id=player.id,
                name=player.web_name,
                team_id=player.team,
                position=position_by_id[player.element_type],
                price_tenths=player.now_cost,
                status=player.status,
                chance_of_playing_next_round=(player.chance_of_playing_next_round),
                news=player.news,
                can_select=player.can_select,
                total_points=player.total_points,
                minutes=player.minutes,
                starts=player.starts,
                form=player.form,
                points_per_game=player.points_per_game,
                selected_by_percent=player.selected_by_percent,
                expected_points_next=player.ep_next,
                expected_goals=player.expected_goals,
                expected_assists=player.expected_assists,
                expected_goal_involvements=(player.expected_goal_involvements),
                transfers_in_event=player.transfers_in_event,
                transfers_out_event=player.transfers_out_event,
            )
            for player in bootstrap.elements
        ),
        fixtures=tuple(
            Fixture(
                id=fixture.id,
                gameweek=fixture.event,
                kickoff_at=fixture.kickoff_time,
                home_team_id=fixture.team_h,
                away_team_id=fixture.team_a,
                home_difficulty=fixture.team_h_difficulty,
                away_difficulty=fixture.team_a_difficulty,
                started=fixture.started,
                finished=fixture.finished,
            )
            for fixture in fixtures
        ),
    )


def _verify_bootstrap_rules(
    bootstrap: _RawBootstrap,
    rules: SeasonPolicy,
) -> None:
    """Reject live FPL settings that disagree with the verified policy."""

    settings = bootstrap.game_settings
    live_squad_rules = (
        settings.squad_squadsize,
        settings.squad_squadplay,
        settings.squad_team_limit,
        settings.squad_total_spend,
    )
    verified_squad_rules = (
        rules.squad_size,
        rules.starting_size,
        rules.max_players_per_team,
        rules.budget_tenths,
    )

    live_position_rules = {
        position.singular_name_short: (
            position.squad_select,
            position.squad_min_play,
            position.squad_max_play,
        )
        for position in bootstrap.element_types
    }
    verified_position_rules = {
        position.position: (
            position.squad_count,
            position.minimum_starters,
            position.maximum_starters,
        )
        for position in rules.positions
    }

    has_duplicate_live_positions = len(live_position_rules) != len(bootstrap.element_types)

    if (
        live_squad_rules != verified_squad_rules
        or live_position_rules != verified_position_rules
        or has_duplicate_live_positions
    ):
        raise ValueError(
            f"FPL bootstrap rules do not match the verified {rules.season} season policy"
        )


def _extract_season(static_content_url: str) -> str:
    # Bootstrap has no season field; its official static URL contains the
    # canonical season identifier, such as "2026_27".
    match = re.search(
        r"/(\d{4})_(\d{2})/?$",
        static_content_url,
    )

    if match is None:
        raise ValueError("FPL static content URL does not identify the season")

    starting_year, ending_year = match.groups()
    return f"{starting_year}/{ending_year}"
