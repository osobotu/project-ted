from typing import Annotated, Any

from langchain.agents import create_agent
from langchain.agents.middleware.types import InputAgentState
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from pydantic import Field

from project_ted.fpl import (
    Fixture,
    InvalidPlanError,
    PlanningContext,
    Player,
    Position,
    Team,
)
from project_ted.news import (
    FootballNewsSearch,
    NewsSearchError,
)
from project_ted.planning import GameweekPlan

_MAX_PLAN_ATTEMPTS = 2
_RECURSION_LIMIT = 30

_SYSTEM_PROMPT = """
You are an expert Fantasy Premier League manager.

Your objective is to maximize expected FPL points while respecting the live
season rules supplied in the task.

Requirements:
- Use the supplied tools to research players, fixtures and current news.
- Never invent player IDs, prices, teams, fixtures, injuries or news.
- Treat official FPL availability as authoritative.
- Treat external football news as evidence, not certainty.
- Consider expected minutes, fixture difficulty, form, underlying statistics,
  ownership, captaincy upside and risk.
- Return one complete GameweekPlan.
- The squad, XI, bench, captain and vice-captain must all be internally
  consistent.
- Keep the rationale concise and identify genuine risks.
""".strip()


class AgentPlanningError(RuntimeError):
    """Report that a model could not produce a valid plan."""


def plan_gameweek(
    model: BaseChatModel,
    context: PlanningContext,
    news: FootballNewsSearch,
) -> GameweekPlan:
    """Run one provider model and return a live-rule-validated plan."""

    tools = _build_tools(context, news)
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=_SYSTEM_PROMPT,
        response_format=ToolStrategy(GameweekPlan),
    )

    messages: list[AnyMessage | dict[str, Any]] = [HumanMessage(content=_planning_request(context))]
    agent_config: RunnableConfig = {
        "recursion_limit": _RECURSION_LIMIT,
    }
    last_validation_error: InvalidPlanError | None = None

    for attempt in range(_MAX_PLAN_ATTEMPTS):
        agent_input: InputAgentState = {
            "messages": messages,
        }

        try:
            state = agent.invoke(
                agent_input,
                config=agent_config,
            )
        except Exception as error:
            raise AgentPlanningError("Agent execution failed") from error

        plan = state.get("structured_response")

        if not isinstance(plan, GameweekPlan):
            raise AgentPlanningError("Agent did not return a GameweekPlan")

        try:
            return context.validate_plan(plan)
        except InvalidPlanError as error:
            last_validation_error = error

            if attempt == _MAX_PLAN_ATTEMPTS - 1:
                break

            state_messages = state.get("messages")

            if not isinstance(state_messages, list):
                raise AgentPlanningError("Agent returned invalid message state") from error

            correction = (
                "Your proposed plan violates these live FPL rules:\n- "
                + "\n- ".join(error.violations)
                + "\nRevise the plan using valid player IDs. "
                "Return the complete corrected plan."
            )
            messages = [
                *state_messages,
                HumanMessage(content=correction),
            ]

    if last_validation_error is None:
        raise AgentPlanningError("Agent failed without a validation result")

    raise AgentPlanningError(
        "Agent could not produce a valid FPL plan after correction: "
        + "; ".join(last_validation_error.violations)
    ) from last_validation_error


def _planning_request(context: PlanningContext) -> str:
    position_rules = ", ".join(
        (
            f"{rule.position.value}: {rule.squad_count} in squad, "
            f"{rule.minimum_starters}-{rule.maximum_starters} starters"
        )
        for rule in context.rules.positions
    )

    budget_millions = context.rules.budget_tenths / 10

    return f"""
Build the best available team for:

Season: {context.season}
Gameweek: {context.target_gameweek.id}
Deadline: {context.target_gameweek.deadline_at.isoformat()}
Budget: £{budget_millions:.1f}m
Squad size: {context.rules.squad_size}
Starting XI size: {context.rules.starting_size}
Maximum players per club: {context.rules.max_players_per_team}
Position rules: {position_rules}

Research before deciding. Use only player IDs returned by the tools.
""".strip()


def _build_tools(
    context: PlanningContext,
    news: FootballNewsSearch,
) -> tuple[BaseTool, ...]:
    team_by_id = {team.id: team for team in context.teams}
    player_by_id = {player.id: player for player in context.players}

    @tool
    def find_players(
        query: str = "",
        position: Position | None = None,
        team: str = "",
        max_price_tenths: int | None = None,
        available_only: bool = True,
        limit: Annotated[int, Field(ge=1, le=20)] = 10,
    ) -> dict[str, object]:
        """Find FPL players by name, position, club, price and availability.

        Prices are integer tenths of a million: 75 means £7.5m.
        Results are ranked by expected next points, form and total points.
        """

        name_query = query.strip().casefold()
        team_query = team.strip().casefold()

        matching_team_ids = {
            candidate.id
            for candidate in context.teams
            if not team_query
            or team_query in candidate.name.casefold()
            or team_query in candidate.short_name.casefold()
        }

        matches = [
            player
            for player in context.players
            if (not name_query or name_query in player.name.casefold())
            and (position is None or player.position is position)
            and player.team_id in matching_team_ids
            and (max_price_tenths is None or player.price_tenths <= max_price_tenths)
            and (not available_only or player.can_select)
        ]
        matches.sort(
            key=_player_ranking,
            reverse=True,
        )

        selected = matches[:limit]

        return {
            "matched": len(matches),
            "returned": len(selected),
            "players": [_player_record(player, team_by_id) for player in selected],
        }

    @tool
    def compare_players(
        player_ids: Annotated[
            list[int],
            Field(min_length=2, max_length=10),
        ],
    ) -> dict[str, object]:
        """Compare between two and ten players using current FPL data."""

        unknown_ids = sorted(set(player_ids) - player_by_id.keys())

        return {
            "unknown_player_ids": unknown_ids,
            "players": [
                _player_record(
                    player_by_id[player_id],
                    team_by_id,
                )
                for player_id in player_ids
                if player_id in player_by_id
            ],
        }

    @tool
    def get_fixtures(
        team_id: int | None = None,
        gameweeks: Annotated[int, Field(ge=1, le=10)] = 5,
    ) -> dict[str, object]:
        """Return upcoming fixtures from the target gameweek.

        Optionally restrict results to one FPL team ID.
        """

        first_gameweek = context.target_gameweek.id
        last_gameweek = first_gameweek + gameweeks - 1

        fixtures = [
            fixture
            for fixture in context.fixtures
            if fixture.gameweek is not None
            and first_gameweek <= fixture.gameweek <= last_gameweek
            and (
                team_id is None
                or fixture.home_team_id == team_id
                or fixture.away_team_id == team_id
            )
        ]

        return {
            "first_gameweek": first_gameweek,
            "last_gameweek": last_gameweek,
            "fixtures": [_fixture_record(fixture, team_by_id) for fixture in fixtures],
        }

    @tool
    def get_player_news(
        player_ids: list[int] | None = None,
    ) -> dict[str, object]:
        """Return official FPL availability and news.

        With no IDs, returns every player carrying availability risk or news.
        """

        requested_ids = set(player_ids) if player_ids is not None else None

        players = [
            player
            for player in context.players
            if (requested_ids is not None and player.id in requested_ids)
            or (
                requested_ids is None
                and (
                    player.status != "a"
                    or bool(player.news)
                    or player.chance_of_playing_next_round not in (None, 100)
                )
            )
        ]

        unknown_ids = (
            sorted(requested_ids - player_by_id.keys()) if requested_ids is not None else []
        )

        return {
            "unknown_player_ids": unknown_ids,
            "players": [_player_record(player, team_by_id) for player in players],
        }

    @tool
    def search_football_news(query: str) -> dict[str, object]:
        """Search recent trusted football sources for current evidence."""

        try:
            result = news.search(query)
        except NewsSearchError as error:
            return {
                "query": query,
                "articles": [],
                "error": str(error),
            }

        return result.model_dump(mode="json")

    return (
        find_players,
        compare_players,
        get_fixtures,
        get_player_news,
        search_football_news,
    )


def _player_ranking(player: Player) -> tuple[float, float, int]:
    expected_points = (
        player.expected_points_next if player.expected_points_next is not None else -1.0
    )

    return (
        expected_points,
        player.form,
        player.total_points,
    )


def _player_record(
    player: Player,
    team_by_id: dict[int, Team],
) -> dict[str, object]:
    record: dict[str, object] = player.model_dump(mode="json")
    team = team_by_id.get(player.team_id)

    record["team_name"] = team.name if team is not None else None
    record["team_short_name"] = team.short_name if team is not None else None

    return record


def _fixture_record(
    fixture: Fixture,
    team_by_id: dict[int, Team],
) -> dict[str, object]:
    record: dict[str, object] = fixture.model_dump(mode="json")
    home_team = team_by_id.get(fixture.home_team_id)
    away_team = team_by_id.get(fixture.away_team_id)

    record["home_team_name"] = home_team.name if home_team is not None else None
    record["away_team_name"] = away_team.name if away_team is not None else None

    return record
