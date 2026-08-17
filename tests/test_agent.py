from datetime import UTC, datetime
from typing import cast

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import HttpUrl

import project_ted.agent as agent_module
from project_ted.agent import (
    AgentPlanningError,
    plan_gameweek,
)
from project_ted.fpl import (
    Gameweek,
    PlanningContext,
    Player,
    Position,
    PositionRule,
    SeasonRules,
    Team,
)
from project_ted.news import (
    FootballNews,
    FootballNewsSearch,
    NewsArticle,
)
from project_ted.planning import GameweekPlan


class FakeNewsSearch:
    def search(self, query: str) -> FootballNews:
        return FootballNews(
            query=query,
            searched_at=datetime(2026, 8, 17, tzinfo=UTC),
            articles=(
                NewsArticle(
                    title="Team news",
                    url=HttpUrl("https://www.premierleague.com/news/test"),
                    summary="Both players are available.",
                    relevance=0.9,
                ),
            ),
        )


class FakeAgent:
    def __init__(
        self,
        states: list[dict[str, object]],
    ) -> None:
        self.states = states
        self.calls: list[dict[str, object]] = []

    def invoke(
        self,
        inputs: dict[str, object],
        config: dict[str, object],
    ) -> dict[str, object]:
        self.calls.append(inputs)
        return self.states[len(self.calls) - 1]


def small_context() -> PlanningContext:
    players = tuple(
        Player(
            id=player_id,
            name=name,
            team_id=player_id,
            position=Position.MIDFIELDER,
            price_tenths=50,
            status="a",
            chance_of_playing_next_round=None,
            news="",
            can_select=True,
            total_points=100 - player_id,
            minutes=3000,
            starts=34,
            form=5.0,
            points_per_game=5.0,
            selected_by_percent=10.0,
            expected_points_next=5.0,
            expected_goals=5.0,
            expected_assists=5.0,
            expected_goal_involvements=10.0,
            transfers_in_event=100,
            transfers_out_event=10,
        )
        for player_id, name in (
            (1, "Alpha"),
            (2, "Beta"),
        )
    )

    return PlanningContext(
        fetched_at=datetime(2026, 8, 17, tzinfo=UTC),
        season="2026/27",
        target_gameweek=Gameweek(
            id=1,
            name="Gameweek 1",
            deadline_at=datetime(
                2026,
                8,
                21,
                17,
                30,
                tzinfo=UTC,
            ),
        ),
        rules=SeasonRules(
            squad_size=2,
            starting_size=2,
            max_players_per_team=1,
            budget_tenths=100,
            positions=(
                PositionRule(
                    position=Position.MIDFIELDER,
                    squad_count=2,
                    minimum_starters=2,
                    maximum_starters=2,
                ),
            ),
        ),
        teams=(
            Team(id=1, name="Alpha FC", short_name="ALP"),
            Team(id=2, name="Beta FC", short_name="BET"),
        ),
        players=players,
        fixtures=(),
    )


def valid_plan() -> GameweekPlan:
    return GameweekPlan(
        season="2026/27",
        gameweek=1,
        squad=(1, 2),
        starting_xi=(1, 2),
        bench=(),
        captain_id=1,
        vice_captain_id=2,
        rationale="Alpha offers the strongest captaincy option.",
    )


def test_runs_any_chat_model_through_the_shared_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_agent = FakeAgent(
        [
            {
                "structured_response": valid_plan(),
                "messages": [HumanMessage(content="finished")],
            }
        ]
    )
    captured: dict[str, object] = {}

    def fake_create_agent(**kwargs: object) -> FakeAgent:
        captured.update(kwargs)
        return fake_agent

    monkeypatch.setattr(
        agent_module,
        "create_agent",
        fake_create_agent,
    )

    model = cast(BaseChatModel, object())
    news = cast(FootballNewsSearch, FakeNewsSearch())

    result = plan_gameweek(model, small_context(), news)

    assert result == valid_plan()
    assert captured["model"] is model

    tools = captured["tools"]
    assert isinstance(tools, tuple)
    assert {tool.name for tool in tools} == {
        "find_players",
        "compare_players",
        "get_fixtures",
        "get_player_news",
        "search_football_news",
    }


def test_repairs_a_plan_that_breaks_live_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_plan = GameweekPlan(
        season="2026/27",
        gameweek=1,
        squad=(1, 999),
        starting_xi=(1, 999),
        bench=(),
        captain_id=1,
        vice_captain_id=999,
        rationale="An invalid first attempt.",
    )
    fake_agent = FakeAgent(
        [
            {
                "structured_response": invalid_plan,
                "messages": [HumanMessage(content="first attempt")],
            },
            {
                "structured_response": valid_plan(),
                "messages": [HumanMessage(content="corrected")],
            },
        ]
    )

    monkeypatch.setattr(
        agent_module,
        "create_agent",
        lambda **kwargs: fake_agent,
    )

    result = plan_gameweek(
        cast(BaseChatModel, object()),
        small_context(),
        cast(FootballNewsSearch, FakeNewsSearch()),
    )

    assert result == valid_plan()
    assert len(fake_agent.calls) == 2

    correction_messages = fake_agent.calls[1]["messages"]
    assert isinstance(correction_messages, list)
    assert "unknown player IDs: 999" in str(correction_messages[-1].content)


def test_fails_after_one_bounded_correction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_plan = GameweekPlan(
        season="2026/27",
        gameweek=2,
        squad=(1, 2),
        starting_xi=(1, 2),
        bench=(),
        captain_id=1,
        vice_captain_id=2,
        rationale="Wrong gameweek.",
    )
    fake_agent = FakeAgent(
        [
            {
                "structured_response": invalid_plan,
                "messages": [HumanMessage(content="first")],
            },
            {
                "structured_response": invalid_plan,
                "messages": [HumanMessage(content="second")],
            },
        ]
    )

    monkeypatch.setattr(
        agent_module,
        "create_agent",
        lambda **kwargs: fake_agent,
    )

    with pytest.raises(
        AgentPlanningError,
        match="could not produce a valid FPL plan",
    ):
        plan_gameweek(
            cast(BaseChatModel, object()),
            small_context(),
            cast(FootballNewsSearch, FakeNewsSearch()),
        )

    assert len(fake_agent.calls) == 2
