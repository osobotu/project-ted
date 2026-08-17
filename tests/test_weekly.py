from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import cast

import langsmith as ls
import pytest
from langchain_core.language_models.chat_models import BaseChatModel

import project_ted.weekly as weekly_module
from project_ted.agent import AgentPlanningError
from project_ted.fpl import (
    Gameweek,
    PlanningContext,
    SeasonRules,
)
from project_ted.news import FootballNewsSearch
from project_ted.planning import (
    AgentProvider,
    GameweekPlan,
    RunStatus,
)
from project_ted.providers import ProviderModel
from project_ted.weekly import run_weekly_planning


class FakeTrace:
    """Capture trace information without contacting LangSmith."""

    def __init__(
        self,
        name: str,
        options: dict[str, object],
    ) -> None:
        self.name = name
        self.options = options
        self.outputs: dict[str, object] | None = None

    def end(
        self,
        *,
        outputs: dict[str, object],
    ) -> None:
        self.outputs = outputs


@pytest.fixture
def traces(
    monkeypatch: pytest.MonkeyPatch,
) -> list[FakeTrace]:
    captured: list[FakeTrace] = []

    @contextmanager
    def fake_trace(
        name: str,
        *args: object,
        **kwargs: object,
    ) -> Iterator[FakeTrace]:
        trace = FakeTrace(name, kwargs)
        captured.append(trace)
        yield trace

    monkeypatch.setattr(
        ls,
        "trace",
        fake_trace,
    )

    return captured


def planning_context() -> PlanningContext:
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
            positions=(),
        ),
        teams=(),
        players=(),
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
        rationale="A valid test plan.",
    )


def install_provider_models(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ProviderModel, ProviderModel]:
    openai_model = ProviderModel(
        provider=AgentProvider.OPENAI,
        model_name="gpt-test",
        chat_model=cast(BaseChatModel, object()),
    )
    anthropic_model = ProviderModel(
        provider=AgentProvider.ANTHROPIC,
        model_name="claude-test",
        chat_model=cast(BaseChatModel, object()),
    )

    monkeypatch.setattr(
        weekly_module,
        "create_openai_model",
        lambda: openai_model,
    )
    monkeypatch.setattr(
        weekly_module,
        "create_anthropic_model",
        lambda: anthropic_model,
    )

    return openai_model, anthropic_model


def test_returns_both_successful_provider_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    traces: list[FakeTrace],
) -> None:
    openai_model, anthropic_model = install_provider_models(monkeypatch)
    calls: list[BaseChatModel] = []

    def fake_plan_gameweek(
        model: BaseChatModel,
        context: PlanningContext,
        news: FootballNewsSearch,
    ) -> GameweekPlan:
        calls.append(model)
        return valid_plan()

    monkeypatch.setattr(
        weekly_module,
        "plan_gameweek",
        fake_plan_gameweek,
    )

    result = run_weekly_planning(
        planning_context(),
        cast(FootballNewsSearch, object()),
    )

    assert result.status is RunStatus.SUCCEEDED
    assert result.season == "2026/27"
    assert result.gameweek == 1
    assert result.created_at.utcoffset() is not None

    assert calls == [
        openai_model.chat_model,
        anthropic_model.chat_model,
    ]

    assert [outcome.provider for outcome in result.outcomes] == [
        AgentProvider.OPENAI,
        AgentProvider.ANTHROPIC,
    ]
    assert [outcome.model for outcome in result.outcomes] == [
        "gpt-test",
        "claude-test",
    ]
    assert all(outcome.plan == valid_plan() for outcome in result.outcomes)

    assert [trace.name for trace in traces] == [
        "weekly-fpl-planning",
        "openai-planning",
        "anthropic-planning",
    ]
    assert traces[0].options["run_id"] == result.run_id
    assert traces[0].outputs == result.model_dump(mode="json")


@pytest.mark.parametrize(
    "failing_provider",
    [
        AgentProvider.OPENAI,
        AgentProvider.ANTHROPIC,
    ],
)
def test_preserves_one_plan_when_the_other_provider_fails(
    monkeypatch: pytest.MonkeyPatch,
    traces: list[FakeTrace],
    failing_provider: AgentProvider,
) -> None:
    openai_model, anthropic_model = install_provider_models(monkeypatch)
    model_by_provider = {
        AgentProvider.OPENAI: openai_model,
        AgentProvider.ANTHROPIC: anthropic_model,
    }
    failing_model = model_by_provider[failing_provider]
    calls: list[BaseChatModel] = []

    def fake_plan_gameweek(
        model: BaseChatModel,
        context: PlanningContext,
        news: FootballNewsSearch,
    ) -> GameweekPlan:
        calls.append(model)

        if model is failing_model.chat_model:
            raise AgentPlanningError("provider failed")

        return valid_plan()

    monkeypatch.setattr(
        weekly_module,
        "plan_gameweek",
        fake_plan_gameweek,
    )

    result = run_weekly_planning(
        planning_context(),
        cast(FootballNewsSearch, object()),
    )

    assert result.status is RunStatus.PARTIAL
    assert calls == [
        openai_model.chat_model,
        anthropic_model.chat_model,
    ]

    outcome_by_provider = {outcome.provider: outcome for outcome in result.outcomes}
    failed_outcome = outcome_by_provider[failing_provider]
    successful_provider = (
        AgentProvider.ANTHROPIC
        if failing_provider is AgentProvider.OPENAI
        else AgentProvider.OPENAI
    )
    successful_outcome = outcome_by_provider[successful_provider]

    assert failed_outcome.plan is None
    assert failed_outcome.error == "AgentPlanningError: provider failed"
    assert successful_outcome.plan == valid_plan()
    assert successful_outcome.error is None

    assert [trace.name for trace in traces] == [
        "weekly-fpl-planning",
        "openai-planning",
        "anthropic-planning",
    ]
