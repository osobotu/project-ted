from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from project_ted.planning import (
    AgentOutcome,
    AgentProvider,
    GameweekPlan,
    RunStatus,
    WeeklyRun,
)
from project_ted.strategy import Chip


def valid_plan_data() -> dict[str, object]:
    return {
        "season": "2026/27",
        "gameweek": 1,
        "squad": tuple(range(1, 16)),
        "starting_xi": tuple(range(1, 12)),
        "bench": (12, 13, 14, 15),
        "captain_id": 1,
        "vice_captain_id": 2,
        "rationale": "A balanced opening-gameweek squad.",
        "risks": ("Several players have uncertain minutes.",),
    }


def test_accepts_a_complete_gameweek_plan() -> None:
    plan = GameweekPlan.model_validate(valid_plan_data())

    assert len(plan.squad) == 15
    assert len(plan.starting_xi) == 11
    assert plan.bench == (12, 13, 14, 15)


def test_plan_can_select_a_chip() -> None:
    data = valid_plan_data()
    data["chip"] = Chip.TRIPLE_CAPTAIN

    plan = GameweekPlan.model_validate(data)

    assert plan.chip is Chip.TRIPLE_CAPTAIN


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("squad", (*range(1, 15), 14), "squad player IDs must be unique"),
        (
            "starting_xi",
            (*range(1, 11), 10),
            "starting_xi player IDs must be unique",
        ),
        ("bench", (12, 13, 14, 14), "bench player IDs must be unique"),
    ],
)
def test_rejects_duplicate_players(
    field: str,
    value: object,
    message: str,
) -> None:
    data = valid_plan_data()
    data[field] = value

    with pytest.raises(ValidationError, match=message):
        GameweekPlan.model_validate(data)


def test_starting_xi_must_come_from_squad() -> None:
    data = valid_plan_data()
    data["starting_xi"] = (*range(1, 11), 99)

    with pytest.raises(
        ValidationError,
        match="starting_xi must contain only squad players",
    ):
        GameweekPlan.model_validate(data)


def test_bench_must_be_the_remaining_squad() -> None:
    data = valid_plan_data()
    data["bench"] = (1, 12, 13, 14)

    with pytest.raises(
        ValidationError,
        match="bench must contain every non-starting squad player exactly once",
    ):
        GameweekPlan.model_validate(data)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("captain_id", 12, "captain must be in the starting XI"),
        ("vice_captain_id", 12, "vice-captain must be in the starting XI"),
        (
            "vice_captain_id",
            1,
            "captain and vice-captain must be different players",
        ),
    ],
)
def test_validates_captain_choices(
    field: str,
    value: object,
    message: str,
) -> None:
    data = valid_plan_data()
    data[field] = value

    with pytest.raises(ValidationError, match=message):
        GameweekPlan.model_validate(data)


def test_rejects_unknown_agent_output_fields() -> None:
    data = valid_plan_data()
    data["confidence_score"] = 0.95

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        GameweekPlan.model_validate(data)


def valid_plan() -> GameweekPlan:
    return GameweekPlan.model_validate(valid_plan_data())


def agent_outcome(
    provider: AgentProvider,
    *,
    failed: bool = False,
) -> AgentOutcome:
    if failed:
        return AgentOutcome(
            provider=provider,
            model="test-model",
            error="Provider request failed.",
        )

    return AgentOutcome(
        provider=provider,
        model="test-model",
        plan=valid_plan(),
    )


def weekly_run(
    outcomes: tuple[AgentOutcome, AgentOutcome],
) -> WeeklyRun:
    return WeeklyRun(
        run_id=UUID("b180ace4-911f-48ab-b050-b6b286dd3949"),
        season="2026/27",
        gameweek=1,
        created_at=datetime(2026, 8, 20, 8, tzinfo=UTC),
        deadline_at=datetime(2026, 8, 21, 17, 30, tzinfo=UTC),
        outcomes=outcomes,
    )


def test_agent_outcome_accepts_a_plan() -> None:
    outcome = agent_outcome(AgentProvider.OPENAI)

    assert outcome.succeeded
    assert outcome.plan is not None
    assert outcome.error is None


def test_agent_outcome_accepts_a_failure() -> None:
    outcome = agent_outcome(AgentProvider.ANTHROPIC, failed=True)

    assert not outcome.succeeded
    assert outcome.plan is None
    assert outcome.error == "Provider request failed."


@pytest.mark.parametrize(
    ("plan", "error"),
    [
        (None, None),
        (valid_plan(), "Provider request failed."),
    ],
)
def test_agent_outcome_requires_exactly_one_result(
    plan: GameweekPlan | None,
    error: str | None,
) -> None:
    with pytest.raises(
        ValidationError,
        match="must contain exactly one of plan or error",
    ):
        AgentOutcome(
            provider=AgentProvider.OPENAI,
            model="test-model",
            plan=plan,
            error=error,
        )


@pytest.mark.parametrize(
    ("openai_failed", "anthropic_failed", "expected_status"),
    [
        (False, False, RunStatus.SUCCEEDED),
        (False, True, RunStatus.PARTIAL),
        (True, True, RunStatus.FAILED),
    ],
)
def test_weekly_run_derives_its_status(
    openai_failed: bool,
    anthropic_failed: bool,
    expected_status: RunStatus,
) -> None:
    run = weekly_run(
        (
            agent_outcome(
                AgentProvider.OPENAI,
                failed=openai_failed,
            ),
            agent_outcome(
                AgentProvider.ANTHROPIC,
                failed=anthropic_failed,
            ),
        )
    )

    assert run.status is expected_status


def test_weekly_run_requires_both_providers() -> None:
    with pytest.raises(
        ValidationError,
        match="must contain one OpenAI and one Anthropic outcome",
    ):
        weekly_run(
            (
                agent_outcome(AgentProvider.OPENAI),
                agent_outcome(AgentProvider.OPENAI),
            )
        )


def test_weekly_run_rejects_a_plan_for_another_gameweek() -> None:
    data = valid_plan_data()
    data["gameweek"] = 2

    mismatched_outcome = AgentOutcome(
        provider=AgentProvider.OPENAI,
        model="test-model",
        plan=GameweekPlan.model_validate(data),
    )

    with pytest.raises(
        ValidationError,
        match="plans must match the weekly run season and gameweek",
    ):
        weekly_run(
            (
                mismatched_outcome,
                agent_outcome(AgentProvider.ANTHROPIC),
            )
        )


def test_weekly_run_requires_timezone_aware_timestamps() -> None:
    with pytest.raises(
        ValidationError,
        match="run timestamps must include a timezone",
    ):
        WeeklyRun(
            run_id=UUID("b180ace4-911f-48ab-b050-b6b286dd3949"),
            season="2026/27",
            gameweek=1,
            created_at=datetime(2026, 8, 20, 8),
            deadline_at=datetime(2026, 8, 21, 17, 30, tzinfo=UTC),
            outcomes=(
                agent_outcome(AgentProvider.OPENAI),
                agent_outcome(AgentProvider.ANTHROPIC),
            ),
        )
