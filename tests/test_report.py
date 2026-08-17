from datetime import UTC, datetime
from uuid import UUID

import pytest

from project_ted.fpl import (
    Gameweek,
    PlanningContext,
    Player,
    Position,
    SeasonRules,
    Team,
)
from project_ted.planning import (
    AgentOutcome,
    AgentProvider,
    GameweekPlan,
    WeeklyRun,
)
from project_ted.report import render_weekly_report


def make_player(
    player_id: int,
    name: str,
    team_id: int,
    position: Position,
    price_tenths: int,
) -> Player:
    return Player(
        id=player_id,
        name=name,
        team_id=team_id,
        position=position,
        price_tenths=price_tenths,
        status="a",
        chance_of_playing_next_round=None,
        news="",
        can_select=True,
        total_points=0,
        minutes=0,
        starts=0,
        form=0.0,
        points_per_game=0.0,
        selected_by_percent=0.0,
        expected_points_next=None,
        expected_goals=0.0,
        expected_assists=0.0,
        expected_goal_involvements=0.0,
        transfers_in_event=0,
        transfers_out_event=0,
    )


def planning_context() -> PlanningContext:
    return PlanningContext(
        fetched_at=datetime(2026, 8, 17, 12, tzinfo=UTC),
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
            squad_size=4,
            starting_size=3,
            max_players_per_team=3,
            budget_tenths=1000,
            positions=(),
        ),
        teams=(
            Team(id=1, name="Alpha FC", short_name="AAA"),
            Team(id=2, name="Beta FC", short_name="BBB"),
        ),
        players=(
            make_player(
                1,
                "Goalkeeper",
                1,
                Position.GOALKEEPER,
                45,
            ),
            make_player(
                2,
                "Defender",
                2,
                Position.DEFENDER,
                50,
            ),
            make_player(
                3,
                "Midfielder",
                1,
                Position.MIDFIELDER,
                75,
            ),
            make_player(
                4,
                "Forward A",
                2,
                Position.FORWARD,
                70,
            ),
            make_player(
                5,
                "Forward B",
                1,
                Position.FORWARD,
                65,
            ),
        ),
        fixtures=(),
    )


def make_plan(
    *,
    forward_id: int,
    rationale: str,
    risks: tuple[str, ...],
) -> GameweekPlan:
    return GameweekPlan(
        season="2026/27",
        gameweek=1,
        squad=(1, 2, 3, forward_id),
        starting_xi=(1, 2, 3),
        bench=(forward_id,),
        captain_id=3,
        vice_captain_id=2,
        rationale=rationale,
        risks=risks,
    )


def successful_run() -> WeeklyRun:
    return WeeklyRun(
        run_id=UUID("12345678-1234-5678-1234-567812345678"),
        season="2026/27",
        gameweek=1,
        created_at=datetime(2026, 8, 17, 13, tzinfo=UTC),
        deadline_at=datetime(
            2026,
            8,
            21,
            17,
            30,
            tzinfo=UTC,
        ),
        outcomes=(
            AgentOutcome(
                provider=AgentProvider.OPENAI,
                model="gpt-test",
                plan=make_plan(
                    forward_id=4,
                    rationale="Prioritize midfield captaincy.",
                    risks=("Forward A has uncertain minutes.",),
                ),
            ),
            AgentOutcome(
                provider=AgentProvider.ANTHROPIC,
                model="claude-test",
                plan=make_plan(
                    forward_id=5,
                    rationale="Balance price and expected minutes.",
                    risks=(),
                ),
            ),
        ),
    )


def test_renders_both_plans_as_a_human_readable_report() -> None:
    report = render_weekly_report(
        successful_run(),
        planning_context(),
    )

    assert report.startswith("# Project Ted — Gameweek 1")
    assert "**Season:** 2026/27" in report
    assert "**Deadline:** 2026-08-21 17:30 UTC" in report
    assert "**Run status:** Succeeded" in report

    assert "## OpenAI — gpt-test" in report
    assert "## Anthropic — claude-test" in report
    assert "### Starting XI" in report
    assert "| Midfielder (C) | AAA | MID | £7.5m |" in report
    assert "| Defender (VC) | BBB | DEF | £5.0m |" in report
    assert "| 1 | Forward A | BBB | FWD | £7.0m |" in report
    assert "**Squad cost:** £24.0m" in report

    assert "Prioritize midfield captaincy." in report
    assert "- Forward A has uncertain minutes." in report
    assert "- No additional risks supplied." in report

    assert "## Agent comparison" in report
    assert "**Shared squad picks:** 3 of 4" in report
    assert "**OpenAI only:** Forward A" in report
    assert "**Anthropic only:** Forward B" in report


def test_renders_a_provider_failure_without_losing_the_other_plan() -> None:
    successful = successful_run()
    partial = successful.model_copy(
        update={
            "outcomes": (
                successful.outcomes[0],
                AgentOutcome(
                    provider=AgentProvider.ANTHROPIC,
                    model="claude-test",
                    error="AgentPlanningError: provider failed",
                ),
            ),
        }
    )

    report = render_weekly_report(
        partial,
        planning_context(),
    )

    assert "**Run status:** Partial" in report
    assert "## OpenAI — gpt-test" in report
    assert "Prioritize midfield captaincy." in report
    assert "## Anthropic — claude-test" in report
    assert "**Status:** Failed" in report
    assert "**Error:** AgentPlanningError: provider failed" in report
    assert "## Agent comparison" not in report


def test_rejects_a_context_from_another_season() -> None:
    context = planning_context().model_copy(update={"season": "2025/26"})

    with pytest.raises(
        ValueError,
        match="weekly run and planning context must match",
    ):
        render_weekly_report(
            successful_run(),
            context,
        )


def test_rejects_unknown_player_ids() -> None:
    run = successful_run()
    unknown_plan = make_plan(
        forward_id=999,
        rationale="An invalid report input.",
        risks=(),
    )
    invalid_run = run.model_copy(
        update={
            "outcomes": (
                AgentOutcome(
                    provider=AgentProvider.OPENAI,
                    model="gpt-test",
                    plan=unknown_plan,
                ),
                run.outcomes[1],
            ),
        }
    )

    with pytest.raises(
        ValueError,
        match="report cannot resolve player IDs: 999",
    ):
        render_weekly_report(
            invalid_run,
            planning_context(),
        )
