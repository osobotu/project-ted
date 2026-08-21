from datetime import UTC, datetime
from uuid import UUID

import pytest

from project_ted.fpl import (
    Gameweek,
    PlanningContext,
    Player,
    Team,
)
from project_ted.planning import (
    AgentOutcome,
    AgentProvider,
    GameweekPlan,
    WeeklyRun,
)
from project_ted.report import render_weekly_report
from project_ted.strategy import Position, PositionRule, SeasonPolicy, season_policy_for


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


def report_policy() -> SeasonPolicy:
    policy_data: dict[str, object] = season_policy_for("2026/27").model_dump()

    policy_data.update(
        {
            "squad_size": 4,
            "starting_size": 3,
            "max_players_per_team": 3,
            "positions": (
                PositionRule(
                    position=Position.GOALKEEPER,
                    squad_count=1,
                    minimum_starters=1,
                    maximum_starters=1,
                ),
                PositionRule(
                    position=Position.DEFENDER,
                    squad_count=1,
                    minimum_starters=1,
                    maximum_starters=1,
                ),
                PositionRule(
                    position=Position.MIDFIELDER,
                    squad_count=1,
                    minimum_starters=1,
                    maximum_starters=1,
                ),
                PositionRule(
                    position=Position.FORWARD,
                    squad_count=1,
                    minimum_starters=0,
                    maximum_starters=1,
                ),
            ),
        }
    )

    return SeasonPolicy.model_validate(policy_data)


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
        rules=report_policy(),
        teams=(
            Team(
                id=1,
                name="Alpha FC",
                short_name="AAA",
            ),
            Team(
                id=2,
                name="Beta FC",
                short_name="BBB",
            ),
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
        created_at=datetime(
            2026,
            8,
            17,
            13,
            tzinfo=UTC,
        ),
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
                    rationale=("Prioritize midfield captaincy."),
                    risks=("Forward A has uncertain minutes.",),
                ),
            ),
            AgentOutcome(
                provider=AgentProvider.ANTHROPIC,
                model="claude-test",
                plan=make_plan(
                    forward_id=5,
                    rationale=("Balance price and expected minutes."),
                    risks=(),
                ),
            ),
        ),
    )


def test_renders_both_plans_as_markdown() -> None:
    report = render_weekly_report(
        successful_run(),
        planning_context(),
    )
    markdown = report.markdown

    assert markdown.startswith("# Project Ted — Gameweek 1")
    assert "**Season:** 2026/27" in markdown
    assert "**Deadline:** 2026-08-21 17:30 UTC" in markdown
    assert "**Run status:** Succeeded" in markdown

    assert "## OpenAI — gpt-test" in markdown
    assert "## Anthropic — claude-test" in markdown
    assert "### Starting XI" in markdown
    assert "| Midfielder (C) | AAA | MID | £7.5m |" in markdown
    assert "| Defender (VC) | BBB | DEF | £5.0m |" in markdown
    assert "| 1 | Forward A | BBB | FWD | £7.0m |" in markdown
    assert "**Squad cost:** £24.0m" in markdown

    assert "Prioritize midfield captaincy." in markdown
    assert "- Forward A has uncertain minutes." in markdown
    assert "- No additional risks supplied." in markdown

    assert "## Agent comparison" in markdown
    assert "**Shared squad picks:** 3 of 4" in markdown
    assert "**OpenAI only:** Forward A" in markdown
    assert "**Anthropic only:** Forward B" in markdown


def test_renders_plain_text_and_html() -> None:
    report = render_weekly_report(
        successful_run(),
        planning_context(),
    )

    assert "PROJECT TED — GAMEWEEK 1" in report.text
    assert "OPENAI — gpt-test" in report.text
    assert "Midfielder (C) — AAA — £7.5m" in report.text
    assert "BENCH" in report.text
    assert "AGENT COMPARISON" in report.text

    assert report.html.startswith("<!doctype html>")
    assert "<title>Project Ted — Gameweek 1</title>" in report.html
    assert "Starting XI" in report.html
    assert "Midfielder" in report.html
    assert "Squad cost: £24.0m" in report.html
    assert "Agent comparison" in report.html


def test_renders_a_provider_failure_without_losing_other_plan() -> None:
    run = successful_run()
    partial = run.model_copy(
        update={
            "outcomes": (
                run.outcomes[0],
                AgentOutcome(
                    provider=(AgentProvider.ANTHROPIC),
                    model="claude-test",
                    error=("AgentPlanningError: provider failed"),
                ),
            ),
        }
    )

    report = render_weekly_report(
        partial,
        planning_context(),
    )

    assert "**Run status:** Partial" in report.markdown
    assert "## OpenAI — gpt-test" in report.markdown
    assert "Prioritize midfield captaincy." in report.markdown
    assert "## Anthropic — claude-test" in report.markdown
    assert "**Status:** Failed" in report.markdown
    assert "**Error:** AgentPlanningError: provider failed" in report.markdown
    assert "## Agent comparison" not in report.markdown

    assert "Status: Failed" in report.text
    assert "AgentPlanningError: provider failed" in report.text
    assert "AGENT COMPARISON" not in report.text

    assert "AgentPlanningError: provider failed" in report.html
    assert "Agent comparison" not in report.html


def test_html_escapes_external_content() -> None:
    run = successful_run()
    unsafe_plan = run.outcomes[0].plan

    assert unsafe_plan is not None

    unsafe_plan = unsafe_plan.model_copy(
        update={
            "rationale": ("<script>unsafe()</script>"),
            "risks": ("Minutes & fitness <unknown>",),
        }
    )
    unsafe_run = run.model_copy(
        update={
            "outcomes": (
                run.outcomes[0].model_copy(update={"plan": unsafe_plan}),
                run.outcomes[1],
            )
        }
    )

    report = render_weekly_report(
        unsafe_run,
        planning_context(),
    )

    assert "<script>unsafe()</script>" not in report.html
    assert "&lt;script&gt;unsafe()&lt;/script&gt;" in report.html
    assert "Minutes &amp; fitness &lt;unknown&gt;" in report.html


def test_rejects_a_context_from_another_season() -> None:
    context = planning_context().model_copy(update={"season": "2025/26"})

    with pytest.raises(
        ValueError,
        match=("weekly run and planning context must match"),
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
        match=("report cannot resolve player IDs: 999"),
    ):
        render_weekly_report(
            invalid_run,
            planning_context(),
        )
