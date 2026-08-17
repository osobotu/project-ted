from typing import cast

import pytest

import project_ted.app as app_module
from project_ted.app import (
    main,
    run_weekly_job,
)
from project_ted.fpl import PlanningContext
from project_ted.news import FootballNewsSearch
from project_ted.planning import WeeklyRun


def test_runs_the_complete_weekly_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = cast(PlanningContext, object())
    news = cast(FootballNewsSearch, object())
    weekly_run = cast(WeeklyRun, object())
    report = "# Project Ted\n\nWeekly report."
    events: list[str] = []

    def fake_fetch_context() -> PlanningContext:
        events.append("fetch")
        return context

    def fake_create_news() -> FootballNewsSearch:
        events.append("news")
        return news

    def fake_run_weekly(
        received_context: PlanningContext,
        received_news: FootballNewsSearch,
    ) -> WeeklyRun:
        assert received_context is context
        assert received_news is news
        events.append("plan")
        return weekly_run

    def fake_render_report(
        received_run: WeeklyRun,
        received_context: PlanningContext,
    ) -> str:
        assert received_run is weekly_run
        assert received_context is context
        events.append("report")
        return report

    def fake_send_report(
        received_report: str,
        received_run: WeeklyRun,
    ) -> str:
        assert received_report == report
        assert received_run is weekly_run
        events.append("email")
        return "email_123"

    monkeypatch.setattr(
        app_module,
        "fetch_planning_context",
        fake_fetch_context,
    )
    monkeypatch.setattr(
        app_module,
        "FootballNewsSearch",
        fake_create_news,
    )
    monkeypatch.setattr(
        app_module,
        "run_weekly_planning",
        fake_run_weekly,
    )
    monkeypatch.setattr(
        app_module,
        "render_weekly_report",
        fake_render_report,
    )
    monkeypatch.setattr(
        app_module,
        "send_weekly_report",
        fake_send_report,
    )

    email_id = run_weekly_job()

    assert email_id == "email_123"
    assert events == [
        "fetch",
        "news",
        "plan",
        "report",
        "email",
    ]


def test_main_prints_the_delivery_identifier(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_run_weekly_job() -> str:
        return "email_123"

    monkeypatch.setattr(
        app_module,
        "run_weekly_job",
        fake_run_weekly_job,
    )

    main()

    captured = capsys.readouterr()
    assert captured.out == ("Project Ted report delivered: email_123\n")
    assert captured.err == ""
