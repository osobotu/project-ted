import json
from datetime import UTC, datetime
from typing import cast
from uuid import UUID

import httpx
import pytest
import respx

from project_ted.delivery import (
    EmailDeliveryError,
    send_weekly_report,
)
from project_ted.planning import (
    AgentOutcome,
    AgentProvider,
    WeeklyRun,
)

RESEND_EMAIL_URL = "https://api.resend.com/emails"


def weekly_run() -> WeeklyRun:
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
                error="AgentPlanningError: provider failed",
            ),
            AgentOutcome(
                provider=AgentProvider.ANTHROPIC,
                model="claude-test",
                error="AgentPlanningError: provider failed",
            ),
        ),
    )


def configure_email(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "RESEND_API_KEY",
        "re_test_key",
    )
    monkeypatch.setenv(
        "PROJECT_TED_EMAIL_FROM",
        "Project Ted <onboarding@resend.dev>",
    )
    monkeypatch.setenv(
        "PROJECT_TED_EMAIL_TO",
        "owner@example.com",
    )


def test_sends_the_report_and_returns_the_email_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_email(monkeypatch)
    run = weekly_run()

    with respx.mock:
        route = respx.post(RESEND_EMAIL_URL).mock(
            return_value=httpx.Response(
                200,
                json={"id": "email_123"},
            )
        )

        email_id = send_weekly_report(
            "# Project Ted\n\nWeekly report.",
            run,
        )

    assert email_id == "email_123"
    assert route.called

    request = route.calls[0].request
    payload = cast(
        dict[str, object],
        json.loads(request.content),
    )

    assert request.headers["Authorization"] == ("Bearer re_test_key")
    assert request.headers["Idempotency-Key"] == (
        "project-ted-weekly-12345678-1234-5678-1234-567812345678"
    )
    assert payload == {
        "from": "Project Ted <onboarding@resend.dev>",
        "to": ["owner@example.com"],
        "subject": "Project Ted — Gameweek 1 — Failed",
        "text": "# Project Ted\n\nWeekly report.",
    }


@pytest.mark.parametrize(
    "variable",
    [
        "RESEND_API_KEY",
        "PROJECT_TED_EMAIL_FROM",
        "PROJECT_TED_EMAIL_TO",
    ],
)
def test_requires_email_configuration(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
) -> None:
    configure_email(monkeypatch)
    monkeypatch.delenv(variable)

    with pytest.raises(
        EmailDeliveryError,
        match=f"{variable} is not configured",
    ):
        send_weekly_report(
            "# Project Ted",
            weekly_run(),
        )


def test_rejects_an_empty_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_email(monkeypatch)

    with pytest.raises(
        ValueError,
        match="report must not be empty",
    ):
        send_weekly_report(
            "   ",
            weekly_run(),
        )


def test_hides_resend_http_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_email(monkeypatch)

    with respx.mock:
        respx.post(RESEND_EMAIL_URL).mock(
            return_value=httpx.Response(
                422,
                json={"message": "invalid sender"},
            )
        )

        with pytest.raises(
            EmailDeliveryError,
            match="Could not send weekly report",
        ):
            send_weekly_report(
                "# Project Ted",
                weekly_run(),
            )


def test_rejects_an_invalid_resend_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_email(monkeypatch)

    with respx.mock:
        respx.post(RESEND_EMAIL_URL).mock(
            return_value=httpx.Response(
                200,
                json={},
            )
        )

        with pytest.raises(
            EmailDeliveryError,
            match="Resend returned an invalid response",
        ):
            send_weekly_report(
                "# Project Ted",
                weekly_run(),
            )
