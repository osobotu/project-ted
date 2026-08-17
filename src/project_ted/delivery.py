"""Deliver weekly planning reports through the Resend email API."""

import os
from base64 import b64encode

import httpx
from pydantic import BaseModel, Field

from project_ted.planning import WeeklyRun
from project_ted.report import RenderedReport

_RESEND_EMAIL_URL = "https://api.resend.com/emails"
_TIMEOUT_SECONDS = 10.0


class EmailDeliveryError(RuntimeError):
    """Report that a weekly email could not be configured or delivered."""


class _ResendResponse(BaseModel):
    id: str = Field(min_length=1)


def send_weekly_report(
    report: RenderedReport,
    run: WeeklyRun,
) -> str:
    """Email all report representations and return Resend's identifier."""

    representations = (
        report.markdown,
        report.text,
        report.html,
    )

    if any(not representation.strip() for representation in representations):
        raise ValueError("report representations must not be empty")

    api_key = _required_environment("RESEND_API_KEY")
    sender = _required_environment("PROJECT_TED_EMAIL_FROM")
    recipient = _required_environment("PROJECT_TED_EMAIL_TO")

    subject = f"Project Ted — Gameweek {run.gameweek} — {run.status.value.capitalize()}"
    markdown_content = b64encode(report.markdown.encode("utf-8")).decode("ascii")

    try:
        response = httpx.post(
            _RESEND_EMAIL_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Idempotency-Key": f"project-ted-weekly-{run.run_id}",
                "User-Agent": "project-ted/0.1",
            },
            json={
                "from": sender,
                "to": [recipient],
                "subject": subject,
                "text": report.text,
                "html": report.html,
                "attachments": [
                    {
                        "filename": (f"project-ted-gameweek-{run.gameweek}.md"),
                        "content": markdown_content,
                    }
                ],
            },
            timeout=_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise EmailDeliveryError("Could not send weekly report") from error

    try:
        result = _ResendResponse.model_validate(response.json())
    except ValueError as error:
        raise EmailDeliveryError("Resend returned an invalid response") from error

    return result.id


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()

    if not value:
        raise EmailDeliveryError(f"{name} is not configured")

    return value
