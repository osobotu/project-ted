"""Deliver weekly planning reports through the Resend email API."""

import os

import httpx
from pydantic import BaseModel, Field

from project_ted.planning import WeeklyRun

_RESEND_EMAIL_URL = "https://api.resend.com/emails"
_TIMEOUT_SECONDS = 10.0


class EmailDeliveryError(RuntimeError):
    """Report that a weekly email could not be configured or delivered."""


class _ResendResponse(BaseModel):
    id: str = Field(min_length=1)


def send_weekly_report(report: str, run: WeeklyRun) -> str:
    """Email a weekly report and return Resend's delivery identifier."""

    if not report.strip():
        raise ValueError("report must not be empty")

    api_key = _required_environment("RESEND_API_KEY")
    sender = _required_environment("PROJECT_TED_EMAIL_FROM")
    recipient = _required_environment("PROJECT_TED_EMAIL_TO")

    subject = f"Project Ted — Gameweek {run.gameweek} — {run.status.value.capitalize()}"

    try:
        response = httpx.post(
            _RESEND_EMAIL_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Idempotency-Key": (f"project-ted-weekly-{run.run_id}"),
                "User-Agent": "project-ted/0.1",
            },
            json={
                "from": sender,
                "to": [recipient],
                "subject": subject,
                "text": report,
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
