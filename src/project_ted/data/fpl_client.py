"""Raw HTTP access to the public Fantasy Premier League API"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx
from pydantic import HttpUrl

FPL_API_BASE_URL = "https://fantasy.premierleague.com/api/"

type Clock = Callable[[], datetime]


class UnexpectedContentTypeError(RuntimeError):
    """Raised when an FPL endpoint does not return JSON."""


@dataclass(frozen=True, slots=True)
class RawFPLResponse:
    """A raw FPL response with retrieval metadata"""

    source_url: HttpUrl
    retrieved_at: datetime
    payload: bytes


def utc_now() -> datetime:
    """Return the current time in UTC."""
    return datetime.now(UTC)


class FPLClient:
    """Retrieve raw responses from known FPL endpoints."""

    def __init__(self, http_client: httpx.Client, *, clock: Clock = utc_now) -> None:
        self._http_client = http_client
        self._clock = clock

    def get_bootstrap(self) -> RawFPLResponse:
        """Retrieve players, teams, gameweeks, prices, and availability."""
        return self._get("bootstrap-static/")

    def _get(self, endpoint: str) -> RawFPLResponse:
        response = self._http_client.get(
            endpoint,
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        media_type = response.headers.get("content-type", "").partition(";")[0].strip().lower()
        if media_type != "application/json":
            raise UnexpectedContentTypeError(
                f"expected application/json, received {media_type or 'no content type'}"
            )
        retrieved_at = self._clock()
        if retrieved_at.utcoffset() != timedelta(0):
            raise ValueError("clock must return a UTC timestamp")

        return RawFPLResponse(
            source_url=HttpUrl(str(response.url)),
            retrieved_at=retrieved_at,
            payload=response.content,
        )
