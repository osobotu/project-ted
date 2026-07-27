"""Raw HTTP access to the public Fantasy Premier League API."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx
from pydantic import HttpUrl

FPL_API_BASE_URL = "https://fantasy.premierleague.com/api/"
FPL_USER_AGENT = "project-ted/0.1"
FPL_CONNECT_RETRIES = 3
FPL_CONNECT_TIMEOUT_SECONDS = 10.0
FPL_REQUEST_TIMEOUT_SECONDS = 30.0

type Clock = Callable[[], datetime]


class UnexpectedContentTypeError(RuntimeError):
    """Raised when an FPL endpoint does not return JSON."""


@dataclass(frozen=True, slots=True)
class RawFPLResponse:
    """A raw FPL response with retrieval metadata."""

    source_url: HttpUrl
    retrieved_at: datetime
    payload: bytes


def utc_now() -> datetime:
    """Return the current time in UTC."""
    return datetime.now(UTC)


def create_fpl_http_client(
    *,
    transport: httpx.BaseTransport | None = None,
) -> httpx.Client:
    """Create a consistently configured HTTP client for FPL."""

    resolved_transport = transport
    if resolved_transport is None:
        resolved_transport = httpx.HTTPTransport(
            retries=FPL_CONNECT_RETRIES,
        )

    return httpx.Client(
        base_url=FPL_API_BASE_URL,
        headers={"User-Agent": FPL_USER_AGENT},
        timeout=httpx.Timeout(
            FPL_REQUEST_TIMEOUT_SECONDS,
            connect=FPL_CONNECT_TIMEOUT_SECONDS,
        ),
        transport=resolved_transport,
    )


class FPLClient:
    """Retrieve raw responses from known FPL endpoints."""

    def __init__(self, http_client: httpx.Client, *, clock: Clock = utc_now) -> None:
        self._http_client = http_client
        self._clock = clock

    def get_bootstrap(self) -> RawFPLResponse:
        """Retrieve players, teams, gameweeks, prices, and availability."""
        return self._get("bootstrap-static/")

    def get_fixtures(self) -> RawFPLResponse:
        """Retrieve all scheduled fixtures."""
        return self._get("fixtures/")

    def get_live_gameweek(self, gameweek: int) -> RawFPLResponse:
        """Retrieve player points for one gameweek."""
        self._validate_positive_id("gameweek", gameweek)
        return self._get(f"event/{gameweek}/live/")

    def get_player_summary(self, player_id: int) -> RawFPLResponse:
        """Retrieve one player's fixtures and history."""

        self._validate_positive_id("player_id", player_id)
        return self._get(f"element-summary/{player_id}/")

    def get_manager_entry(self, manager_id: int) -> RawFPLResponse:
        """Retrieve one manager's public entry information."""
        self._validate_positive_id("manager_id", manager_id)
        return self._get(f"entry/{manager_id}/")

    def get_manager_history(self, manager_id: int) -> RawFPLResponse:
        """Retrieve one manager's season history."""

        self._validate_positive_id("manager_id", manager_id)
        return self._get(f"entry/{manager_id}/history/")

    def get_manager_transfers(self, manager_id: int) -> RawFPLResponse:
        """Retrieve one manager's complete transfer history."""

        self._validate_positive_id("manager_id", manager_id)
        return self._get(f"entry/{manager_id}/transfers/")

    def get_manager_picks(
        self,
        manager_id: int,
        gameweek: int,
    ) -> RawFPLResponse:
        """Retrieve one manager's picks for one gameweek."""

        self._validate_positive_id("manager_id", manager_id)
        self._validate_positive_id("gameweek", gameweek)
        return self._get(f"entry/{manager_id}/event/{gameweek}/picks/")

    @staticmethod
    def _validate_positive_id(name: str, value: int) -> None:
        if isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

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
