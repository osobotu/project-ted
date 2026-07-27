from datetime import UTC, datetime, timedelta, timezone

import httpx
import pytest

from project_ted.data.fpl_client import (
    FPL_API_BASE_URL,
    FPLClient,
    UnexpectedContentTypeError,
    utc_now,
)


def test_utc_now_returns_a_utc_timestamp() -> None:
    assert utc_now().utcoffset() == timedelta(0)


def test_get_bootstrap_returns_the_exact_response_bytes() -> None:
    payload = b'{"events": [], "elements": []}'
    retrieved_at = datetime(2026, 7, 27, 12, tzinfo=UTC)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url) == ("https://fantasy.premierleague.com/api/bootstrap-static/")
        assert request.headers["accept"] == "application/json"

        return httpx.Response(
            200,
            content=payload,
            headers={"Content-Type": "application/json; charset=utf-8"},
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(
        base_url=FPL_API_BASE_URL,
        transport=transport,
    ) as http_client:
        response = FPLClient(
            http_client,
            clock=lambda: retrieved_at,
        ).get_bootstrap()

    assert response.payload == payload
    assert response.retrieved_at == retrieved_at
    assert str(response.source_url) == ("https://fantasy.premierleague.com/api/bootstrap-static/")


def test_get_bootstrap_raises_for_an_http_error() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            503,
            json={"detail": "service unavailable"},
        )
    )

    with (
        httpx.Client(
            base_url=FPL_API_BASE_URL,
            transport=transport,
        ) as http_client,
        pytest.raises(httpx.HTTPStatusError),
    ):
        FPLClient(http_client).get_bootstrap()


def test_get_bootstrap_rejects_a_non_json_response() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            content=b"<html>maintenance</html>",
            headers={"Content-Type": "text/html"},
        )
    )

    with (
        httpx.Client(
            base_url=FPL_API_BASE_URL,
            transport=transport,
        ) as http_client,
        pytest.raises(
            UnexpectedContentTypeError,
            match="expected application/json",
        ),
    ):
        FPLClient(http_client).get_bootstrap()


def test_get_bootstrap_rejects_a_non_utc_clock() -> None:
    kigali_time = datetime(
        2026,
        7,
        27,
        14,
        tzinfo=timezone(timedelta(hours=2)),
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={"events": []},
        )
    )

    with (
        httpx.Client(
            base_url=FPL_API_BASE_URL,
            transport=transport,
        ) as http_client,
        pytest.raises(ValueError, match="clock must return a UTC timestamp"),
    ):
        FPLClient(
            http_client,
            clock=lambda: kigali_time,
        ).get_bootstrap()
