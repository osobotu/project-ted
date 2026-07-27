from datetime import UTC, datetime, timedelta, timezone

import httpx
import pytest

from project_ted.data.fpl_client import (
    FPL_API_BASE_URL,
    FPL_CONNECT_TIMEOUT_SECONDS,
    FPL_REQUEST_TIMEOUT_SECONDS,
    FPL_USER_AGENT,
    FPLClient,
    UnexpectedContentTypeError,
    create_fpl_http_client,
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


def test_known_endpoints_build_the_expected_paths() -> None:
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)

    with httpx.Client(
        base_url=FPL_API_BASE_URL,
        transport=transport,
    ) as http_client:
        client = FPLClient(http_client)

        client.get_fixtures()
        client.get_live_gameweek(7)
        client.get_player_summary(351)
        client.get_manager_entry(12345)
        client.get_manager_history(12345)
        client.get_manager_transfers(12345)
        client.get_manager_picks(12345, 7)

    assert requested_paths == [
        "/api/fixtures/",
        "/api/event/7/live/",
        "/api/element-summary/351/",
        "/api/entry/12345/",
        "/api/entry/12345/history/",
        "/api/entry/12345/transfers/",
        "/api/entry/12345/event/7/picks/",
    ]


def test_known_endpoints_reject_invalid_ids_before_requesting() -> None:
    def unexpected_request(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request to {request.url}")

    transport = httpx.MockTransport(unexpected_request)

    with httpx.Client(
        base_url=FPL_API_BASE_URL,
        transport=transport,
    ) as http_client:
        client = FPLClient(http_client)

        with pytest.raises(ValueError, match="gameweek must be a positive integer"):
            client.get_live_gameweek(0)

        with pytest.raises(ValueError, match="player_id must be a positive integer"):
            client.get_player_summary(0)

        with pytest.raises(ValueError, match="manager_id must be a positive integer"):
            client.get_manager_entry(0)

        with pytest.raises(ValueError, match="manager_id must be a positive integer"):
            client.get_manager_history(0)

        with pytest.raises(ValueError, match="manager_id must be a positive integer"):
            client.get_manager_transfers(0)

        with pytest.raises(ValueError, match="manager_id must be a positive integer"):
            client.get_manager_picks(0, 7)

        with pytest.raises(ValueError, match="gameweek must be a positive integer"):
            client.get_manager_picks(12345, 0)

        with pytest.raises(ValueError, match="manager_id must be a positive integer"):
            client.get_manager_entry(False)


def test_create_fpl_http_client_applies_production_configuration() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["user-agent"] == FPL_USER_AGENT
        return httpx.Response(200, json={"events": []})

    transport = httpx.MockTransport(handler)

    with create_fpl_http_client(transport=transport) as http_client:
        response = FPLClient(http_client).get_bootstrap()

        assert http_client.timeout.connect == FPL_CONNECT_TIMEOUT_SECONDS
        assert http_client.timeout.read == FPL_REQUEST_TIMEOUT_SECONDS

    assert response.payload == b'{"events":[]}'


def test_create_fpl_http_client_builds_its_default_transport() -> None:
    with create_fpl_http_client() as http_client:
        assert str(http_client.base_url) == FPL_API_BASE_URL
