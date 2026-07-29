from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from project_ted.data.fpl_client import FPL_API_BASE_URL, FPLClient
from project_ted.data.snapshots import (
    freeze_run_snapshot,
    load_run_snapshot,
)


def test_freeze_run_snapshot_fetches_once_and_can_be_reloaded(
    tmp_path: Path,
) -> None:
    requested_paths: list[str] = []
    retrieved_at = datetime(2026, 7, 29, 12, tzinfo=UTC)
    payloads = {
        "/api/bootstrap-static/": b'{"elements": [{"id": 7}]}',
        "/api/fixtures/": b'[{"id": 1, "event": 1}]',
    }

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        return httpx.Response(
            200,
            content=payloads[request.url.path],
            headers={"Content-Type": "application/json"},
        )

    with httpx.Client(
        base_url=FPL_API_BASE_URL,
        transport=httpx.MockTransport(handler),
    ) as http_client:
        snapshot = freeze_run_snapshot(
            client=FPLClient(
                http_client,
                clock=lambda: retrieved_at,
            ),
            root=tmp_path / "snapshots",
            season="2026/27",
            gameweek=1,
            run_type="initial",
        )

    assert requested_paths == [
        "/api/bootstrap-static/",
        "/api/fixtures/",
    ]
    assert snapshot.snapshot_id == "2026-27-gw01-initial"
    assert snapshot.bootstrap.payload == payloads["/api/bootstrap-static/"]
    assert snapshot.fixtures.payload == payloads["/api/fixtures/"]

    loaded = load_run_snapshot(snapshot.directory)

    assert loaded == snapshot


def test_freeze_run_snapshot_refuses_an_existing_slot_before_fetching(
    tmp_path: Path,
) -> None:
    root = tmp_path / "snapshots"
    existing = root / "2026-27-gw01-initial"
    existing.mkdir(parents=True)

    def unexpected_request(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request to {request.url}")

    with (
        httpx.Client(
            base_url=FPL_API_BASE_URL,
            transport=httpx.MockTransport(unexpected_request),
        ) as http_client,
        pytest.raises(FileExistsError, match="snapshot already exists"),
    ):
        freeze_run_snapshot(
            client=FPLClient(http_client),
            root=root,
            season="2026/27",
            gameweek=1,
            run_type="initial",
        )
