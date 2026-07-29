import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import HttpUrl

from project_ted.data.artifacts import (
    ArtifactProvenance,
    VerifiedArtifact,
    sha256_digest,
)
from project_ted.data.catalog import (
    CatalogDataError,
    catalog_from_snapshot,
)
from project_ted.data.snapshots import RunSnapshot
from project_ted.engine.models import Player, Position


def make_artifact(
    *,
    payload: bytes,
    raw_file: str,
    source_url: str,
) -> VerifiedArtifact:
    return VerifiedArtifact(
        provenance=ArtifactProvenance(
            source_url=HttpUrl(source_url),
            retrieved_at=datetime(2026, 7, 29, 12, tzinfo=UTC),
            season="2026/27",
            gameweek=1,
            sha256=sha256_digest(payload),
            raw_file=raw_file,
        ),
        payload=payload,
    )


def make_snapshot(*, away_team_id: int = 2) -> RunSnapshot:
    bootstrap_payload = json.dumps(
        {
            "elements": [
                {
                    "id": 7,
                    "first_name": "Example",
                    "second_name": "Player",
                    "web_name": "Example",
                    "team": 1,
                    "element_type": 3,
                    "now_cost": 125,
                    "status": "a",
                    "news": "",
                    "chance_of_playing_next_round": 100,
                    "form": "6.5",
                    "points_per_game": "5.8",
                    "selected_by_percent": "42.1",
                    "total_points": 180,
                    "minutes": 2800,
                }
            ],
            "teams": [
                {
                    "id": 1,
                    "name": "Home FC",
                    "short_name": "HOM",
                    "strength": 4,
                },
                {
                    "id": 2,
                    "name": "Away FC",
                    "short_name": "AWY",
                    "strength": 3,
                },
            ],
            "events": [
                {
                    "id": 1,
                    "name": "Gameweek 1",
                    "deadline_time": "2026-08-21T17:30:00Z",
                    "finished": False,
                    "data_checked": False,
                    "is_current": False,
                    "is_next": True,
                }
            ],
        }
    ).encode()
    fixtures_payload = json.dumps(
        [
            {
                "id": 1,
                "event": 1,
                "kickoff_time": "2026-08-21T19:00:00Z",
                "team_h": 1,
                "team_a": away_team_id,
                "team_h_difficulty": 2,
                "team_a_difficulty": 5,
                "started": False,
                "finished": False,
                "team_h_score": None,
                "team_a_score": None,
            }
        ]
    ).encode()

    return RunSnapshot(
        snapshot_id="2026-27-gw01-initial",
        directory=Path("snapshots/2026-27-gw01-initial"),
        bootstrap=make_artifact(
            payload=bootstrap_payload,
            raw_file="bootstrap-static.json",
            source_url=("https://fantasy.premierleague.com/api/bootstrap-static/"),
        ),
        fixtures=make_artifact(
            payload=fixtures_payload,
            raw_file="fixtures.json",
            source_url=("https://fantasy.premierleague.com/api/fixtures/"),
        ),
    )


def test_catalog_translates_players_and_fixtures() -> None:
    catalog = catalog_from_snapshot(make_snapshot())

    assert catalog.snapshot_id == "2026-27-gw01-initial"
    assert catalog.players[0].web_name == "Example"
    assert catalog.players[0].position is Position.MIDFIELDER
    assert catalog.players[0].form == 6.5
    assert catalog.fixtures[0].home_difficulty == 2

    assert catalog.engine_players() == {
        7: Player(
            player_id=7,
            team_id=1,
            position=Position.MIDFIELDER,
            now_cost=125,
        )
    }


def test_catalog_rejects_a_fixture_with_an_unknown_team() -> None:
    with pytest.raises(
        CatalogDataError,
        match="unknown away team 99",
    ):
        catalog_from_snapshot(make_snapshot(away_team_id=99))
