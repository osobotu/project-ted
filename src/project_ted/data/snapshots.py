"""Freeze FPL responses for one agent run."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from project_ted.data.artifacts import (
    ArtifactIntegrityError,
    VerifiedArtifact,
    read_raw_artifact,
    write_raw_artifact,
)
from project_ted.data.fpl_client import FPLClient

type RunType = Literal["initial", "planning", "final"]


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    """Verified FPL data shared by every agent in one run."""

    snapshot_id: str
    directory: Path
    bootstrap: VerifiedArtifact
    fixtures: VerifiedArtifact


def freeze_run_snapshot(
    *,
    client: FPLClient,
    root: Path,
    season: str,
    gameweek: int,
    run_type: RunType,
) -> RunSnapshot:
    """Fetch bootstrap and fixtures once and persist them immutably."""

    if isinstance(gameweek, bool) or gameweek < 1:
        raise ValueError("gameweek must be a positive integer")

    snapshot_id = f"{season.replace('/', '-')}-gw{gameweek:02d}-{run_type}"
    directory = root / snapshot_id

    if directory.exists():
        raise FileExistsError(f"snapshot already exists: {snapshot_id}")

    bootstrap_response = client.get_bootstrap()
    fixtures_response = client.get_fixtures()

    write_raw_artifact(
        directory=directory / "bootstrap",
        raw_file="bootstrap-static.json",
        payload=bootstrap_response.payload,
        source_url=bootstrap_response.source_url,
        retrieved_at=bootstrap_response.retrieved_at,
        season=season,
        gameweek=gameweek,
    )
    write_raw_artifact(
        directory=directory / "fixtures",
        raw_file="fixtures.json",
        payload=fixtures_response.payload,
        source_url=fixtures_response.source_url,
        retrieved_at=fixtures_response.retrieved_at,
        season=season,
        gameweek=gameweek,
    )

    return load_run_snapshot(directory)


def load_run_snapshot(directory: Path) -> RunSnapshot:
    """Load and verify a previously frozen run snapshot."""

    bootstrap = read_raw_artifact(directory / "bootstrap")
    fixtures = read_raw_artifact(directory / "fixtures")

    metadata_matches = (
        bootstrap.provenance.season == fixtures.provenance.season
        and bootstrap.provenance.gameweek == fixtures.provenance.gameweek
    )
    if not metadata_matches:
        raise ArtifactIntegrityError(f"snapshot artifacts disagree: {directory}")

    return RunSnapshot(
        snapshot_id=directory.name,
        directory=directory,
        bootstrap=bootstrap,
        fixtures=fixtures,
    )
