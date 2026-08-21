"""Command-line entry point for human team-state confirmation."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from google.auth.exceptions import GoogleAuthError
from google.cloud.firestore_v1.client import Client
from pydantic import ValidationError

from project_ted.confirmation import confirm_team_state
from project_ted.fpl import FplDataError, fetch_planning_context
from project_ted.team_state import TeamState
from project_ted.team_state_store import (
    FirestoreTeamStateStore,
    TeamStateStoreError,
)


class TeamStateInputError(ValueError):
    """Report a team-state file that cannot be read or validated."""


def confirm_team_from_file(
    team_id: str,
    state_file: Path,
) -> TeamState:
    """Confirm one JSON team-state submission against live FPL data."""

    state = _read_team_state(state_file)
    context = fetch_planning_context()
    store = FirestoreTeamStateStore(Client())

    return confirm_team_state(
        team_id,
        state,
        planning_gameweek=context.target_gameweek.id,
        policy=context.rules,
        players=context.players,
        store=store,
    )


def main(
    argv: Sequence[str] | None = None,
) -> None:
    """Parse a human submission and persist its confirmed state."""

    parser = _build_parser()
    arguments = parser.parse_args(argv)

    try:
        state = confirm_team_from_file(
            arguments.team_id,
            arguments.state_file,
        )
    except (
        FplDataError,
        GoogleAuthError,
        TeamStateStoreError,
        ValueError,
    ) as error:
        parser.error(f"team confirmation failed: {error}")

    print(
        f"Confirmed {arguments.team_id} team state "
        f"version {state.version} "
        f"for gameweek {state.planning_gameweek}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Validate and save a human-confirmed Project Ted team state.")
    )
    parser.add_argument(
        "team_id",
        help="Stable managed-team identifier, such as openai or anthropic.",
    )
    parser.add_argument(
        "state_file",
        type=Path,
        help="Path to the JSON TeamState submission.",
    )

    return parser


def _read_team_state(
    state_file: Path,
) -> TeamState:
    try:
        payload = state_file.read_text(
            encoding="utf-8",
        )
    except OSError as error:
        raise TeamStateInputError(f"Could not read team-state file {state_file}") from error

    try:
        return TeamState.model_validate_json(payload)
    except ValidationError as error:
        raise TeamStateInputError(f"{state_file} does not contain a valid team state") from error
