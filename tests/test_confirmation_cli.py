from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import project_ted.confirmation_cli as cli_module
import pytest
from google.cloud.firestore_v1.client import Client
from project_ted.confirmation_cli import (
    TeamStateInputError,
    confirm_team_from_file,
    main,
)

from project_ted.fpl import PlanningContext
from project_ted.team_state import OwnedPlayer, TeamState
from project_ted.team_state_store import FirestoreTeamStateStore


def team_state() -> TeamState:
    return TeamState(
        season="2026/27",
        planning_gameweek=10,
        squad=(
            OwnedPlayer(
                player_id=1,
                purchase_price_tenths=50,
                selling_price_tenths=50,
            ),
        ),
        bank_tenths=10,
        free_transfers=2,
        used_chips=(),
        version=1,
        confirmed_at=datetime(2026, 10, 20, 12, tzinfo=UTC),
    )


def write_state(path: Path) -> TeamState:
    state = team_state()
    path.write_text(
        state.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return state


def test_reads_and_confirms_a_team_state_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_file = tmp_path / "team-state.json"
    expected_state = write_state(state_file)
    context = cast(PlanningContext, MagicMock())
    client = cast(Client, object())
    store = cast(FirestoreTeamStateStore, object())
    events: list[str] = []

    monkeypatch.setattr(
        cli_module,
        "fetch_planning_context",
        lambda: context,
    )
    monkeypatch.setattr(
        cli_module,
        "Client",
        lambda: client,
    )

    def fake_store(received_client: Client) -> FirestoreTeamStateStore:
        assert received_client is client
        return store

    monkeypatch.setattr(
        cli_module,
        "FirestoreTeamStateStore",
        fake_store,
    )

    def fake_confirm(
        team_id: str,
        state: TeamState,
        **options: object,
    ) -> TeamState:
        assert team_id == "openai"
        assert state == expected_state
        assert options == {
            "planning_gameweek": context.target_gameweek.id,
            "policy": context.rules,
            "players": context.players,
            "store": store,
        }
        events.append("confirm")
        return state

    monkeypatch.setattr(
        cli_module,
        "confirm_team_state",
        fake_confirm,
    )

    result = confirm_team_from_file(
        "openai",
        state_file,
    )

    assert result == expected_state
    assert events == ["confirm"]


def test_rejects_an_invalid_state_file(tmp_path: Path) -> None:
    state_file = tmp_path / "team-state.json"
    state_file.write_text(
        "not-json",
        encoding="utf-8",
    )

    with pytest.raises(
        TeamStateInputError,
        match="does not contain a valid team state",
    ):
        confirm_team_from_file(
            "openai",
            state_file,
        )


def test_main_confirms_and_prints_the_saved_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state_file = tmp_path / "team-state.json"
    expected_state = team_state()

    def fake_confirm_from_file(
        team_id: str,
        received_path: Path,
    ) -> TeamState:
        assert team_id == "openai"
        assert received_path == state_file
        return expected_state

    monkeypatch.setattr(
        cli_module,
        "confirm_team_from_file",
        fake_confirm_from_file,
    )

    main(
        [
            "openai",
            str(state_file),
        ]
    )

    captured = capsys.readouterr()
    assert captured.out == "Confirmed openai team state version 1 for gameweek 10\n"
    assert captured.err == ""
