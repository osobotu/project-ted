from datetime import UTC, datetime
from typing import cast
from unittest.mock import MagicMock

import pytest
from google.api_core.exceptions import ServiceUnavailable
from google.cloud.firestore_v1.client import Client

from project_ted.strategy import Chip
from project_ted.team_state import ChipUsage, OwnedPlayer, TeamState
from project_ted.team_state_store import (
    CorruptTeamStateError,
    FirestoreTeamStateStore,
    TeamStateConflictError,
    TeamStateUnavailableError,
)


def _run_directly(function: object) -> object:
    return function


@pytest.fixture(autouse=True)
def execute_transactions_inline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "project_ted.team_state_store.transactional",
        _run_directly,
    )


def team_state(version: int = 1) -> TeamState:
    return TeamState(
        season="2026/27",
        planning_gameweek=10,
        squad=(
            OwnedPlayer(
                player_id=1,
                purchase_price_tenths=75,
                selling_price_tenths=76,
            ),
        ),
        bank_tenths=10,
        free_transfers=2,
        used_chips=(
            ChipUsage(
                chip=Chip.TRIPLE_CAPTAIN,
                gameweek=5,
            ),
        ),
        version=version,
        confirmed_at=datetime(
            2026,
            10,
            20,
            12,
            tzinfo=UTC,
        ),
    )


def stored_document(state: TeamState) -> dict[str, object]:
    return {
        "schema_version": 1,
        "state": state.model_dump(mode="python"),
    }


def configured_store() -> tuple[
    FirestoreTeamStateStore,
    MagicMock,
    MagicMock,
    MagicMock,
]:
    client_mock = MagicMock()
    teams_collection = client_mock.collection.return_value
    team_document = teams_collection.document.return_value
    state_collection = team_document.collection.return_value
    current_document = state_collection.document.return_value
    transaction = client_mock.transaction.return_value
    snapshot = current_document.get.return_value

    store = FirestoreTeamStateStore(
        cast(Client, client_mock),
    )

    return (
        store,
        current_document,
        transaction,
        snapshot,
    )


def test_returns_none_when_team_has_not_been_confirmed() -> None:
    store, current_document, _, snapshot = configured_store()
    snapshot.exists = False

    result = store.load("openai")

    assert result is None
    current_document.get.assert_called_once_with()


def test_loads_valid_team_state() -> None:
    store, _, _, snapshot = configured_store()
    expected_state = team_state()
    snapshot.exists = True
    snapshot.to_dict.return_value = stored_document(expected_state)

    result = store.load("openai")

    assert result == expected_state


def test_rejects_an_unsupported_storage_schema() -> None:
    store, _, _, snapshot = configured_store()
    snapshot.exists = True
    snapshot.to_dict.return_value = {
        "schema_version": 999,
        "state": {},
    }

    with pytest.raises(
        CorruptTeamStateError,
        match="unsupported schema version",
    ):
        store.load("openai")


def test_translates_firestore_load_failures() -> None:
    store, current_document, _, _ = configured_store()
    current_document.get.side_effect = ServiceUnavailable(  # type: ignore[no-untyped-call]
        "offline"
    )

    with pytest.raises(
        TeamStateUnavailableError,
        match="Could not load team state for openai",
    ):
        store.load("openai")


def test_creates_the_first_team_state() -> None:
    store, current_document, transaction, snapshot = configured_store()
    state = team_state(version=1)
    snapshot.exists = False

    store.save(
        "openai",
        state,
        expected_version=None,
    )

    current_document.get.assert_called_once_with(
        transaction=transaction,
    )
    transaction.set.assert_called_once()

    saved_reference, saved_document = transaction.set.call_args.args

    assert saved_reference is current_document
    assert saved_document["schema_version"] == 1
    assert TeamState.model_validate(saved_document["state"]) == state


def test_first_team_state_must_have_version_one() -> None:
    store, _, _, snapshot = configured_store()
    snapshot.exists = False

    with pytest.raises(
        TeamStateConflictError,
        match="first team-state version must be 1",
    ):
        store.save(
            "openai",
            team_state(version=2),
            expected_version=None,
        )


def test_existing_state_requires_an_expected_version() -> None:
    store, _, _, snapshot = configured_store()
    snapshot.exists = True
    snapshot.to_dict.return_value = stored_document(
        team_state(version=1),
    )

    with pytest.raises(
        TeamStateConflictError,
        match="Expected version is required",
    ):
        store.save(
            "openai",
            team_state(version=2),
            expected_version=None,
        )


def test_rejects_a_stale_expected_version() -> None:
    store, _, _, snapshot = configured_store()
    snapshot.exists = True
    snapshot.to_dict.return_value = stored_document(
        team_state(version=3),
    )

    with pytest.raises(
        TeamStateConflictError,
        match="Expected version 2, but found version 3",
    ):
        store.save(
            "openai",
            team_state(version=4),
            expected_version=2,
        )


def test_requires_the_next_sequential_version() -> None:
    store, _, _, snapshot = configured_store()
    snapshot.exists = True
    snapshot.to_dict.return_value = stored_document(
        team_state(version=3),
    )

    with pytest.raises(
        TeamStateConflictError,
        match="New team-state version must be 4",
    ):
        store.save(
            "openai",
            team_state(version=5),
            expected_version=3,
        )


def test_updates_state_with_the_next_version() -> None:
    store, current_document, transaction, snapshot = configured_store()
    current_state = team_state(version=3)
    new_state = team_state(version=4)
    snapshot.exists = True
    snapshot.to_dict.return_value = stored_document(current_state)

    store.save(
        "openai",
        new_state,
        expected_version=3,
    )

    transaction.set.assert_called_once()
    saved_reference, saved_document = transaction.set.call_args.args

    assert saved_reference is current_document
    assert TeamState.model_validate(saved_document["state"]) == new_state


@pytest.mark.parametrize(
    "team_id",
    [
        "",
        " openai",
        "openai ",
        "open/ai",
    ],
)
def test_rejects_invalid_team_ids(team_id: str) -> None:
    store, _, _, _ = configured_store()

    with pytest.raises(ValueError, match="team ID must be non-empty"):
        store.load(team_id)
