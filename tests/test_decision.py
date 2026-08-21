from datetime import UTC, datetime
from typing import cast
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from google.api_core.exceptions import Conflict, ServiceUnavailable
from google.cloud.firestore_v1.client import Client
from pydantic import ValidationError

from project_ted.decision import LockedGameweekDecision
from project_ted.decision_store import (
    CorruptDecisionError,
    DecisionAlreadyLockedError,
    DecisionStoreUnavailableError,
    FirestoreDecisionStore,
)
from project_ted.planning import AgentProvider, GameweekPlan


def gameweek_plan() -> GameweekPlan:
    return GameweekPlan(
        season="2026/27",
        gameweek=1,
        squad=tuple(range(1, 16)),
        starting_xi=tuple(range(1, 12)),
        bench=(12, 13, 14, 15),
        captain_id=1,
        vice_captain_id=2,
        rationale="A balanced opening-gameweek squad.",
        risks=("Several players have uncertain minutes.",),
    )


def locked_decision(
    *,
    model_name: str = "gpt-test",
    locked_at: datetime | None = None,
) -> LockedGameweekDecision:
    return LockedGameweekDecision(
        source_run_id=UUID("b180ace4-911f-48ab-b050-b6b286dd3949"),
        provider=AgentProvider.OPENAI,
        model_name=model_name,
        plan=gameweek_plan(),
        deadline_at=datetime(2026, 8, 21, 17, 30, tzinfo=UTC),
        locked_at=locked_at or datetime(2026, 8, 20, 12, tzinfo=UTC),
    )


def stored_document(
    decision: LockedGameweekDecision,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "decision": decision.model_dump(mode="json"),
    }


def configured_store() -> tuple[
    FirestoreDecisionStore,
    MagicMock,
    MagicMock,
]:
    client_mock = MagicMock()
    teams_collection = client_mock.collection.return_value
    team_document = teams_collection.document.return_value
    decisions_collection = team_document.collection.return_value
    decision_document = decisions_collection.document.return_value
    snapshot = decision_document.get.return_value

    store = FirestoreDecisionStore(
        cast(Client, client_mock),
    )

    return store, decision_document, snapshot


def test_accepts_a_locked_decision_before_the_deadline() -> None:
    decision = locked_decision()

    assert decision.plan.gameweek == 1
    assert decision.provider is AgentProvider.OPENAI


def test_locked_decision_is_immutable() -> None:
    decision = locked_decision()

    with pytest.raises(ValidationError, match="Instance is frozen"):
        decision.model_name = "another-model"


@pytest.mark.parametrize(
    "field",
    [
        "deadline_at",
        "locked_at",
    ],
)
def test_decision_timestamps_must_include_timezones(field: str) -> None:
    data = locked_decision().model_dump(mode="python")
    data[field] = datetime(2026, 8, 20, 12)

    with pytest.raises(
        ValidationError,
        match="decision timestamps must include a timezone",
    ):
        LockedGameweekDecision.model_validate(data)


def test_decision_must_be_locked_before_the_deadline() -> None:
    deadline = datetime(2026, 8, 21, 17, 30, tzinfo=UTC)

    with pytest.raises(
        ValidationError,
        match="decision must be locked before the FPL deadline",
    ):
        locked_decision(locked_at=deadline)


def test_decision_requires_a_canonical_season() -> None:
    data = locked_decision().model_dump(mode="python")
    plan_data = gameweek_plan().model_dump(mode="python")
    plan_data["season"] = "season-2026"
    data["plan"] = plan_data

    with pytest.raises(
        ValidationError,
        match="decision plan must use a canonical FPL season",
    ):
        LockedGameweekDecision.model_validate(data)


def test_transfer_cost_cannot_be_negative() -> None:
    data = locked_decision().model_dump(mode="python")
    data["transfer_cost_points"] = -4

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        LockedGameweekDecision.model_validate(data)


def test_returns_none_when_no_decision_is_locked() -> None:
    store, decision_document, snapshot = configured_store()
    snapshot.exists = False

    result = store.load("openai", "2026/27", 1)

    assert result is None
    decision_document.get.assert_called_once_with()


def test_loads_a_locked_decision() -> None:
    store, _, snapshot = configured_store()
    expected = locked_decision()
    snapshot.exists = True
    snapshot.to_dict.return_value = stored_document(expected)

    result = store.load("openai", "2026/27", 1)

    assert result == expected


def test_locks_a_decision_once() -> None:
    store, decision_document, _ = configured_store()
    decision = locked_decision()

    result = store.lock("openai", decision)

    assert result == decision
    decision_document.create.assert_called_once()

    saved_document = decision_document.create.call_args.args[0]
    assert saved_document["schema_version"] == 1
    assert LockedGameweekDecision.model_validate(saved_document["decision"]) == decision


def test_repeating_the_same_lock_is_idempotent() -> None:
    store, decision_document, snapshot = configured_store()
    decision = locked_decision()
    decision_document.create.side_effect = Conflict(  # type: ignore[no-untyped-call]
        "already exists"
    )
    snapshot.exists = True
    snapshot.to_dict.return_value = stored_document(decision)

    result = store.lock("openai", decision)

    assert result == decision


def test_rejects_a_different_decision_for_a_locked_gameweek() -> None:
    store, decision_document, snapshot = configured_store()
    decision_document.create.side_effect = Conflict(  # type: ignore[no-untyped-call]
        "already exists"
    )
    snapshot.exists = True
    snapshot.to_dict.return_value = stored_document(
        locked_decision(model_name="first-model"),
    )

    with pytest.raises(
        DecisionAlreadyLockedError,
        match="already has a different locked decision",
    ):
        store.lock(
            "openai",
            locked_decision(model_name="second-model"),
        )


def test_rejects_corrupt_stored_decisions() -> None:
    store, _, snapshot = configured_store()
    snapshot.exists = True
    snapshot.to_dict.return_value = {
        "schema_version": 999,
        "decision": {},
    }

    with pytest.raises(
        CorruptDecisionError,
        match="unsupported schema version",
    ):
        store.load("openai", "2026/27", 1)


def test_translates_firestore_lock_failures() -> None:
    store, decision_document, _ = configured_store()
    decision_document.create.side_effect = ServiceUnavailable(  # type: ignore[no-untyped-call]
        "offline"
    )

    with pytest.raises(
        DecisionStoreUnavailableError,
        match="Could not lock decision for openai",
    ):
        store.lock("openai", locked_decision())


@pytest.mark.parametrize(
    ("team_id", "season", "gameweek"),
    [
        ("", "2026/27", 1),
        ("open/ai", "2026/27", 1),
        ("openai", "season-2026", 1),
        ("openai", "2026/27", 0),
    ],
)
def test_rejects_invalid_decision_keys(
    team_id: str,
    season: str,
    gameweek: int,
) -> None:
    store, _, _ = configured_store()

    with pytest.raises(ValueError):
        store.load(team_id, season, gameweek)
