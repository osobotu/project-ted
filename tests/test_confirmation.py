from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from project_ted.confirmation import confirm_team_state
from project_ted.strategy import Position, season_policy_for
from project_ted.team_state import (
    InvalidTeamStateError,
    OwnedPlayer,
    TeamState,
)


@dataclass(frozen=True)
class CatalogPlayer:
    id: int
    team_id: int
    position: Position
    price_tenths: int


class FakeTeamStateStore:
    def __init__(self, existing_state: TeamState | None = None) -> None:
        self.existing_state = existing_state
        self.loaded_team_ids: list[str] = []
        self.saved: list[tuple[str, TeamState, int | None]] = []

    def load(self, team_id: str) -> TeamState | None:
        self.loaded_team_ids.append(team_id)
        return self.existing_state

    def save(
        self,
        team_id: str,
        state: TeamState,
        *,
        expected_version: int | None,
    ) -> None:
        self.saved.append(
            (
                team_id,
                state,
                expected_version,
            )
        )


def catalog_players() -> tuple[CatalogPlayer, ...]:
    positions = (
        Position.GOALKEEPER,
        Position.GOALKEEPER,
        Position.DEFENDER,
        Position.DEFENDER,
        Position.DEFENDER,
        Position.DEFENDER,
        Position.DEFENDER,
        Position.MIDFIELDER,
        Position.MIDFIELDER,
        Position.MIDFIELDER,
        Position.MIDFIELDER,
        Position.MIDFIELDER,
        Position.FORWARD,
        Position.FORWARD,
        Position.FORWARD,
    )

    return tuple(
        CatalogPlayer(
            id=player_id,
            team_id=((player_id - 1) % 5) + 1,
            position=position,
            price_tenths=50,
        )
        for player_id, position in enumerate(
            positions,
            start=1,
        )
    )


def team_state(
    *,
    version: int = 1,
    squad_size: int = 15,
) -> TeamState:
    return TeamState(
        season="2026/27",
        planning_gameweek=10,
        squad=tuple(
            OwnedPlayer(
                player_id=player_id,
                purchase_price_tenths=50,
                selling_price_tenths=50,
            )
            for player_id in range(1, squad_size + 1)
        ),
        bank_tenths=10,
        free_transfers=2,
        used_chips=(),
        version=version,
        confirmed_at=datetime(2026, 10, 20, 12, tzinfo=UTC),
    )


def test_confirms_the_first_valid_team_state() -> None:
    store = FakeTeamStateStore()
    candidate = team_state(version=1)

    result = confirm_team_state(
        "openai",
        candidate,
        planning_gameweek=10,
        policy=season_policy_for("2026/27"),
        players=catalog_players(),
        store=store,
    )

    assert result is candidate
    assert store.saved == [
        (
            "openai",
            candidate,
            None,
        )
    ]


def test_uses_the_current_version_when_updating_state() -> None:
    current = team_state(version=3)
    candidate = team_state(version=4)
    store = FakeTeamStateStore(current)

    confirm_team_state(
        "openai",
        candidate,
        planning_gameweek=10,
        policy=season_policy_for("2026/27"),
        players=catalog_players(),
        store=store,
    )

    assert store.saved == [
        (
            "openai",
            candidate,
            3,
        )
    ]


def test_rejects_invalid_state_before_reading_or_writing_storage() -> None:
    store = FakeTeamStateStore()

    with pytest.raises(
        InvalidTeamStateError,
        match="squad must contain 15 players; received 14",
    ):
        confirm_team_state(
            "openai",
            team_state(squad_size=14),
            planning_gameweek=10,
            policy=season_policy_for("2026/27"),
            players=catalog_players(),
            store=store,
        )

    assert store.loaded_team_ids == []
    assert store.saved == []


def test_rejects_state_for_another_planning_gameweek() -> None:
    store = FakeTeamStateStore()

    with pytest.raises(
        InvalidTeamStateError,
        match="state planning gameweek 10 does not match current gameweek 11",
    ):
        confirm_team_state(
            "openai",
            team_state(),
            planning_gameweek=11,
            policy=season_policy_for("2026/27"),
            players=catalog_players(),
            store=store,
        )

    assert store.loaded_team_ids == []
    assert store.saved == []
