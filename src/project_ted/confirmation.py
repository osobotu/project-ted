"""Application workflow for confirming authoritative team state."""

from collections.abc import Iterable
from typing import Protocol

from project_ted.strategy import SeasonPolicy
from project_ted.team_state import (
    InvalidTeamStateError,
    TeamState,
    TeamStatePlayer,
    validate_team_state,
)


class TeamStateStore(Protocol):
    """Persistence operations required by team-state confirmation."""

    def load(
        self,
        team_id: str,
    ) -> TeamState | None: ...

    def save(
        self,
        team_id: str,
        state: TeamState,
        *,
        expected_version: int | None,
    ) -> None: ...


def confirm_team_state(
    team_id: str,
    state: TeamState,
    *,
    planning_gameweek: int,
    policy: SeasonPolicy,
    players: Iterable[TeamStatePlayer],
    store: TeamStateStore,
) -> TeamState:
    """Validate and persist one human-confirmed team state."""

    if state.planning_gameweek != planning_gameweek:
        raise InvalidTeamStateError(
            [
                f"state planning gameweek {state.planning_gameweek} "
                f"does not match current gameweek {planning_gameweek}"
            ]
        )

    validated_state = validate_team_state(
        state,
        policy,
        players,
    )

    current_state = store.load(team_id)
    expected_version = None if current_state is None else current_state.version

    store.save(
        team_id,
        validated_state,
        expected_version=expected_version,
    )

    return validated_state
