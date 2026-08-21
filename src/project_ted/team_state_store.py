"""Firestore persistence."""

from typing import Final, cast

from google.api_core.exceptions import GoogleAPICallError
from google.cloud.firestore_v1.client import Client
from google.cloud.firestore_v1.document import DocumentReference
from google.cloud.firestore_v1.transaction import Transaction, transactional
from pydantic import ValidationError

from project_ted.team_state import TeamState

_TEAM_COLLECTION: Final = "managed_teams"
_STATE_COLLECTION: Final = "state"
_CURRENT_STATE_DOCUMENT: Final = "current"
_SCHEMA_VERSION: Final = 1


class TeamStateStoreError(RuntimeError):
    """Base error for authoritative team-state persistence."""


class TeamStateConflictError(TeamStateStoreError):
    """Report an attempt to overwrite newer or unexpected state."""


class CorruptTeamStateError(TeamStateStoreError):
    """Report that stored state cannot be decoded safely."""


class TeamStateUnavailableError(TeamStateStoreError):
    """Report that Firestore could not complete an operation."""


class FirestoreTeamStateStore:
    """Persist one current, versioned state document per managed team."""

    def __init__(self, client: Client) -> None:
        self._client = client

    def load(self, team_id: str) -> TeamState | None:
        """Load the current state, or return None before its first confirmation."""

        _validate_team_id(team_id)
        document = self._current_state_document(team_id)

        try:
            snapshot = document.get()
        except GoogleAPICallError as error:
            raise TeamStateUnavailableError(f"Could not load team state for {team_id}") from error

        if not snapshot.exists:
            return None

        return _decode_team_state(
            snapshot.to_dict(),
            team_id=team_id,
        )

    def save(
        self,
        team_id: str,
        state: TeamState,
        *,
        expected_version: int | None,
    ) -> None:
        """Atomically save state when its predecessor matches expectations."""

        _validate_team_id(team_id)
        document = self._current_state_document(team_id)
        transaction = self._client.transaction()

        @transactional
        def persist(transaction: Transaction) -> None:
            snapshot = document.get(transaction=transaction)
            existing_state = None

            if snapshot.exists:
                existing_state = _decode_team_state(
                    snapshot.to_dict(),
                    team_id=team_id,
                )

            _validate_version_transition(
                existing_state,
                state,
                expected_version=expected_version,
            )
            transaction.set(
                document,
                _encode_team_state(state),
            )

        try:
            persist(transaction)
        except GoogleAPICallError as error:
            raise TeamStateUnavailableError(f"Could not save team state for {team_id}") from error

    def _current_state_document(
        self,
        team_id: str,
    ) -> DocumentReference:
        document = (
            self._client.collection(_TEAM_COLLECTION)
            .document(team_id)
            .collection(_STATE_COLLECTION)
            .document(_CURRENT_STATE_DOCUMENT)
        )

        return cast(DocumentReference, document)


def _validate_team_id(team_id: str) -> None:
    if not team_id or team_id != team_id.strip() or "/" in team_id:
        raise ValueError(
            "team ID must be non-empty, contain no surrounding whitespace, and contain no slash"
        )


def _validate_version_transition(
    existing_state: TeamState | None,
    new_state: TeamState,
    *,
    expected_version: int | None,
) -> None:
    if existing_state is None:
        if expected_version is not None:
            raise TeamStateConflictError(
                f"Expected version {expected_version}, but no team state exists"
            )

        if new_state.version != 1:
            raise TeamStateConflictError("The first team-state version must be 1")

        return

    if expected_version is None:
        raise TeamStateConflictError("Expected version is required when team state already exists")

    if existing_state.version != expected_version:
        raise TeamStateConflictError(
            f"Expected version {expected_version}, but found version {existing_state.version}"
        )

    required_new_version = existing_state.version + 1

    if new_state.version != required_new_version:
        raise TeamStateConflictError(f"New team-state version must be {required_new_version}")


def _encode_team_state(state: TeamState) -> dict[str, object]:
    state_document: dict[str, object] = {
        "season": state.season,
        "planning_gameweek": state.planning_gameweek,
        "squad": [
            {
                "player_id": player.player_id,
                "purchase_price_tenths": player.purchase_price_tenths,
                "selling_price_tenths": player.selling_price_tenths,
            }
            for player in state.squad
        ],
        "bank_tenths": state.bank_tenths,
        "free_transfers": state.free_transfers,
        "used_chips": [
            {
                "chip": usage.chip.value,
                "gameweek": usage.gameweek,
            }
            for usage in state.used_chips
        ],
        "version": state.version,
        "confirmed_at": state.confirmed_at,
    }

    return {
        "schema_version": _SCHEMA_VERSION,
        "state": state_document,
    }


def _decode_team_state(
    document_data: object,
    *,
    team_id: str,
) -> TeamState:
    if not isinstance(document_data, dict):
        raise CorruptTeamStateError(f"Stored team state for {team_id} is not a document")

    if document_data.get("schema_version") != _SCHEMA_VERSION:
        raise CorruptTeamStateError(
            f"Stored team state for {team_id} has an unsupported schema version"
        )

    try:
        return TeamState.model_validate(document_data.get("state"))
    except ValidationError as error:
        raise CorruptTeamStateError(f"Stored team state for {team_id} is invalid") from error
