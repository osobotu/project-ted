"""Firestore persistence for immutable agent gameweek decisions."""

import re
from typing import Final, cast

from google.api_core.exceptions import Conflict, GoogleAPICallError
from google.cloud.firestore_v1.client import Client
from google.cloud.firestore_v1.document import DocumentReference
from pydantic import ValidationError

from project_ted.decision import LockedGameweekDecision

_TEAM_COLLECTION: Final = "managed_teams"
_DECISION_COLLECTION: Final = "decisions"
_SCHEMA_VERSION: Final = 1


class DecisionStoreError(RuntimeError):
    """Base error for locked-decision persistence."""


class DecisionAlreadyLockedError(DecisionStoreError):
    """Report an attempt to replace a different locked decision."""


class CorruptDecisionError(DecisionStoreError):
    """Report a stored decision that cannot be decoded safely."""


class DecisionStoreUnavailableError(DecisionStoreError):
    """Report that Firestore could not complete an operation."""


class FirestoreDecisionStore:
    """Create and retrieve immutable decisions for managed teams."""

    def __init__(self, client: Client) -> None:
        self._client = client

    def load(
        self,
        team_id: str,
        season: str,
        gameweek: int,
    ) -> LockedGameweekDecision | None:
        """Load the locked decision for one team and deadline."""

        document = self._decision_document(
            team_id,
            season,
            gameweek,
        )

        try:
            snapshot = document.get()
        except GoogleAPICallError as error:
            raise DecisionStoreUnavailableError(f"Could not load decision for {team_id}") from error

        if not snapshot.exists:
            return None

        return _decode_decision(
            snapshot.to_dict(),
            team_id=team_id,
            season=season,
            gameweek=gameweek,
        )

    def lock(
        self,
        team_id: str,
        decision: LockedGameweekDecision,
    ) -> LockedGameweekDecision:
        """Create a decision once and reject conflicting replacements."""

        season = decision.plan.season
        gameweek = decision.plan.gameweek
        document = self._decision_document(
            team_id,
            season,
            gameweek,
        )

        try:
            document.create(
                _encode_decision(decision),
            )
        except Conflict as error:
            existing_decision = self.load(
                team_id,
                season,
                gameweek,
            )

            if existing_decision == decision:
                return existing_decision

            raise DecisionAlreadyLockedError(
                f"{team_id} already has a different locked decision "
                f"for {season} gameweek {gameweek}"
            ) from error
        except GoogleAPICallError as error:
            raise DecisionStoreUnavailableError(f"Could not lock decision for {team_id}") from error

        return decision

    def _decision_document(
        self,
        team_id: str,
        season: str,
        gameweek: int,
    ) -> DocumentReference:
        _validate_decision_key(
            team_id,
            season,
            gameweek,
        )
        document_id = _decision_document_id(
            season,
            gameweek,
        )

        document = (
            self._client.collection(_TEAM_COLLECTION)
            .document(team_id)
            .collection(_DECISION_COLLECTION)
            .document(document_id)
        )

        return cast(DocumentReference, document)


def _validate_decision_key(
    team_id: str,
    season: str,
    gameweek: int,
) -> None:
    if not team_id or team_id != team_id.strip() or "/" in team_id:
        raise ValueError(
            "team ID must be non-empty, contain no surrounding whitespace, and contain no slash"
        )

    if re.fullmatch(r"\d{4}/\d{2}", season) is None:
        raise ValueError("season must use the YYYY/YY format")

    if gameweek <= 0:
        raise ValueError("gameweek must be positive")


def _decision_document_id(
    season: str,
    gameweek: int,
) -> str:
    season_id = season.replace("/", "-")
    return f"{season_id}-gw-{gameweek:02d}"


def _encode_decision(
    decision: LockedGameweekDecision,
) -> dict[str, object]:
    decision_document: object = decision.model_dump(mode="json")

    return {
        "schema_version": _SCHEMA_VERSION,
        "decision": decision_document,
    }


def _decode_decision(
    document_data: object,
    *,
    team_id: str,
    season: str,
    gameweek: int,
) -> LockedGameweekDecision:
    location = f"{team_id} {season} gameweek {gameweek}"

    if not isinstance(document_data, dict):
        raise CorruptDecisionError(f"Stored decision for {location} is not a document")

    if document_data.get("schema_version") != _SCHEMA_VERSION:
        raise CorruptDecisionError(
            f"Stored decision for {location} has an unsupported schema version"
        )

    try:
        decision = LockedGameweekDecision.model_validate(document_data.get("decision"))
    except ValidationError as error:
        raise CorruptDecisionError(f"Stored decision for {location} is invalid") from error

    if decision.plan.season != season or decision.plan.gameweek != gameweek:
        raise CorruptDecisionError(f"Stored decision for {location} does not match its location")

    return decision
