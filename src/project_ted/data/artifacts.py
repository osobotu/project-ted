"""Contracts and utilities for immutable raw-data artifacts."""

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path, PurePath
from tempfile import NamedTemporaryFile
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    ValidationError,
    field_validator,
)

PROVENANCE_FILENAME = "provenance.json"


class IncompleteArtifactError(RuntimeError):
    """Raised when an artifact has no completion marker."""


class ArtifactIntegrityError(RuntimeError):
    """Raised when a completed artifact is missing or changed."""


class ArtifactProvenance(BaseModel):
    """Provenance stored beside an immutable raw response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    source_url: HttpUrl
    retrieved_at: datetime
    season: str = Field(pattern=r"^\d{4}/\d{2}$")
    gameweek: int | None = Field(default=None, ge=1)
    manager_id: int | None = Field(default=None, gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_file: str = Field(min_length=1)

    @field_validator("retrieved_at")
    @classmethod
    def retrieved_at_must_be_utc(cls, value: datetime) -> datetime:
        if value.utcoffset() != timedelta(0):
            raise ValueError("retrieved_at must be in UTC")
        return value

    @field_validator("raw_file")
    @classmethod
    def raw_file_must_be_a_filename(cls, value: str) -> str:
        if PurePath(value).name != value or value in {".", ".."}:
            raise ValueError("raw_file must be a filename, not a path")
        if value == PROVENANCE_FILENAME:
            raise ValueError("raw_file uses the reserved provenance filename")
        return value


@dataclass(frozen=True, slots=True)
class VerifiedArtifact:
    """A raw response whose fingerprint has been verified."""

    provenance: ArtifactProvenance
    payload: bytes


def sha256_digest(payload: bytes) -> str:
    """Return the lowercase SHA-256 digest for raw response bytes."""
    return sha256(payload).hexdigest()


def _atomic_replace(path: Path, payload: bytes) -> None:
    """Durably replace one file without exposing partial contents."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _fsync_directory(directory: Path) -> None:
    """Ensure directory-entry changes reach durable storage."""

    directory_descriptor = os.open(directory, os.O_RDONLY)

    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def write_raw_artifact(
    *,
    directory: Path,
    raw_file: str,
    payload: bytes,
    source_url: HttpUrl,
    retrieved_at: datetime,
    season: str,
    gameweek: int | None = None,
    manager_id: int | None = None,
) -> ArtifactProvenance:
    """Write an immutable raw response and its provenance record."""

    provenance = ArtifactProvenance(
        source_url=source_url,
        retrieved_at=retrieved_at,
        season=season,
        gameweek=gameweek,
        manager_id=manager_id,
        sha256=sha256_digest(payload),
        raw_file=raw_file,
    )

    provenance_path = directory / PROVENANCE_FILENAME

    if provenance_path.exists():
        raise FileExistsError(f"artifact already completed: {directory}")

    _atomic_replace(directory / raw_file, payload)

    provenance_payload = (provenance.model_dump_json(indent=2) + "\n").encode()
    _atomic_replace(provenance_path, provenance_payload)

    _fsync_directory(directory)
    return provenance


def read_raw_artifact(directory: Path) -> VerifiedArtifact:
    """Read an artifact only after verifying its provenance and fingerprint."""

    provenance_path = directory / PROVENANCE_FILENAME

    try:
        provenance_payload = provenance_path.read_bytes()
    except FileNotFoundError as error:
        raise IncompleteArtifactError(f"artifact has no completion marker: {directory}") from error

    try:
        provenance = ArtifactProvenance.model_validate_json(provenance_payload)
    except ValidationError as error:
        raise ArtifactIntegrityError(f"artifact has invalid provenance: {directory}") from error

    raw_path = directory / provenance.raw_file

    try:
        payload = raw_path.read_bytes()
    except FileNotFoundError as error:
        raise ArtifactIntegrityError(f"artifact raw file is missing: {raw_path}") from error

    actual_sha256 = sha256_digest(payload)
    if actual_sha256 != provenance.sha256:
        raise ArtifactIntegrityError(f"artifact fingerprint does not match: {raw_path}")

    return VerifiedArtifact(
        provenance=provenance,
        payload=payload,
    )
