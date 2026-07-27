"""Contracts and utilities for immutable raw-data artifacts."""

from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import PurePath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


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
        return value


def sha256_digest(payload: bytes) -> str:
    """Return the lowercase SHA-256 digest for raw response bytes."""
    return sha256(payload).hexdigest()
