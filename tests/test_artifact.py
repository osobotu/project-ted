from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import HttpUrl, ValidationError

from project_ted.data.artifacts import ArtifactProvenance, sha256_digest


def make_provenance(
    *,
    retrieved_at: datetime = datetime(2026, 7, 27, 12, tzinfo=UTC),
    raw_file: str = "bootstrap-static.json",
) -> ArtifactProvenance:
    return ArtifactProvenance(
        source_url=HttpUrl("https://fantasy.premierleague.com/api/bootstrap-static/"),
        retrieved_at=retrieved_at,
        season="2025/26",
        sha256="a" * 64,
        raw_file=raw_file,
    )


def test_sha256_digest_is_stable() -> None:
    assert sha256_digest(b"project-ted") == (
        "55566d751af0305bab5a4f16abaa82c3a51c8b81cb01b57526ecac496b37b972"
    )


def test_provenance_accepts_utc_and_a_plain_filename() -> None:
    provenance = make_provenance()

    assert provenance.retrieved_at.utcoffset() == timedelta(0)
    assert provenance.raw_file == "bootstrap-static.json"


def test_provenance_rejects_a_non_utc_timestamp() -> None:
    kigali_time = datetime(2026, 7, 27, 14, tzinfo=timezone(timedelta(hours=2)))

    with pytest.raises(ValidationError, match="retrieved_at must be in UTC"):
        make_provenance(retrieved_at=kigali_time)


def test_provenance_rejects_a_nested_raw_file_path() -> None:
    with pytest.raises(ValidationError, match="raw_file must be a filename"):
        make_provenance(raw_file="../bootstrap-static.json")
