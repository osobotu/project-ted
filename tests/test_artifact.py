from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Never

import pytest
from pydantic import HttpUrl, ValidationError

from project_ted.data.artifacts import (
    PROVENANCE_FILENAME,
    ArtifactIntegrityError,
    ArtifactProvenance,
    IncompleteArtifactError,
    read_raw_artifact,
    sha256_digest,
    write_raw_artifact,
)


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


def test_provenance_rejects_the_reserved_filename() -> None:
    with pytest.raises(ValidationError, match="reserved provenance filename"):
        make_provenance(raw_file=PROVENANCE_FILENAME)


def test_write_raw_artifact_persists_raw_bytes_and_provenance(tmp_path: Path) -> None:
    artifact_directory = tmp_path / "bootstrap"
    payload = b'{"events": [], "elements": []}'

    provenance = write_raw_artifact(
        directory=artifact_directory,
        raw_file="bootstrap-static.json",
        payload=payload,
        source_url=HttpUrl("https://fantasy.premierleague.com/api/bootstrap-static/"),
        retrieved_at=datetime(2026, 7, 27, 12, tzinfo=UTC),
        season="2025/26",
    )

    stored_provenance = ArtifactProvenance.model_validate_json(
        (artifact_directory / PROVENANCE_FILENAME).read_text()
    )

    assert (artifact_directory / "bootstrap-static.json").read_bytes() == payload
    assert stored_provenance == provenance
    assert provenance.sha256 == sha256_digest(payload)


def test_write_raw_artifact_refuses_to_overwrite_a_completed_artifact(
    tmp_path: Path,
) -> None:
    artifact_directory = tmp_path / "bootstrap"
    original_payload = b'{"version": 1}'

    write_raw_artifact(
        directory=artifact_directory,
        raw_file="bootstrap-static.json",
        payload=original_payload,
        source_url=HttpUrl("https://fantasy.premierleague.com/api/bootstrap-static/"),
        retrieved_at=datetime(2026, 7, 27, 12, tzinfo=UTC),
        season="2025/26",
    )

    with pytest.raises(FileExistsError, match="artifact already completed"):
        write_raw_artifact(
            directory=artifact_directory,
            raw_file="bootstrap-static.json",
            payload=b'{"version": 2}',
            source_url=HttpUrl("https://fantasy.premierleague.com/api/bootstrap-static/"),
            retrieved_at=datetime(2026, 7, 27, 13, tzinfo=UTC),
            season="2025/26",
        )

    assert (artifact_directory / "bootstrap-static.json").read_bytes() == original_payload


def test_write_raw_artifact_removes_temporary_file_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_directory = tmp_path / "bootstrap"

    def fail_replace(*_: object) -> Never:
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(
        "project_ted.data.artifacts.os.replace",
        fail_replace,
    )

    with pytest.raises(OSError, match="simulated replacement failure"):
        write_raw_artifact(
            directory=artifact_directory,
            raw_file="bootstrap-static.json",
            payload=b'{"events": []}',
            source_url=HttpUrl("https://fantasy.premierleague.com/api/bootstrap-static/"),
            retrieved_at=datetime(2026, 7, 27, 12, tzinfo=UTC),
            season="2025/26",
        )

    assert list(artifact_directory.iterdir()) == []
    assert not (artifact_directory / PROVENANCE_FILENAME).exists()


def test_read_raw_artifact_returns_verified_content(tmp_path: Path) -> None:
    artifact_directory = tmp_path / "bootstrap"
    payload = b'{"events": [], "elements": []}'

    expected_provenance = write_raw_artifact(
        directory=artifact_directory,
        raw_file="bootstrap-static.json",
        payload=payload,
        source_url=HttpUrl("https://fantasy.premierleague.com/api/bootstrap-static/"),
        retrieved_at=datetime(2026, 7, 27, 12, tzinfo=UTC),
        season="2025/26",
    )

    artifact = read_raw_artifact(artifact_directory)

    assert artifact.payload == payload
    assert artifact.provenance == expected_provenance


def test_read_raw_artifact_rejects_a_missing_completion_marker(
    tmp_path: Path,
) -> None:
    with pytest.raises(IncompleteArtifactError, match="no completion marker"):
        read_raw_artifact(tmp_path / "incomplete")


def test_read_raw_artifact_rejects_invalid_provenance(tmp_path: Path) -> None:
    artifact_directory = tmp_path / "bootstrap"
    artifact_directory.mkdir()
    (artifact_directory / PROVENANCE_FILENAME).write_text("{not valid json")

    with pytest.raises(ArtifactIntegrityError, match="invalid provenance"):
        read_raw_artifact(artifact_directory)


def test_read_raw_artifact_rejects_a_missing_raw_file(tmp_path: Path) -> None:
    artifact_directory = tmp_path / "bootstrap"

    provenance = write_raw_artifact(
        directory=artifact_directory,
        raw_file="bootstrap-static.json",
        payload=b'{"events": []}',
        source_url=HttpUrl("https://fantasy.premierleague.com/api/bootstrap-static/"),
        retrieved_at=datetime(2026, 7, 27, 12, tzinfo=UTC),
        season="2025/26",
    )
    (artifact_directory / provenance.raw_file).unlink()

    with pytest.raises(ArtifactIntegrityError, match="raw file is missing"):
        read_raw_artifact(artifact_directory)


def test_read_raw_artifact_rejects_changed_raw_content(tmp_path: Path) -> None:
    artifact_directory = tmp_path / "bootstrap"

    provenance = write_raw_artifact(
        directory=artifact_directory,
        raw_file="bootstrap-static.json",
        payload=b'{"version": 1}',
        source_url=HttpUrl("https://fantasy.premierleague.com/api/bootstrap-static/"),
        retrieved_at=datetime(2026, 7, 27, 12, tzinfo=UTC),
        season="2025/26",
    )
    (artifact_directory / provenance.raw_file).write_bytes(b'{"version": 2}')

    with pytest.raises(ArtifactIntegrityError, match="fingerprint does not match"):
        read_raw_artifact(artifact_directory)
