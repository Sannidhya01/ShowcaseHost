from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class StoredSnapshot:
    archive_key: str
    manifest_key: str
    file_count: int
    total_bytes: int


@dataclass(frozen=True)
class MaterializedSnapshot:
    root: Path
    files: tuple[ManifestEntry, ...]


class SnapshotError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class SnapshotStore(Protocol):
    def temporary_archive_path(self) -> Path: ...

    def validate_and_publish(
        self, repository_id: str, commit_sha: str, temporary_archive: Path
    ) -> StoredSnapshot: ...

    def delete(self, archive_key: str, manifest_key: str) -> None: ...

    def read_manifest(self, manifest_key: str) -> tuple[ManifestEntry, ...]: ...

    def materialize(
        self, archive_key: str, manifest_key: str
    ) -> AbstractContextManager[MaterializedSnapshot]: ...
