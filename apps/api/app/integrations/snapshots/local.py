from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from app.core.config import Settings
from app.integrations.snapshots.base import (
    ManifestEntry,
    MaterializedSnapshot,
    SnapshotError,
    StoredSnapshot,
)


class LocalSnapshotStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = settings.snapshot_root.resolve()
        self.temporary_root = self.root / ".tmp"

    def temporary_archive_path(self) -> Path:
        self.temporary_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, name = tempfile.mkstemp(suffix=".tar.gz", dir=self.temporary_root)
        os.close(descriptor)
        return Path(name)

    def validate_and_publish(
        self, repository_id: str, commit_sha: str, temporary_archive: Path
    ) -> StoredSnapshot:
        workspace = Path(tempfile.mkdtemp(prefix="workspace-", dir=self.temporary_root))
        temporary_manifest = self.temporary_root / f"{temporary_archive.name}.manifest"
        try:
            entries = self._extract_and_manifest(temporary_archive, workspace)
            if not entries:
                raise SnapshotError("empty_archive", "Repository archive contains no regular files")
            destination = self.root / "github" / repository_id / commit_sha
            destination.mkdir(parents=True, exist_ok=True, mode=0o700)
            archive = destination / "source.tar.gz"
            manifest = destination / "manifest.json"
            temporary_manifest.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "repository_id": repository_id,
                        "commit_sha": commit_sha,
                        "files": [entry.__dict__ for entry in entries],
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            os.replace(temporary_archive, archive)
            os.replace(temporary_manifest, manifest)
            return StoredSnapshot(
                archive_key=str(archive.relative_to(self.root)),
                manifest_key=str(manifest.relative_to(self.root)),
                file_count=len(entries),
                total_bytes=sum(entry.size for entry in entries),
            )
        except (tarfile.TarError, OSError) as exc:
            raise SnapshotError(
                "invalid_archive", "Repository archive could not be processed"
            ) from exc
        finally:
            temporary_archive.unlink(missing_ok=True)
            temporary_manifest.unlink(missing_ok=True)
            shutil.rmtree(workspace, ignore_errors=True)

    def _extract_and_manifest(self, archive: Path, workspace: Path) -> list[ManifestEntry]:
        entries: list[ManifestEntry] = []
        seen: set[str] = set()
        total_bytes = 0
        root_component: str | None = None
        with tarfile.open(archive, mode="r:*") as source:
            for member in source:
                raw_path = PurePosixPath(member.name)
                if raw_path.is_absolute() or ".." in raw_path.parts or "\\" in member.name:
                    raise SnapshotError("unsafe_archive_path", "Archive contains an unsafe path")
                if not raw_path.parts:
                    continue
                if root_component is None:
                    root_component = raw_path.parts[0]
                if raw_path.parts[0] != root_component:
                    raise SnapshotError("invalid_archive_root", "Archive contains multiple roots")
                relative_parts = raw_path.parts[1:]
                if not relative_parts:
                    continue
                relative = PurePosixPath(*relative_parts)
                if member.isdir():
                    continue
                if member.issym() or member.islnk():
                    continue
                if not member.isfile():
                    raise SnapshotError("unsafe_archive_entry", "Archive contains a special file")
                if str(relative) in seen:
                    raise SnapshotError(
                        "duplicate_archive_path", "Archive contains duplicate paths"
                    )
                if member.size > self.settings.ingestion_max_file_bytes:
                    raise SnapshotError("file_too_large", f"File exceeds limit: {relative}")
                if len(entries) + 1 > self.settings.ingestion_max_files:
                    raise SnapshotError("too_many_files", "Repository exceeds file-count limit")
                total_bytes += member.size
                if total_bytes > self.settings.ingestion_max_extracted_bytes:
                    raise SnapshotError(
                        "extracted_content_too_large", "Repository exceeds extracted-size limit"
                    )
                extracted = source.extractfile(member)
                if extracted is None:
                    raise SnapshotError("invalid_archive", "Archive file cannot be read")
                destination = workspace.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                digest = hashlib.sha256()
                written = 0
                with destination.open("wb") as handle:
                    while chunk := extracted.read(1024 * 1024):
                        written += len(chunk)
                        if (
                            written > member.size
                            or written > self.settings.ingestion_max_file_bytes
                        ):
                            raise SnapshotError("file_too_large", f"File exceeds limit: {relative}")
                        digest.update(chunk)
                        handle.write(chunk)
                if written != member.size:
                    raise SnapshotError(
                        "invalid_archive", "Archive file size does not match metadata"
                    )
                seen.add(str(relative))
                entries.append(
                    ManifestEntry(path=str(relative), size=written, sha256=digest.hexdigest())
                )
        entries.sort(key=lambda entry: entry.path)
        return entries

    def delete(self, archive_key: str, manifest_key: str) -> None:
        for key in (archive_key, manifest_key):
            path = (self.root / key).resolve()
            if self.root not in path.parents:
                raise SnapshotError("invalid_snapshot_key", "Snapshot key escapes storage root")
            path.unlink(missing_ok=True)
        directory = (self.root / archive_key).parent
        try:
            directory.rmdir()
        except OSError:
            pass

    def read_manifest(self, manifest_key: str) -> tuple[ManifestEntry, ...]:
        path = self._resolve_key(manifest_key)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("version") != 1 or not isinstance(payload.get("files"), list):
                raise ValueError("unsupported manifest")
            return tuple(
                ManifestEntry(path=item["path"], size=item["size"], sha256=item["sha256"])
                for item in payload["files"]
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise SnapshotError("invalid_manifest", "Snapshot manifest is invalid") from exc

    @contextmanager
    def materialize(self, archive_key: str, manifest_key: str) -> Iterator[MaterializedSnapshot]:
        archive = self._resolve_key(archive_key)
        expected = self.read_manifest(manifest_key)
        self.temporary_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        workspace = Path(tempfile.mkdtemp(prefix="materialized-", dir=self.temporary_root))
        try:
            actual = tuple(self._extract_and_manifest(archive, workspace))
            if actual != expected:
                raise SnapshotError(
                    "snapshot_manifest_mismatch", "Snapshot contents do not match the manifest"
                )
            yield MaterializedSnapshot(root=workspace, files=actual)
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def _resolve_key(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if self.root not in path.parents:
            raise SnapshotError("invalid_snapshot_key", "Snapshot key escapes storage root")
        return path
