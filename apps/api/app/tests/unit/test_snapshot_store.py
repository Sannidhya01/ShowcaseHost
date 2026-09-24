import io
import json
import tarfile
from pathlib import Path

import pytest

from app.core.config import Settings
from app.integrations.snapshots.base import SnapshotError
from app.integrations.snapshots.local import LocalSnapshotStore


def archive(path: Path, files: dict[str, bytes], *, symlink: str | None = None) -> None:
    with tarfile.open(path, "w:gz") as output:
        for name, content in files.items():
            info = tarfile.TarInfo(f"owner-repo-sha/{name}")
            info.size = len(content)
            output.addfile(info, io.BytesIO(content))
        if symlink:
            info = tarfile.TarInfo("owner-repo-sha/link")
            info.type = tarfile.SYMTYPE
            info.linkname = symlink
            output.addfile(info)


def test_snapshot_is_validated_manifested_and_published(tmp_path: Path) -> None:
    settings = Settings(snapshot_root=tmp_path / "snapshots")
    store = LocalSnapshotStore(settings)
    temporary = store.temporary_archive_path()
    archive(temporary, {"src/a.py": b"print('a')\n", "README.md": b"hello"}, symlink="/etc/passwd")

    stored = store.validate_and_publish("123", "a" * 40, temporary)

    assert stored.file_count == 2
    assert stored.total_bytes == 16
    manifest = json.loads((store.root / stored.manifest_key).read_text())
    assert [item["path"] for item in manifest["files"]] == ["README.md", "src/a.py"]
    assert not temporary.exists()


def test_snapshot_rejects_path_traversal(tmp_path: Path) -> None:
    settings = Settings(snapshot_root=tmp_path / "snapshots")
    store = LocalSnapshotStore(settings)
    temporary = store.temporary_archive_path()
    with tarfile.open(temporary, "w:gz") as output:
        info = tarfile.TarInfo("owner-repo-sha/../escape")
        info.size = 1
        output.addfile(info, io.BytesIO(b"x"))

    with pytest.raises(SnapshotError, match="unsafe path"):
        store.validate_and_publish("123", "b" * 40, temporary)

    assert not (tmp_path / "escape").exists()


def test_snapshot_enforces_file_size_limit(tmp_path: Path) -> None:
    settings = Settings(snapshot_root=tmp_path / "snapshots", ingestion_max_file_bytes=2)
    store = LocalSnapshotStore(settings)
    temporary = store.temporary_archive_path()
    archive(temporary, {"large.txt": b"abc"})

    with pytest.raises(SnapshotError) as error:
        store.validate_and_publish("123", "c" * 40, temporary)

    assert error.value.code == "file_too_large"


def test_snapshot_materialization_revalidates_manifest_and_cleans_workspace(tmp_path: Path) -> None:
    settings = Settings(snapshot_root=tmp_path / "snapshots")
    store = LocalSnapshotStore(settings)
    temporary = store.temporary_archive_path()
    archive(temporary, {"src/a.py": b"print('a')\n"})
    stored = store.validate_and_publish("123", "d" * 40, temporary)

    with store.materialize(stored.archive_key, stored.manifest_key) as materialized:
        workspace = materialized.root
        assert (workspace / "src/a.py").read_bytes() == b"print('a')\n"
        assert materialized.files[0].sha256

    assert not workspace.exists()
