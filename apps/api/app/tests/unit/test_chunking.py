import json
import sys
import uuid
from pathlib import Path

import pytest
from sqlalchemy import String

from app.chunking.client import ENGINE_VERSION, ChunkingWorkerClient, ChunkingWorkerError
from app.chunking.profile import ChunkingProfile, Utf8ByteTokenCounter
from app.core.config import Settings
from app.db.models import ChunkedFile
from app.integrations.snapshots.base import ManifestEntry
from app.services.chunking import file_fingerprint, searchable_chunk_metadata, skip_reason


def test_profile_is_deterministic_and_embedding_window_aware() -> None:
    settings = Settings(
        embedding_input_token_limit=1000,
        chunking_target_raw_tokens=900,
        chunking_safety_margin_ratio=0.1,
    )

    first = ChunkingProfile.from_settings(settings)
    second = ChunkingProfile.from_settings(settings)

    assert first.key == second.key
    assert first.max_chunk_size == 630
    assert first.options["contextMode"] == "full"
    assert first.options["overlapRatio"] == 0.2
    assert round(first.max_chunk_size * first.overlap_ratio) == 126
    default = ChunkingProfile.from_settings(Settings())
    assert round(default.max_chunk_size * default.overlap_ratio) == 205


def test_utf8_counter_is_conservative_and_handles_multibyte_text() -> None:
    counter = Utf8ByteTokenCounter()

    assert counter.count("abc") == 3
    assert counter.count("λ") == 2


def test_skip_classification_is_explicit() -> None:
    assert skip_reason(ManifestEntry("README.md", 1, "a" * 64)) is None
    assert (
        skip_reason(ManifestEntry("dist/bundle.js", 1, "b" * 64)) == "generated_or_dependency_path"
    )
    assert skip_reason(ManifestEntry("src/bundle.min.js", 1, "c" * 64)) == "minified_file"
    assert skip_reason(ManifestEntry("src/main.rs", 1, "d" * 64)) is None
    assert skip_reason(ManifestEntry("src/main.cpp", 1, "e" * 64)) is None
    assert skip_reason(ManifestEntry("Gemfile", 1, "f" * 64)) is None
    assert skip_reason(ManifestEntry("src/Main.kt", 1, "1" * 64)) is None
    assert skip_reason(ManifestEntry("src/legacy.cob", 1, "2" * 64)) is None
    assert skip_reason(ManifestEntry("assets/logo.png", 1, "3" * 64)) == "unsupported_language"


def test_file_fingerprint_changes_with_path_content_and_options() -> None:
    repository_id = uuid.uuid4()
    entry = ManifestEntry("src/app.py", 4, "a" * 64)
    base = file_fingerprint(repository_id, entry, "profile", {"maxChunkSize": 1000})

    assert base == file_fingerprint(repository_id, entry, "profile", {"maxChunkSize": 1000})
    assert base != file_fingerprint(repository_id, entry, "profile", {"maxChunkSize": 500})
    assert base != file_fingerprint(
        repository_id,
        ManifestEntry("src/renamed.py", 4, "a" * 64),
        "profile",
        {"maxChunkSize": 1000},
    )


def test_engine_version_fits_persistence_column() -> None:
    column_type = ChunkedFile.__table__.c.engine_version.type
    assert isinstance(column_type, String)
    assert column_type.length is not None
    assert column_type.length >= len(ENGINE_VERSION)


def test_searchable_chunk_metadata_contains_structural_and_file_fields() -> None:
    document = searchable_chunk_metadata(
        "src/services/chat.py",
        "python",
        {"entities": ["reciprocal_rank_fusion"], "imports": ["sqlalchemy"]},
    )

    assert json.loads(document) == {
        "context": {
            "entities": ["reciprocal_rank_fusion"],
            "imports": ["sqlalchemy"],
        },
        "language": "python",
        "path": "src/services/chat.py",
    }


@pytest.mark.asyncio
async def test_worker_client_reassembles_streamed_chunks_and_accepts_large_records(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = "plain text\n"
    (workspace / "notes.txt").write_text(source)
    worker_script = tmp_path / "fake_worker.py"
    worker_script.write_text(
        "\n".join(
            [
                "import json, sys",
                "for line in sys.stdin:",
                "    request = json.loads(line)",
                "    base = {'protocolVersion': 3, 'requestId': request['requestId']}",
                "    chunk = {'index': 0, 'text': 'plain text\\n', "
                "'contextualizedText': 'x' * 100_000, 'context': {}}",
                "    print(json.dumps(base | {'type': 'chunk', 'path': 'notes.txt', "
                "'chunk': chunk}), flush=True)",
                "    print(json.dumps(base | {'type': 'file', 'path': 'notes.txt', "
                "'status': 'succeeded'}), flush=True)",
                f"    print(json.dumps(base | {{'type': 'complete', "
                f"'engineVersion': {ENGINE_VERSION!r}}}), flush=True)",
            ]
        )
    )
    client = ChunkingWorkerClient(sys.executable, worker_script)
    profile = ChunkingProfile.from_settings(Settings())
    entry = ManifestEntry("notes.txt", len(source.encode()), "a" * 64)

    try:
        events = [event async for event in client.chunk(workspace, (entry,), profile)]
    finally:
        await client.stop()

    assert len(events) == 1
    assert events[0]["status"] == "succeeded"
    chunks = events[0]["chunks"]
    assert isinstance(chunks, list)
    assert len(chunks) == 1
    assert len(str(chunks[0]["contextualizedText"])) == 100_000


@pytest.mark.asyncio
async def test_worker_client_restarts_after_an_oversized_protocol_record(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = "plain text\n"
    (workspace / "notes.txt").write_text(source)
    worker_script = tmp_path / "oversized_worker.py"
    worker_script.write_text(
        "\n".join(
            [
                "import json, sys",
                "for line in sys.stdin:",
                "    json.loads(line)",
                "    print(json.dumps({'padding': 'x' * 2_000_000}), flush=True)",
            ]
        )
    )
    client = ChunkingWorkerClient(sys.executable, worker_script)
    profile = ChunkingProfile.from_settings(Settings())
    entry = ManifestEntry("notes.txt", len(source.encode()), "a" * 64)

    with pytest.raises(ChunkingWorkerError, match="exceeded the reader limit"):
        _ = [event async for event in client.chunk(workspace, (entry,), profile)]

    assert client.restart_count == 1
    assert client._process is None
