from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

import pytest

from app.chunking.client import ChunkingWorkerClient
from app.chunking.profile import ChunkingProfile
from app.core.config import Settings
from app.integrations.snapshots.base import ManifestEntry

WORKER_PATH = Path(__file__).resolve().parents[4] / "chunker" / "dist" / "index.js"


@pytest.mark.skipif(not WORKER_PATH.exists(), reason="build the Node chunking worker first")
async def test_real_worker_handles_mixed_fallback_batch_and_large_output(tmp_path: Path) -> None:
    sources = {
        "src/native.py": "def answer() -> int:\n    return 42\n",
        "src/broken.cpp": "class Broken {\n  int answer( { return 42; }\n}\n",
        "src/Answer.cls": "public class Answer { public Integer value() { return 42; } }\n",
        "src/example.futurelang": "import core\n\nfunction answer() { return 42; }\n",
        "notes.txt": "".join(
            f"Paragraph {index} contains retrieval-oriented plain text.\n\n"
            for index in range(2_500)
        ),
    }
    entries: list[ManifestEntry] = []
    for path, source in sources.items():
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source)
        encoded = source.encode()
        entries.append(ManifestEntry(path, len(encoded), hashlib.sha256(encoded).hexdigest()))

    profile = ChunkingProfile.from_settings(Settings())
    worker = ChunkingWorkerClient("node", WORKER_PATH)
    try:
        events = [event async for event in worker.chunk(tmp_path, tuple(entries), profile)]
    finally:
        await worker.stop()

    assert len(events) == len(sources)
    by_path = {cast(str, event["path"]): event for event in events}
    assert all(event["status"] == "succeeded" for event in events)
    for path, source in sources.items():
        chunks = cast(list[dict[str, Any]], by_path[path]["chunks"])
        assert chunks
        reconstructed = "".join(cast(str, chunk["text"]) for chunk in chunks)
        if path == "src/native.py":
            assert reconstructed == source.rstrip()
        else:
            assert reconstructed == source
        assert all(
            len(cast(str, chunk["text"]).encode()) <= profile.max_chunk_size for chunk in chunks
        )

    broken = cast(list[dict[str, Any]], by_path["src/broken.cpp"]["chunks"])
    broken_context = cast(dict[str, Any], broken[0]["context"])
    assert broken_context["parser"]["name"] == "tree-sitter"
    assert broken_context["parseError"]["recoverable"] is True

    logical = cast(list[dict[str, Any]], by_path["src/Answer.cls"]["chunks"])
    assert logical[0]["context"]["parser"]["name"] == "logical-boundary"
    unknown = cast(list[dict[str, Any]], by_path["src/example.futurelang"]["chunks"])
    assert unknown[0]["context"]["sourceMetadata"]["detectedBy"] == "heuristic"
    plain = cast(list[dict[str, Any]], by_path["notes.txt"]["chunks"])
    assert len(plain) > 100
    assert plain[0]["context"]["parser"]["name"] == "plain-text"
