from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

from app.chunking.profile import ChunkingProfile
from app.integrations.snapshots.base import ManifestEntry

logger = logging.getLogger(__name__)
ENGINE_VERSION = (
    "code-chunk@0.1.14+tree-sitter@0.26.3+tree-sitter-wasm@1.1.6+"
    "logical@2+text@2+overlap@2+quality@1"
)
PROTOCOL_VERSION = 3
WORKER_STREAM_LIMIT_BYTES = 1024 * 1024


class ChunkingWorkerError(Exception):
    pass


class ChunkingWorkerClient:
    def __init__(self, command: str, script: Path) -> None:
        self.command = command
        self.script = script
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._stderr_task: asyncio.Task[None] | None = None
        self.restart_count = 0

    async def start(self) -> None:
        if self._process is not None and self._process.returncode is None:
            return
        self._process = await asyncio.create_subprocess_exec(
            self.command,
            str(self.script),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=WORKER_STREAM_LIMIT_BYTES,
        )
        if self._process.stderr is not None:
            self._stderr_task = asyncio.create_task(self._drain_stderr(self._process.stderr))

    async def stop(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdin:
            process.stdin.close()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except TimeoutError:
            process.terminate()
            await process.wait()
        if self._stderr_task:
            await self._stderr_task
            self._stderr_task = None

    async def chunk(
        self,
        workspace: Path,
        files: tuple[ManifestEntry, ...],
        profile: ChunkingProfile,
    ) -> AsyncIterator[dict[str, object]]:
        async with self._lock:
            await self.start()
            process = self._process
            if process is None or process.stdin is None or process.stdout is None:
                raise ChunkingWorkerError("Chunking worker is unavailable")
            request_id = str(uuid.uuid4())
            request = {
                "protocolVersion": PROTOCOL_VERSION,
                "type": "chunk",
                "requestId": request_id,
                "workspace": str(workspace),
                "files": [entry.__dict__ for entry in files],
                "options": profile.options,
                "limits": profile.limits,
            }
            process.stdin.write(json.dumps(request, separators=(",", ":")).encode() + b"\n")
            await process.stdin.drain()
            try:
                chunks_by_path: dict[str, list[dict[str, object]]] = {}
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        raise ChunkingWorkerError("Chunking worker exited unexpectedly")
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ChunkingWorkerError("Chunking worker returned invalid JSON") from exc
                    if not isinstance(event, dict):
                        raise ChunkingWorkerError("Chunking worker returned invalid JSON")
                    if (
                        event.get("protocolVersion") != PROTOCOL_VERSION
                        or event.get("requestId") != request_id
                    ):
                        raise ChunkingWorkerError("Chunking worker protocol mismatch")
                    if event.get("type") == "fatal":
                        raise ChunkingWorkerError(str(event.get("error", "Chunking worker failed")))
                    if event.get("type") == "complete":
                        if event.get("engineVersion") != ENGINE_VERSION:
                            raise ChunkingWorkerError("Chunking worker engine version mismatch")
                        if chunks_by_path:
                            raise ChunkingWorkerError(
                                "Chunking worker returned incomplete file events"
                            )
                        break
                    if event.get("type") == "chunk":
                        path = event.get("path")
                        chunk = event.get("chunk")
                        if not isinstance(path, str) or not isinstance(chunk, dict):
                            raise ChunkingWorkerError(
                                "Chunking worker returned an invalid chunk event"
                            )
                        chunks_by_path.setdefault(path, []).append(chunk)
                        continue
                    if event.get("type") != "file":
                        raise ChunkingWorkerError("Chunking worker returned an unknown event")
                    path = event.get("path")
                    if not isinstance(path, str):
                        raise ChunkingWorkerError("Chunking worker returned an invalid file event")
                    chunks = chunks_by_path.pop(path, [])
                    if event.get("status") == "succeeded":
                        event["chunks"] = chunks
                    elif chunks:
                        raise ChunkingWorkerError(
                            "Chunking worker emitted chunks for a failed file"
                        )
                    yield event
            except ValueError as exc:
                await self._discard_process(process)
                raise ChunkingWorkerError(
                    "Chunking worker output record exceeded the reader limit"
                ) from exc
            except ChunkingWorkerError:
                await self._discard_process(process)
                raise

    async def _discard_process(self, process: asyncio.subprocess.Process) -> None:
        if self._process is process:
            self._process = None
        self.restart_count += 1
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
            await process.wait()
        if self._stderr_task:
            await self._stderr_task
            self._stderr_task = None

    async def _drain_stderr(self, stream: asyncio.StreamReader) -> None:
        while line := await stream.readline():
            logger.warning("Chunking worker: %s", line.decode(errors="replace").rstrip())
