import { createInterface } from "node:readline";
import { lstat, readFile, realpath } from "node:fs/promises";
import { resolve, sep } from "node:path";
import { pathToFileURL } from "node:url";
import { stdin, stdout } from "node:process";
import {
  createChunker,
  detectLanguage,
  formatChunkWithContext,
  type Chunk,
  type ChunkOptions,
  type Chunker,
  type Language,
} from "code-chunk";
import {
  chunkLogically,
  chunkPlainText,
  chunkWithTreeSitter,
  detectFallbackLanguage,
  detectSource,
  selectLineOverlap,
  type CompatibleChunk,
  type DetectedSource,
} from "./tree-sitter-fallback.js";

export const PROTOCOL_VERSION = 3;
export const ENGINE_VERSION =
  "code-chunk@0.1.14+tree-sitter@0.26.3+tree-sitter-wasm@1.1.6+logical@2+text@2+overlap@2+quality@1";

type FileRequest = { path: string; size: number; sha256: string };
type WorkerOptions = Required<
  Pick<ChunkOptions, "maxChunkSize" | "contextMode" | "siblingDetail" | "filterImports">
> & { overlapRatio: number };

export type ChunkRequest = {
  protocolVersion: number;
  type: "chunk";
  requestId: string;
  workspace: string;
  files: FileRequest[];
  options: WorkerOptions;
  limits: {
    embeddingInputTokens: number;
    safetyMarginRatio: number;
    maxFileBytes: number;
    streamingFileBytes: number;
    batchBytes: number;
    concurrency: number;
    memoryBudgetBytes: number;
  };
};

type ChunkLanguage = Language | string;
type WorkerChunk = Chunk | CompatibleChunk;
type LoadedFile = FileRequest & {
  code: string;
  language: ChunkLanguage;
  engine: "code-chunk" | "tree-sitter" | "logical" | "plain-text";
  detectedSource?: DetectedSource;
};
type FileEvent = {
  protocolVersion: number;
  type: "file";
  requestId: string;
  path: string;
  status: "succeeded" | "skipped" | "failed";
  reason?: string;
  error?: string;
  language?: ChunkLanguage;
  resolvedOptions?: WorkerOptions;
  adaptiveRetries?: number;
  durationMs?: number;
};
type ChunkEvent = {
  protocolVersion: number;
  type: "chunk";
  requestId: string;
  path: string;
  chunk: ReturnType<typeof normalizeChunk>;
};

const chunkers = new Map<string, Chunker>();

function chunkerFor(options: WorkerOptions): Chunker {
  const key = JSON.stringify(options);
  const existing = chunkers.get(key);
  if (existing) return existing;
  const { overlapRatio: _overlapRatio, ...chunkOptions } = options;
  const created = createChunker({ ...chunkOptions, overlapLines: 0 });
  chunkers.set(key, created);
  return created;
}

function emit(event: object): void {
  stdout.write(`${JSON.stringify(event)}\n`);
}

function tokenCount(text: string): number {
  return Buffer.byteLength(text, "utf8");
}

function overlapTarget(options: WorkerOptions): number {
  return Math.round(options.maxChunkSize * options.overlapRatio);
}

function applyLineOverlap(chunks: WorkerChunk[], options: WorkerOptions): WorkerChunk[] {
  const target = overlapTarget(options);
  return chunks.map((chunk, index) => {
    const overlapText = index > 0 ? selectLineOverlap(chunks[index - 1]!.text, target) : undefined;
    if (!overlapText) return chunk;
    return {
      ...chunk,
      contextualizedText: formatChunkWithContext(
        chunk.text,
        chunk.context as unknown as Parameters<typeof formatChunkWithContext>[1],
        overlapText,
      ),
    };
  });
}

type ChunkingStrategy = "code-chunk" | "tree-sitter" | "logical-boundary" | "plain-text";

function chunkQuality(chunk: WorkerChunk) {
  const context = chunk.context as unknown as Record<string, unknown>;
  const parser =
    context.parser && typeof context.parser === "object"
      ? (context.parser as Record<string, unknown>)
      : undefined;
  const sourceMetadata =
    context.sourceMetadata && typeof context.sourceMetadata === "object"
      ? (context.sourceMetadata as Record<string, unknown>)
      : undefined;
  const strategy: ChunkingStrategy =
    parser?.name === "tree-sitter" ||
    parser?.name === "logical-boundary" ||
    parser?.name === "plain-text"
      ? parser.name
      : "code-chunk";
  const metadataChecks: [string, boolean][] = [
    ["filepath", typeof context.filepath === "string" && Boolean(context.filepath)],
    ["language", typeof context.language === "string" && Boolean(context.language)],
    ...["scope", "entities", "siblings", "imports"].map((key): [string, boolean] => [
      key,
      Array.isArray(context[key]),
    ]),
    ...(strategy === "code-chunk"
      ? []
      : ([
          ["parser.name", typeof parser?.name === "string" && Boolean(parser.name)],
          ["parser.language", typeof parser?.language === "string" && Boolean(parser.language)],
          [
            "sourceMetadata.name",
            typeof sourceMetadata?.name === "string" && Boolean(sourceMetadata.name),
          ],
          [
            "sourceMetadata.type",
            typeof sourceMetadata?.type === "string" && Boolean(sourceMetadata.type),
          ],
          [
            "sourceMetadata.detectedBy",
            typeof sourceMetadata?.detectedBy === "string" && Boolean(sourceMetadata.detectedBy),
          ],
        ] satisfies [string, boolean][])),
    ...(strategy === "tree-sitter"
      ? ([
          ["parser.grammar", typeof parser?.grammar === "string" && Boolean(parser.grammar)],
          ["parser.recovered", typeof parser?.recovered === "boolean"],
          ["syntaxNodes", Array.isArray(context.syntaxNodes)],
        ] satisfies [string, boolean][])
      : []),
  ];
  const missingMetadata = metadataChecks.filter(([, present]) => !present).map(([key]) => key);
  const metadataCompleteness =
    (metadataChecks.length - missingMetadata.length) / metadataChecks.length;
  const bands = {
    "code-chunk": { floor: 90, ceiling: 100, grade: "A" },
    "tree-sitter": { floor: 75, ceiling: 89, grade: "B" },
    "logical-boundary": { floor: 55, ceiling: 74, grade: "C" },
    "plain-text": { floor: 35, ceiling: 54, grade: "D" },
  } as const;
  const band = bands[strategy];
  let score = band.floor + Math.round((band.ceiling - band.floor) * metadataCompleteness);
  if (parser?.recovered === true) score = Math.max(band.floor, score - 5);
  if (typeof parser?.fallbackReason === "string" && parser.fallbackReason) {
    score = Math.max(band.floor, score - 3);
  }
  return {
    grade: band.grade,
    score,
    strategy,
    metadataCompleteness: Math.round(metadataCompleteness * 100) / 100,
    missingMetadata,
  };
}

function normalizeChunk(chunk: WorkerChunk) {
  return {
    index: chunk.index,
    text: chunk.text,
    contextualizedText: chunk.contextualizedText,
    byteRange: chunk.byteRange,
    lineRange: chunk.lineRange,
    context: chunk.context,
    tokenCount: tokenCount(chunk.contextualizedText),
    quality: chunkQuality(chunk),
  };
}

function withinBudget(chunks: WorkerChunk[], budget: number): boolean {
  return chunks.every((chunk) => tokenCount(chunk.contextualizedText) <= budget);
}

async function safeLoad(workspace: string, file: FileRequest): Promise<LoadedFile | FileEvent> {
  if (
    !file.path ||
    file.path.startsWith("/") ||
    file.path.includes("\\") ||
    file.path.split("/").includes("..")
  ) {
    return fileFailure(file.path, "skipped", "unsafe_path");
  }
  const candidate = resolve(workspace, file.path);
  const root = await realpath(workspace);
  const candidateStats = await lstat(candidate).catch(() => null);
  if (!candidateStats || candidateStats.isSymbolicLink() || !candidateStats.isFile()) {
    return fileFailure(file.path, "skipped", "unsafe_file_type");
  }
  const resolvedFile = await realpath(candidate).catch(() => "");
  if (!resolvedFile || !resolvedFile.startsWith(`${root}${sep}`)) {
    return fileFailure(file.path, "skipped", "unsafe_path");
  }
  if (candidateStats.size !== file.size)
    return fileFailure(file.path, "failed", "manifest_size_mismatch");
  const bytes = await readFile(resolvedFile);
  let code: string;
  try {
    code = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    return fileFailure(file.path, "skipped", "invalid_utf8");
  }
  if (!code.trim()) return fileFailure(file.path, "skipped", "empty_file");
  const header = code.split(/\r?\n/, 6).join("\n").toLowerCase();
  if (
    header.includes("@generated") ||
    header.includes("code generated") ||
    header.includes("do not edit")
  ) {
    return fileFailure(file.path, "skipped", "generated_file_marker");
  }
  if (
    code.length > 20_000 &&
    Math.max(...code.split(/\r?\n/).map((line) => line.length)) > 10_000
  ) {
    return fileFailure(file.path, "skipped", "minified_file");
  }
  const nativeLanguage = detectLanguage(file.path);
  const detectedSource = nativeLanguage ? null : detectSource(file.path, code);
  return {
    ...file,
    code,
    language: nativeLanguage ?? detectedSource!.id,
    engine: nativeLanguage
      ? "code-chunk"
      : detectedSource!.grammar
        ? "tree-sitter"
        : detectedSource!.detectedBy === "plain-text"
          ? "plain-text"
          : "logical",
    ...(detectedSource ? { detectedSource } : {}),
  };
}

function fileFailure(
  path: string,
  status: "skipped" | "failed",
  reason: string,
  error?: string,
): FileEvent {
  return {
    protocolVersion: PROTOCOL_VERSION,
    type: "file",
    requestId: "",
    path,
    status,
    reason,
    error,
  };
}

async function adaptChunks(
  file: LoadedFile,
  initial: WorkerChunk[],
  options: WorkerOptions,
  budget: number,
): Promise<{ chunks: WorkerChunk[]; options: WorkerOptions; retries: number }> {
  if (withinBudget(initial, budget)) return { chunks: initial, options, retries: 0 };
  const largest = Math.max(...initial.map((chunk) => tokenCount(chunk.contextualizedText)));
  const scaledOptions = {
    ...options,
    maxChunkSize: Math.max(128, Math.floor(options.maxChunkSize * (budget / largest) * 0.85)),
  };
  const scaled = await chunkFile(file, scaledOptions);
  if (withinBudget(scaled, budget)) return { chunks: scaled, options: scaledOptions, retries: 1 };
  const fallbackOptions: WorkerOptions = {
    ...scaledOptions,
    contextMode: "minimal",
    siblingDetail: "names",
    overlapRatio: 0,
  };
  const fallback = await chunkFile(file, fallbackOptions);
  if (!withinBudget(fallback, budget)) throw new Error("embedding_input_too_large");
  return { chunks: fallback, options: fallbackOptions, retries: 2 };
}

async function chunkFile(file: LoadedFile, options: WorkerOptions): Promise<WorkerChunk[]> {
  if (file.engine === "code-chunk") {
    return applyLineOverlap(await chunkerFor(options).chunk(file.path, file.code), options);
  }
  if (!file.detectedSource) throw new Error("missing_source_language_metadata");
  if (file.engine === "plain-text") {
    return chunkPlainText(file.path, file.code, file.detectedSource, options);
  }
  if (file.engine === "logical") {
    try {
      return chunkLogically(file.path, file.code, file.detectedSource, options);
    } catch (error) {
      return chunkPlainText(
        file.path,
        file.code,
        file.detectedSource,
        options,
        error instanceof Error ? error.message : "logical_chunking_failed",
      );
    }
  }
  try {
    return await chunkWithTreeSitter(file.path, file.code, file.detectedSource, options);
  } catch (error) {
    try {
      return chunkLogically(
        file.path,
        file.code,
        file.detectedSource,
        options,
        error instanceof Error ? error.message : "tree_sitter_chunking_failed",
      );
    } catch (logicalError) {
      const treeReason = error instanceof Error ? error.message : "tree_sitter_chunking_failed";
      const logicalReason =
        logicalError instanceof Error ? logicalError.message : "logical_chunking_failed";
      return chunkPlainText(
        file.path,
        file.code,
        file.detectedSource,
        options,
        `${treeReason}; ${logicalReason}`,
      );
    }
  }
}

async function emitSuccess(
  request: ChunkRequest,
  file: LoadedFile,
  chunks: WorkerChunk[],
  started: number,
): Promise<void> {
  const budget = Math.floor(
    request.limits.embeddingInputTokens * (1 - request.limits.safetyMarginRatio),
  );
  try {
    const adapted = await adaptChunks(file, chunks, request.options, budget);
    for (const chunk of adapted.chunks) {
      emit({
        protocolVersion: PROTOCOL_VERSION,
        type: "chunk",
        requestId: request.requestId,
        path: file.path,
        chunk: normalizeChunk(chunk),
      } satisfies ChunkEvent);
    }
    emit({
      protocolVersion: PROTOCOL_VERSION,
      type: "file",
      requestId: request.requestId,
      path: file.path,
      status: "succeeded",
      language: file.language,
      resolvedOptions: adapted.options,
      adaptiveRetries: adapted.retries,
      durationMs: Date.now() - started,
    } satisfies FileEvent);
  } catch (error) {
    emit({
      protocolVersion: PROTOCOL_VERSION,
      type: "file",
      requestId: request.requestId,
      path: file.path,
      status: "failed",
      reason: error instanceof Error ? error.message : "chunking_failed",
      durationMs: Date.now() - started,
    } satisfies FileEvent);
  }
}

async function processBatch(request: ChunkRequest, files: LoadedFile[]): Promise<void> {
  if (files.length === 0) return;
  const largest = Math.max(...files.map((file) => file.size));
  const concurrency = Math.max(
    1,
    Math.min(
      request.limits.concurrency,
      files.length,
      Math.floor(request.limits.memoryBudgetBytes / Math.max(1, largest)),
    ),
  );
  const byPath = new Map(files.map((file) => [file.path, file]));
  const started = new Map(files.map((file) => [file.path, Date.now()]));
  if (files[0]?.engine !== "code-chunk") {
    let next = 0;
    const workers = Array.from({ length: concurrency }, async () => {
      while (next < files.length) {
        const file = files[next++];
        if (!file) continue;
        try {
          await emitSuccess(
            request,
            file,
            await chunkFile(file, request.options),
            started.get(file.path) ?? Date.now(),
          );
        } catch (error) {
          emit({
            protocolVersion: PROTOCOL_VERSION,
            type: "file",
            requestId: request.requestId,
            path: file.path,
            status: "failed",
            reason: "chunking_failed",
            error: error instanceof Error ? error.message : undefined,
            durationMs: Date.now() - (started.get(file.path) ?? Date.now()),
          } satisfies FileEvent);
        }
      }
    });
    await Promise.all(workers);
    return;
  }
  for await (const result of chunkerFor(request.options).chunkBatchStream(
    files.map((file) => ({ filepath: file.path, code: file.code })),
    { concurrency },
  )) {
    const file = byPath.get(result.filepath);
    if (!file) continue;
    if (result.error) {
      emit({
        protocolVersion: PROTOCOL_VERSION,
        type: "file",
        requestId: request.requestId,
        path: result.filepath,
        status: "failed",
        reason: "chunking_failed",
        error: result.error.message,
        durationMs: Date.now() - (started.get(result.filepath) ?? Date.now()),
      } satisfies FileEvent);
    } else {
      await emitSuccess(
        request,
        file,
        applyLineOverlap(result.chunks, request.options),
        started.get(file.path) ?? Date.now(),
      );
    }
  }
}

async function processLarge(request: ChunkRequest, file: LoadedFile): Promise<void> {
  const started = Date.now();
  const chunks: WorkerChunk[] = [];
  try {
    if (file.engine !== "code-chunk") {
      chunks.push(...(await chunkFile(file, request.options)));
    } else {
      for await (const chunk of chunkerFor(request.options).stream(file.path, file.code)) {
        chunks.push(chunk);
      }
      chunks.splice(0, chunks.length, ...applyLineOverlap(chunks, request.options));
    }
    await emitSuccess(request, file, chunks, started);
  } catch (error) {
    emit({
      protocolVersion: PROTOCOL_VERSION,
      type: "file",
      requestId: request.requestId,
      path: file.path,
      status: "failed",
      reason: "chunking_failed",
      error: error instanceof Error ? error.message : undefined,
      durationMs: Date.now() - started,
    } satisfies FileEvent);
  }
}

async function loadBatch(request: ChunkRequest, files: FileRequest[]): Promise<LoadedFile[]> {
  const loaded: LoadedFile[] = [];
  for (const file of files) {
    try {
      const result = await safeLoad(request.workspace, file);
      if ("status" in result) emit({ ...result, requestId: request.requestId });
      else loaded.push(result);
    } catch (error) {
      emit({
        ...fileFailure(
          file.path,
          "failed",
          "file_read_failed",
          error instanceof Error ? error.message : undefined,
        ),
        requestId: request.requestId,
      });
    }
  }
  return loaded;
}

export async function processRequest(request: ChunkRequest): Promise<void> {
  if (request.protocolVersion !== PROTOCOL_VERSION || request.type !== "chunk") {
    throw new Error("unsupported_protocol");
  }
  const files = [...request.files].sort((a, b) => {
    const languageOrder = (
      detectLanguage(a.path) ??
      detectFallbackLanguage(a.path)?.id ??
      "zz"
    ).localeCompare(detectLanguage(b.path) ?? detectFallbackLanguage(b.path)?.id ?? "zz");
    return languageOrder || a.size - b.size || a.path.localeCompare(b.path);
  });
  const candidates: FileRequest[] = [];
  for (const file of files) {
    if (file.size > request.limits.maxFileBytes) {
      const event = fileFailure(file.path, "skipped", "file_too_large_for_chunking");
      emit({ ...event, requestId: request.requestId });
      continue;
    }
    candidates.push(file);
  }

  const ordinary = candidates.filter((file) => file.size < request.limits.streamingFileBytes);
  const large = candidates.filter((file) => file.size >= request.limits.streamingFileBytes);
  let batch: FileRequest[] = [];
  let batchBytes = 0;
  let batchLanguage: ChunkLanguage | null = null;
  for (const file of ordinary) {
    const language = detectLanguage(file.path) ?? detectFallbackLanguage(file.path)?.id ?? null;
    if (
      batch.length > 0 &&
      (batchBytes + file.size > request.limits.batchBytes || language !== batchLanguage)
    ) {
      await processBatch(request, await loadBatch(request, batch));
      batch = [];
      batchBytes = 0;
    }
    batchLanguage = language;
    batch.push(file);
    batchBytes += file.size;
  }
  await processBatch(request, await loadBatch(request, batch));
  for (const file of large) {
    const loaded = await loadBatch(request, [file]);
    if (loaded[0]) await processLarge(request, loaded[0]);
  }
  emit({
    protocolVersion: PROTOCOL_VERSION,
    type: "complete",
    requestId: request.requestId,
    engineVersion: ENGINE_VERSION,
  });
}

async function main(): Promise<void> {
  const lines = createInterface({ input: stdin, crlfDelay: Infinity });
  for await (const line of lines) {
    if (!line.trim()) continue;
    try {
      await processRequest(JSON.parse(line) as ChunkRequest);
    } catch (error) {
      const requestId = (() => {
        try {
          return (JSON.parse(line) as { requestId?: string }).requestId ?? "unknown";
        } catch {
          return "unknown";
        }
      })();
      emit({
        protocolVersion: PROTOCOL_VERSION,
        type: "fatal",
        requestId,
        error: error instanceof Error ? error.message : "worker_failure",
      });
    }
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  void main();
}
