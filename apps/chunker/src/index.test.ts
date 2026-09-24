import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import { chunk } from "code-chunk";
import { getWasmPath, type SupportedLanguage } from "tree-sitter-wasm";
import {
  chunkLogically,
  chunkPlainText,
  chunkWithTreeSitter,
  detectFallbackLanguage,
  detectSource,
  selectLineOverlap,
} from "./tree-sitter-fallback.js";

const options = {
  maxChunkSize: 1024,
  contextMode: "full" as const,
  siblingDetail: "signatures" as const,
  filterImports: false,
  overlapRatio: 0.2,
};

const fixtures = [
  [
    "src/user.ts",
    "import { db } from './db'\nexport class User { async find(id: string) { return db.get(id) } }\n",
  ],
  ["src/user.js", "import db from './db.js'\nexport function find(id) { return db.get(id) }\n"],
  [
    "src/user.py",
    "from db import get\n\nclass User:\n    def find(self, id: str):\n        return get(id)\n",
  ],
  ["src/user.rs", "pub struct User;\nimpl User { pub fn find(&self, id: u64) -> u64 { id } }\n"],
  [
    "src/user.go",
    "package user\ntype User struct{}\nfunc (u User) Find(id int) int { return id }\n",
  ],
  ["src/User.java", "package user;\nclass User { int find(int id) { return id; } }\n"],
] as const;

test("code-chunk returns embedding context for every supported language", async () => {
  for (const [path, source] of fixtures) {
    const chunks = await chunk(path, source, { ...options, overlapLines: 0 });
    assert.ok(chunks.length > 0, path);
    assert.match(chunks[0]!.contextualizedText, new RegExp(path.replace(".", "\\.")));
    assert.equal(chunks[0]!.context.filepath, path);
    assert.ok(chunks[0]!.lineRange.start >= 0);
  }
});

test("overlap targets twenty percent using complete trailing lines", () => {
  const previous = "first line has 24 bytes\nsecond line has 25 bytes\nthird line has 24 bytes\n";
  const overlap = selectLineOverlap(previous, 48);

  assert.equal(overlap, "second line has 25 bytes\nthird line has 24 bytes\n");
  assert.ok(overlap?.endsWith("\n"));
});

test("tree-sitter fallback packs adjacent semantic blocks into compatible chunks", async () => {
  const source = [
    "#include <stdio.h>",
    "int first(void) { return 1; }",
    "int second(void) { return first() + 1; }",
    "",
  ].join("\n");
  const language = detectFallbackLanguage("src/example.c");
  assert.ok(language?.grammar);
  const chunks = await chunkWithTreeSitter("src/example.c", source, language, options);

  assert.equal(chunks.length, 1);
  assert.equal(chunks[0]?.text, source);
  assert.equal(chunks[0]?.context.filepath, "src/example.c");
  assert.equal(chunks[0]?.context.language, "c");
  assert.equal(chunks[0]?.context.parser?.name, "tree-sitter");
  assert.ok((chunks[0]?.context.syntaxNodes?.length ?? 0) > 0);
  assert.ok(chunks[0]?.context.entities.some((entity) => entity.name === "first"));
  assert.match(chunks[0]?.contextualizedText ?? "", /# src\/example\.c/);
});

test("every registered fallback grammar parses and emits the shared contract", async () => {
  const fallbackFixtures = [
    ["script.sh", "bash", "hello() { echo hello; }\n"],
    ["main.c", "c", "int main(void) { return 0; }\n"],
    ["main.cpp", "cpp", "class Main { public: int run() { return 0; } };\n"],
    ["Program.cs", "csharp", "class Program { static int Main() { return 0; } }\n"],
    ["index.php", "php", "<?php function hello() { return 'hello'; }\n"],
    ["service.rb", "ruby", "class Service\n  def call\n    1\n  end\nend\n"],
  ] as const;

  for (const [path, languageId, source] of fallbackFixtures) {
    const language = detectFallbackLanguage(path);
    assert.equal(language?.id, languageId);
    assert.ok(language?.grammar);
    const chunks = await chunkWithTreeSitter(path, source, language, options);
    assert.ok(chunks.length > 0, path);
    assert.equal(chunks[0]?.context.language, languageId);
    assert.equal(chunks[0]?.context.parser?.recovered, false);
    assert.ok(chunks[0]?.byteRange.end <= Buffer.byteLength(source), path);
  }
  assert.equal(detectFallbackLanguage("Gemfile")?.id, "ruby");
  assert.equal(detectFallbackLanguage("bin/tool", "#!/usr/bin/env elixir\n")?.id, "elixir");
  assert.equal(detectFallbackLanguage("README.md"), null);
});

test("all 112 declared tree-sitter grammar assets are bundled and non-empty", async () => {
  const packageEntry = import.meta.resolve("tree-sitter-wasm");
  const manifest = JSON.parse(
    await readFile(join(dirname(fileURLToPath(packageEntry)), "manifest.json"), "utf8"),
  ) as Record<SupportedLanguage, string[]>;
  assert.equal(Object.keys(manifest).length, 112);
  for (const grammar of Object.keys(manifest) as SupportedLanguage[]) {
    assert.ok((await stat(getWasmPath(grammar))).size > 0, grammar);
  }
});

test("the broad grammar registry handles languages outside code-chunk", async () => {
  const broadFixtures = [
    ["Main.kt", "kotlin", "class Main { fun answer(): Int = 42 }\n"],
    ["Main.swift", "swift", "struct Main { func answer() -> Int { 42 } }\n"],
    ["main.hs", "haskell", "answer :: Int\nanswer = 42\n"],
    ["schema.graphql", "graphql", "type Query { answer: Int! }\n"],
    [
      "main.sol",
      "solidity",
      "contract Main { function answer() public pure returns (uint) { return 42; } }\n",
    ],
  ] as const;

  for (const [path, languageId, source] of broadFixtures) {
    const language = detectFallbackLanguage(path);
    assert.equal(language?.id, languageId);
    assert.ok(language?.grammar, path);
    const chunks = await chunkWithTreeSitter(path, source, language, options);
    assert.equal(chunks.map((item) => item.text).join(""), source);
    assert.equal(chunks[0]?.context.parser?.name, "tree-sitter");
    assert.equal(chunks[0]?.context.sourceMetadata?.name, language.name);
  }
});

test("recognized source without a bundled grammar uses enriched logical boundaries", () => {
  const source = [
    "public class Example {",
    "  public Integer first() { return 1; }",
    "}",
    "",
    "public class Other {",
    "  public Integer second() { return 2; }",
    "}",
    "",
  ].join("\n");
  const language = detectFallbackLanguage("Example.cls");
  assert.equal(language?.name, "Apex");
  assert.equal(language?.grammar, null);
  const chunks = chunkLogically("Example.cls", source, language, {
    ...options,
    maxChunkSize: 90,
    overlapRatio: 0,
  });

  assert.ok(chunks.length > 1);
  assert.ok(chunks.every((item) => Buffer.byteLength(item.text) <= 90));
  assert.equal(chunks.map((item) => item.text).join(""), source);
  assert.equal(chunks[0]?.context.parser?.name, "logical-boundary");
  assert.ok(
    chunks.some((item) => item.context.entities.some((entity) => entity.name === "Example")),
  );
});

test("code-like files unknown to Linguist still receive logical chunks", () => {
  const source = "import core\n\nfunction answer() { return 42; }\n";
  const language = detectSource("example.futurelang", source);
  assert.equal(language?.detectedBy, "heuristic");
  assert.equal(language?.grammar, null);
  const chunks = chunkLogically("example.futurelang", source, language, options);
  assert.equal(chunks.map((item) => item.text).join(""), source);
  assert.ok(
    chunks.some((item) => item.context.entities.some((entity) => entity.name === "answer")),
  );
});

test("all remaining UTF-8 files receive paragraph-aware plain text chunks", () => {
  const source = [
    "Project notes for the next release.",
    "This paragraph stays together.",
    "",
    "A second paragraph contains retrieval details.",
    "It is preserved without invented code metadata.",
    "",
  ].join("\n");
  const language = detectSource("notes.txt", source);
  assert.equal(language.detectedBy, "plain-text");
  const chunks = chunkPlainText("notes.txt", source, language, {
    ...options,
    maxChunkSize: 90,
    overlapRatio: 0,
  });

  assert.ok(chunks.length > 1);
  assert.ok(chunks.every((item) => Buffer.byteLength(item.text) <= 90));
  assert.equal(chunks.map((item) => item.text).join(""), source);
  assert.equal(chunks[0]?.context.parser?.name, "plain-text");
  assert.deepEqual(chunks[0]?.context.entities, []);
  assert.equal(chunks[0]?.context.sourceMetadata?.type, "text");
});

test("tree-sitter fallback splits oversized blocks at AST boundaries within the byte limit", async () => {
  const methods = Array.from(
    { length: 8 },
    (_, index) => `  int method${index}(void) { return ${index}; }`,
  ).join("\n");
  const source = `class Large {\n${methods}\n};\n// λ\n`;
  const constrained = { ...options, maxChunkSize: 100, overlapRatio: 0 };
  const language = detectFallbackLanguage("src/large.cpp");
  assert.ok(language?.grammar);
  const chunks = await chunkWithTreeSitter("src/large.cpp", source, language, constrained);

  assert.ok(chunks.length > 1);
  assert.ok(chunks.every((item) => Buffer.byteLength(item.text, "utf8") <= 100));
  assert.equal(chunks.map((item) => item.text).join(""), source);
  assert.deepEqual(
    chunks.map((item) => item.index),
    chunks.map((_, index) => index),
  );
  assert.ok(chunks.some((item) => item.context.scope.some((entity) => entity.name === "Large")));
});

test("tree-sitter preserves malformed source and Unicode while reporting parser recovery", async () => {
  const source = "class Broken {\n  int value( { return 42; }\n}\n// λ survives\n";
  const language = detectFallbackLanguage("src/broken.cpp");
  assert.ok(language?.grammar);
  const chunks = await chunkWithTreeSitter("src/broken.cpp", source, language, {
    ...options,
    maxChunkSize: 48,
    overlapRatio: 0,
  });

  assert.equal(chunks.map((item) => item.text).join(""), source);
  assert.ok(chunks.every((item) => Buffer.byteLength(item.text) <= 48));
  assert.ok(chunks.some((item) => item.context.parseError?.recoverable));
  assert.ok(chunks.some((item) => item.context.parser?.recovered));
});

test("worker streams normalized results and rejects path traversal", async () => {
  const workspace = await mkdtemp(join(tmpdir(), "showcasehost-chunker-"));
  const pythonSource = "def hello(name: str):\n    return f'hello {name}'\n";
  const cppSource = "int answer(void) { return 42; }\n";
  const apexSource = "public class Answer { public Integer value() { return 42; } }\n";
  await writeFile(join(workspace, "app.py"), pythonSource);
  await writeFile(join(workspace, "answer.cpp"), cppSource);
  await writeFile(join(workspace, "Answer.cls"), apexSource);
  const child = spawn(process.execPath, [new URL("./index.js", import.meta.url).pathname], {
    stdio: ["pipe", "pipe", "pipe"],
  });
  const request = {
    protocolVersion: 3,
    type: "chunk",
    requestId: "test-request",
    workspace,
    files: [
      { path: "app.py", size: Buffer.byteLength(pythonSource), sha256: "a".repeat(64) },
      { path: "answer.cpp", size: Buffer.byteLength(cppSource), sha256: "c".repeat(64) },
      { path: "Answer.cls", size: Buffer.byteLength(apexSource), sha256: "d".repeat(64) },
      { path: "../escape.py", size: 1, sha256: "b".repeat(64) },
    ],
    options,
    limits: {
      embeddingInputTokens: 8192,
      safetyMarginRatio: 0.1,
      maxFileBytes: 1024 * 1024,
      streamingFileBytes: 1024,
      batchBytes: 1024 * 1024,
      concurrency: 2,
      memoryBudgetBytes: 8 * 1024 * 1024,
    },
  };
  child.stdin.write(`${JSON.stringify(request)}\n`);
  child.stdin.end();
  let output = "";
  for await (const data of child.stdout) output += data.toString();
  const exit = await new Promise<number | null>((resolve) => child.on("exit", resolve));
  await rm(workspace, { recursive: true, force: true });

  assert.equal(exit, 0);
  const events = output
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line) as Record<string, unknown>);
  assert.equal(
    events.find((event) => event.type === "file" && event.path === "app.py")?.status,
    "succeeded",
  );
  const fallback = events.find((event) => event.type === "file" && event.path === "answer.cpp");
  assert.equal(fallback?.status, "succeeded");
  assert.equal(fallback?.language, "cpp");
  assert.equal(
    events.filter((event) => event.type === "chunk" && event.path === "answer.cpp").length,
    1,
  );
  const logical = events.find((event) => event.type === "file" && event.path === "Answer.cls");
  assert.equal(logical?.status, "succeeded");
  assert.equal(logical?.language, "apex");
  const logicalChunk = events.find((event) => event.type === "chunk" && event.path === "Answer.cls")
    ?.chunk as {
    context: { parser: { name: string } };
    quality: { grade: string; score: number; strategy: string };
  };
  assert.equal(logicalChunk.context.parser.name, "logical-boundary");
  assert.equal(logicalChunk.quality.grade, "C");
  assert.equal(logicalChunk.quality.score, 74);
  assert.equal(logicalChunk.quality.strategy, "logical-boundary");
  const nativeChunk = events.find((event) => event.type === "chunk" && event.path === "app.py")
    ?.chunk as { quality: { grade: string; score: number } };
  const treeChunk = events.find((event) => event.type === "chunk" && event.path === "answer.cpp")
    ?.chunk as { quality: { grade: string; score: number } };
  assert.equal(nativeChunk.quality.grade, "A");
  assert.equal(nativeChunk.quality.score, 100);
  assert.equal(treeChunk.quality.grade, "B");
  assert.equal(treeChunk.quality.score, 89);
  assert.equal(events.find((event) => event.path === "../escape.py")?.reason, "unsafe_path");
  assert.equal(events.at(-1)?.type, "complete");
});

test("worker keeps large-file protocol records below the subprocess reader limit", async () => {
  const workspace = await mkdtemp(join(tmpdir(), "showcasehost-chunker-large-"));
  const source = Array.from(
    { length: 2_500 },
    (_, index) =>
      `Paragraph ${index} contains enough retrieval text to exercise streamed chunks.\n\n`,
  ).join("");
  await writeFile(join(workspace, "notes.txt"), source);
  const child = spawn(process.execPath, [new URL("./index.js", import.meta.url).pathname], {
    stdio: ["pipe", "pipe", "pipe"],
  });
  child.stdin.end(
    `${JSON.stringify({
      protocolVersion: 3,
      type: "chunk",
      requestId: "large-request",
      workspace,
      files: [{ path: "notes.txt", size: Buffer.byteLength(source), sha256: "e".repeat(64) }],
      options: { ...options, maxChunkSize: 1024 },
      limits: {
        embeddingInputTokens: 8192,
        safetyMarginRatio: 0.1,
        maxFileBytes: 1024 * 1024,
        streamingFileBytes: 1024,
        batchBytes: 1024 * 1024,
        concurrency: 2,
        memoryBudgetBytes: 8 * 1024 * 1024,
      },
    })}\n`,
  );
  let output = "";
  for await (const data of child.stdout) output += data.toString();
  const exit = await new Promise<number | null>((resolve) => child.on("exit", resolve));
  await rm(workspace, { recursive: true, force: true });

  assert.equal(exit, 0);
  const lines = output.trim().split("\n");
  assert.ok(lines.length > 100);
  assert.ok(lines.every((line) => Buffer.byteLength(line) < 64 * 1024));
  const events = lines.map((line) => JSON.parse(line) as Record<string, unknown>);
  const chunkEvents = events.filter((event) => event.type === "chunk");
  assert.ok(chunkEvents.length > 100);
  const firstChunk = chunkEvents[0]?.chunk as { quality: { grade: string; score: number } };
  assert.equal(firstChunk.quality.grade, "D");
  assert.equal(firstChunk.quality.score, 54);
  assert.equal(events.find((event) => event.type === "file")?.status, "succeeded");
  assert.equal(events.at(-1)?.type, "complete");
});
