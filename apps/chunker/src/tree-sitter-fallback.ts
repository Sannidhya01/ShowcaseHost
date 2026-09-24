import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import * as linguistLanguages from "linguist-languages";
import type { Language as LinguistLanguage } from "linguist-languages";
import { getAvailableQueries, getWasmPath, type SupportedLanguage } from "tree-sitter-wasm";
import { Language, Parser, Query, type Node as SyntaxNode } from "web-tree-sitter";

export type DetectedSource = {
  id: string;
  name: string;
  type: string;
  grammar: SupportedLanguage | null;
  detectedBy: "extension" | "filename" | "shebang" | "heuristic" | "plain-text";
};

const SOURCE_TYPES = new Set(["programming", "markup", "data"]);
const GRAMMAR_ALIASES: Readonly<Record<string, SupportedLanguage>> = {
  "C#": "c_sharp",
  "C++": "cpp",
  "Common Lisp": "commonlisp",
  "Emacs Lisp": "elisp",
  "Objective-C": "objc",
  "OCaml Interface": "ocaml_interface",
  "Protocol Buffer": "proto",
  R: "r",
  Shell: "bash",
  Assembly: "asm",
  Makefile: "make",
  Dockerfile: "dockerfile",
  HCL: "hcl",
  Verilog: "systemverilog",
};
const LANGUAGE_IDS: Readonly<Record<string, string>> = {
  "C#": "csharp",
  "C++": "cpp",
  "Objective-C": "objective-c",
  Shell: "bash",
};

const languages = (Object.values(linguistLanguages) as LinguistLanguage[]).filter((value) =>
  SOURCE_TYPES.has(value.type),
);
const languagesByName = new Map(languages.map((language) => [language.name, language]));
const languagesByExtension = new Map<string, LinguistLanguage[]>();
const languagesByFilename = new Map<string, LinguistLanguage[]>();
for (const language of languages) {
  for (const extension of language.extensions ?? []) {
    const candidates = languagesByExtension.get(extension.toLowerCase()) ?? [];
    candidates.push(language);
    languagesByExtension.set(extension.toLowerCase(), candidates);
  }
  for (const filename of language.filenames ?? []) {
    const candidates = languagesByFilename.get(filename.toLowerCase()) ?? [];
    candidates.push(language);
    languagesByFilename.set(filename.toLowerCase(), candidates);
  }
}

const EXTENSION_PREFERENCES: Readonly<Record<string, string>> = {
  ".h": "C",
  ".cls": "Apex",
  ".fs": "F#",
  ".json": "JSON",
  ".m": "Objective-C",
  ".php": "PHP",
  ".pl": "Perl",
  ".r": "R",
  ".rs": "Rust",
  ".sql": "SQL",
  ".v": "Verilog",
  ".yaml": "YAML",
  ".yml": "YAML",
};
const PLAIN_TEXT_EXTENSIONS = [".md", ".mdx", ".rst", ".txt", ".adoc", ".asciidoc"];

export type CompatibleEntityType =
  "function" | "method" | "class" | "interface" | "type" | "enum" | "import" | "export";

type EntityInfo = {
  name: string;
  type: CompatibleEntityType;
  signature?: string;
  docstring?: string | null;
  lineRange?: { start: number; end: number };
  isPartial?: boolean;
};

export type CompatibleChunk = {
  text: string;
  contextualizedText: string;
  byteRange: { start: number; end: number };
  lineRange: { start: number; end: number };
  context: {
    filepath?: string;
    language?: string;
    scope: EntityInfo[];
    entities: EntityInfo[];
    siblings: {
      name: string;
      type: CompatibleEntityType;
      position: "before" | "after";
      distance: number;
    }[];
    imports: { name: string; source: string }[];
    parseError?: { message: string; recoverable: boolean };
    parser?: {
      name: "tree-sitter" | "logical-boundary" | "plain-text";
      language: string;
      grammar?: string;
      recovered?: boolean;
      fallbackReason?: string;
    };
    sourceMetadata?: {
      name: string;
      type: string;
      detectedBy: DetectedSource["detectedBy"];
    };
    syntaxNodes?: {
      type: string;
      byteRange: { start: number; end: number };
      lineRange: { start: number; end: number };
      named: boolean;
      missing: boolean;
      hasError: boolean;
    }[];
  };
  index: number;
  totalChunks: number;
};

export type FallbackOptions = {
  maxChunkSize: number;
  contextMode: "none" | "minimal" | "full";
  siblingDetail: "none" | "names" | "signatures";
  filterImports: boolean;
  overlapRatio: number;
};

type Span = { start: number; end: number };
type ExtractedEntity = EntityInfo & {
  start: number;
  end: number;
  node: SyntaxNode;
};

const languageCache = new Map<SupportedLanguage, Promise<Language>>();
const grammarCache = new Map<string, SupportedLanguage | null>();
const tagQuerySourceCache = new Map<SupportedLanguage, Promise<string | null>>();
let parserInitialization: Promise<void> | undefined;

function basename(filepath: string): string {
  return filepath.split("/").at(-1) ?? filepath;
}

function normalizeIdentifier(value: string): string {
  return value
    .toLowerCase()
    .replace(/\+\+/g, "pp")
    .replace(/#/g, "sharp")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

function grammarFor(language: LinguistLanguage): SupportedLanguage | null {
  if (grammarCache.has(language.name)) return grammarCache.get(language.name) ?? null;
  const alias = GRAMMAR_ALIASES[language.name];
  if (alias && existsSync(getWasmPath(alias))) {
    grammarCache.set(language.name, alias);
    return alias;
  }
  const candidates = [language.name, language.aceMode, ...(language.aliases ?? [])]
    .flatMap((value) => {
      const underscored = value
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "_")
        .replace(/^_|_$/g, "");
      return [underscored, underscored.replaceAll("_", "")];
    })
    .filter(Boolean);
  const grammar =
    (candidates.find((candidate) => existsSync(getWasmPath(candidate as SupportedLanguage))) as
      SupportedLanguage | undefined) ?? null;
  grammarCache.set(language.name, grammar);
  return grammar;
}

function chooseLanguage(
  candidates: LinguistLanguage[],
  extension?: string,
): LinguistLanguage | null {
  const preferredName = extension ? EXTENSION_PREFERENCES[extension] : undefined;
  const preferred = preferredName ? candidates.find((item) => item.name === preferredName) : null;
  if (preferred) return preferred;
  return (
    [...candidates].sort((left, right) => {
      const grammarOrder = Number(Boolean(grammarFor(right))) - Number(Boolean(grammarFor(left)));
      const programmingOrder =
        Number(right.type === "programming") - Number(left.type === "programming");
      return grammarOrder || programmingOrder || left.name.localeCompare(right.name);
    })[0] ?? null
  );
}

function detected(
  language: LinguistLanguage,
  detectedBy: DetectedSource["detectedBy"],
): DetectedSource {
  return {
    id: LANGUAGE_IDS[language.name] ?? normalizeIdentifier(language.name),
    name: language.name,
    type: language.type,
    grammar: grammarFor(language),
    detectedBy,
  };
}

export function detectFallbackLanguage(filepath: string, source?: string): DetectedSource | null {
  const name = basename(filepath).toLowerCase();
  if (/^(?:readme|license|licence|changelog|contributing|authors|notice)(?:\.|$)/.test(name)) {
    return null;
  }
  const filenameMatch = chooseLanguage(languagesByFilename.get(name) ?? []);
  if (filenameMatch) return detected(filenameMatch, "filename");
  if (name.startsWith("dockerfile"))
    return detected(languagesByName.get("Dockerfile")!, "filename");
  const extensions = [...languagesByExtension.keys()]
    .filter((extension) => filepath.toLowerCase().endsWith(extension))
    .sort((left, right) => right.length - left.length);
  const extension = extensions[0];
  if (extension) {
    const extensionMatch = chooseLanguage(languagesByExtension.get(extension) ?? [], extension);
    if (extensionMatch) return detected(extensionMatch, "extension");
  }
  const shebang = source
    ?.split(/\r?\n/, 1)[0]
    ?.match(/^#!\s*(?:\/usr\/bin\/env\s+)?([^\s/]+)(?:\s|$)/);
  if (shebang?.[1]) {
    const interpreter = shebang[1].replace(/^.*\//, "").toLowerCase();
    const interpreterMatch = languages.find((language) =>
      language.interpreters?.some((candidate) => candidate.toLowerCase() === interpreter),
    );
    if (interpreterMatch) return detected(interpreterMatch, "shebang");
  }
  return null;
}

export function detectSource(filepath: string, source: string): DetectedSource {
  if (PLAIN_TEXT_EXTENSIONS.some((extension) => filepath.toLowerCase().endsWith(extension))) {
    return {
      id: "text",
      name: "Plain text",
      type: "text",
      grammar: null,
      detectedBy: "plain-text",
    };
  }
  const known = detectFallbackLanguage(filepath, source);
  if (known) return known;
  const lines = source.split(/\r?\n/).slice(0, 200);
  const codeSignals = [
    lines.some((line) => LOGICAL_DECLARATION.test(line)),
    lines.some((line) => LOGICAL_IMPORT.test(line)),
    /[({][\s\S]*[)}]/.test(source) && /[;:=]/.test(source),
    /^#!\s*\//.test(source),
  ].filter(Boolean).length;
  if (codeSignals >= 2) {
    const name = basename(filepath);
    const extension = name.includes(".") ? name.slice(name.lastIndexOf(".") + 1) : "source";
    const id = normalizeIdentifier(extension) || "source";
    return {
      id,
      name: `Unknown source (${id})`,
      type: "programming",
      grammar: null,
      detectedBy: "heuristic",
    };
  }
  return {
    id: "text",
    name: "Plain text",
    type: "text",
    grammar: null,
    detectedBy: "plain-text",
  };
}

async function loadLanguage(grammar: SupportedLanguage): Promise<Language> {
  const cached = languageCache.get(grammar);
  if (cached) return cached;
  const loading = (async () => {
    parserInitialization ??= Parser.init({
      locateFile: () => fileURLToPath(import.meta.resolve("web-tree-sitter/web-tree-sitter.wasm")),
    });
    await parserInitialization;
    return Language.load(getWasmPath(grammar));
  })();
  languageCache.set(grammar, loading);
  return loading;
}

function splitAtLines(source: Buffer, start: number, end: number, maxSize: number): Span[] {
  const spans: Span[] = [];
  let cursor = start;
  while (cursor < end) {
    const limit = Math.min(cursor + maxSize, end);
    if (limit === end) {
      spans.push({ start: cursor, end });
      break;
    }
    const newline = source.lastIndexOf(0x0a, limit - 1);
    let split = newline >= cursor ? newline + 1 : limit;
    while (split > cursor && split < end && (source[split]! & 0xc0) === 0x80) split--;
    if (split === cursor) split = limit;
    spans.push({ start: cursor, end: split });
    cursor = split;
  }
  return spans;
}

function structuralUnits(node: SyntaxNode, source: Buffer, maxSize: number): Span[] {
  if (node.endIndex - node.startIndex <= maxSize) {
    return [{ start: node.startIndex, end: node.endIndex }];
  }
  const children = node.namedChildren.filter(
    (child) => child.endIndex > child.startIndex && child.startIndex >= node.startIndex,
  );
  if (
    children.length === 0 ||
    (children.length === 1 &&
      children[0]?.startIndex === node.startIndex &&
      children[0]?.endIndex === node.endIndex)
  ) {
    return splitAtLines(source, node.startIndex, node.endIndex, maxSize);
  }

  const units: Span[] = [];
  let cursor = node.startIndex;
  for (const child of children) {
    const childUnits = structuralUnits(child, source, maxSize);
    if (child.startIndex > cursor && childUnits[0]) {
      if (childUnits[0].end - cursor <= maxSize) {
        childUnits[0].start = cursor;
      } else {
        units.push(...splitAtLines(source, cursor, child.startIndex, maxSize));
      }
    }
    units.push(...childUnits);
    cursor = Math.max(cursor, child.endIndex);
  }
  if (cursor < node.endIndex) {
    const last = units.at(-1);
    if (last && node.endIndex - last.start <= maxSize) last.end = node.endIndex;
    else units.push(...splitAtLines(source, cursor, node.endIndex, maxSize));
  }
  return units;
}

function mergeAdjacent(units: Span[], maxSize: number): Span[] {
  const merged: Span[] = [];
  for (const unit of units) {
    if (unit.end <= unit.start) continue;
    const previous = merged.at(-1);
    if (previous && unit.start === previous.end && unit.end - previous.start <= maxSize) {
      previous.end = unit.end;
    } else {
      merged.push({ ...unit });
    }
  }
  return merged;
}

function lineStarts(source: Buffer): number[] {
  const starts = [0];
  for (let index = source.indexOf(0x0a); index >= 0; index = source.indexOf(0x0a, index + 1)) {
    starts.push(index + 1);
  }
  return starts;
}

function lineAt(starts: number[], byteIndex: number): number {
  let low = 0;
  let high = starts.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (starts[middle]! <= byteIndex) low = middle + 1;
    else high = middle;
  }
  return Math.max(0, low - 1);
}

function classifyEntity(node: SyntaxNode): CompatibleEntityType | null {
  const type = node.type.toLowerCase();
  if (type.includes("import") || type === "using_directive" || type === "require") return "import";
  if (type.includes("export")) return "export";
  if (type.includes("interface") || type === "protocol_declaration") return "interface";
  if (type.includes("class") || type.includes("struct") || type.includes("module")) return "class";
  if (type.includes("enum")) return "enum";
  if (type.includes("type") && type.includes("declaration")) return "type";
  if (type.includes("method") || type === "constructor_declaration") return "method";
  if (
    (type.includes("function") || type === "function_definition") &&
    !type.includes("declarator") &&
    !type.includes("expression") &&
    !type.includes("call")
  ) {
    return "function";
  }
  return null;
}

function nodeText(source: Buffer, node: SyntaxNode): string {
  return source.subarray(node.startIndex, node.endIndex).toString("utf8");
}

function entityName(node: SyntaxNode, source: Buffer): string {
  const named =
    node.childForFieldName("name") ??
    node.childForFieldName("declarator")?.childForFieldName("declarator") ??
    node.childForFieldName("declarator");
  if (named) {
    const text = nodeText(source, named).trim();
    const match = text.match(/[A-Za-z_$][\w$]*(?=\s*(?:\(|$))/);
    return match?.[0] ?? text.slice(0, 120);
  }
  return node.type;
}

function signature(node: SyntaxNode, source: Buffer): string {
  const body = node.childForFieldName("body");
  const end = body?.startIndex ?? Math.min(node.endIndex, node.startIndex + 240);
  return source
    .subarray(node.startIndex, end)
    .toString("utf8")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 240);
}

function extractEntities(root: SyntaxNode, source: Buffer): ExtractedEntity[] {
  const entities: ExtractedEntity[] = [];
  const visit = (node: SyntaxNode): void => {
    const type = classifyEntity(node);
    if (type) {
      entities.push({
        name: entityName(node, source),
        type,
        signature: signature(node, source),
        lineRange: { start: node.startPosition.row, end: node.endPosition.row },
        start: node.startIndex,
        end: node.endIndex,
        node,
      });
    }
    for (const child of node.namedChildren) visit(child);
  };
  visit(root);
  return entities;
}

function taggedEntityType(capture: string): CompatibleEntityType | null {
  const kind = capture.slice("definition.".length).toLowerCase();
  if (kind.includes("method")) return "method";
  if (kind.includes("function") || kind.includes("macro")) return "function";
  if (kind.includes("class") || kind.includes("module") || kind.includes("namespace"))
    return "class";
  if (kind.includes("interface") || kind.includes("trait")) return "interface";
  if (kind.includes("enum")) return "enum";
  if (kind.includes("type") || kind.includes("constant")) return "type";
  return null;
}

async function extractTaggedEntities(
  root: SyntaxNode,
  source: Buffer,
  treeSitterLanguage: Language,
  grammar: SupportedLanguage,
): Promise<ExtractedEntity[]> {
  const available = getAvailableQueries(grammar) as Record<string, string>;
  const tagsPath = available.tags;
  if (!tagsPath) return [];
  let query: Query | undefined;
  try {
    let sourcePromise = tagQuerySourceCache.get(grammar);
    if (!sourcePromise) {
      sourcePromise = readFile(tagsPath, "utf8").catch(() => null);
      tagQuerySourceCache.set(grammar, sourcePromise);
    }
    const querySource = await sourcePromise;
    if (!querySource) return [];
    query = new Query(treeSitterLanguage, querySource);
    return query.matches(root).flatMap((match) => {
      const definition = match.captures.find((capture) => capture.name.startsWith("definition."));
      const type = definition ? taggedEntityType(definition.name) : null;
      if (!definition || !type) return [];
      const nameCapture = match.captures.find(
        (capture) => capture.name === "name" || capture.name.endsWith(".name"),
      );
      const docCapture = match.captures.find(
        (capture) => capture.name === "doc" || capture.name.includes("doc"),
      );
      const name = nameCapture
        ? nodeText(source, nameCapture.node).trim()
        : entityName(definition.node, source);
      return [
        {
          name,
          type,
          signature: signature(definition.node, source),
          ...(docCapture ? { docstring: nodeText(source, docCapture.node).trim() } : {}),
          lineRange: {
            start: definition.node.startPosition.row,
            end: definition.node.endPosition.row,
          },
          start: definition.node.startIndex,
          end: definition.node.endIndex,
          node: definition.node,
        },
      ];
    });
  } catch {
    return [];
  } finally {
    query?.delete();
  }
}

function publicEntity(entity: ExtractedEntity, partial = false): EntityInfo {
  return {
    name: entity.name,
    type: entity.type,
    ...(entity.signature ? { signature: entity.signature } : {}),
    ...(entity.docstring !== undefined ? { docstring: entity.docstring } : {}),
    ...(entity.lineRange ? { lineRange: entity.lineRange } : {}),
    ...(partial ? { isPartial: true } : {}),
  };
}

function contextFor(
  filepath: string,
  language: DetectedSource,
  span: Span,
  entities: ExtractedEntity[],
  root: SyntaxNode,
  source: Buffer,
  options: FallbackOptions,
): CompatibleChunk["context"] {
  if (options.contextMode === "none") return { scope: [], entities: [], siblings: [], imports: [] };
  const overlapping = entities.filter(
    (entity) => entity.start < span.end && entity.end > span.start,
  );
  const containing = overlapping
    .filter((entity) => entity.start <= span.start && entity.end > span.start)
    .sort((a, b) => a.end - a.start - (b.end - b.start));
  const defined = overlapping.filter(
    (entity) => entity.start >= span.start && entity.start < span.end && entity.type !== "import",
  );
  const topLevel = entities.filter((entity) => {
    const parentEntity = entities.find(
      (candidate) =>
        candidate !== entity &&
        candidate.start <= entity.start &&
        candidate.end >= entity.end &&
        (candidate.start < entity.start || candidate.end > entity.end),
    );
    return !parentEntity;
  });
  const before = topLevel.filter((entity) => entity.end <= span.start).at(-1);
  const after = topLevel.find((entity) => entity.start >= span.end);
  let imports = entities
    .filter((entity) => entity.type === "import")
    .slice(0, 10)
    .map((entity) => {
      const text = source.subarray(entity.start, entity.end).toString("utf8");
      const sourceMatch = text.match(/["']([^"']+)["']/);
      return {
        name: entity.name,
        source: sourceMatch?.[1] ?? text.replace(/\s+/g, " ").slice(0, 120),
      };
    });
  if (options.filterImports) {
    const usedNames = new Set(
      defined.flatMap(
        (entity) => `${entity.name} ${entity.signature ?? ""}`.match(/[A-Za-z_$][\w$]*/g) ?? [],
      ),
    );
    imports = imports.filter((item) => usedNames.has(item.name));
  }
  const syntaxNodes = [...root.namedChildren, ...overlapping.map((entity) => entity.node)]
    .filter((node) => node.startIndex < span.end && node.endIndex > span.start)
    .filter(
      (node, index, nodes) =>
        nodes.findIndex(
          (candidate) =>
            candidate.type === node.type &&
            candidate.startIndex === node.startIndex &&
            candidate.endIndex === node.endIndex,
        ) === index,
    )
    .slice(0, 256);
  return {
    filepath,
    language: language.id,
    scope: containing.map((entity) => publicEntity(entity)),
    entities: defined.map((entity) => publicEntity(entity, entity.end > span.end)),
    siblings:
      options.siblingDetail === "none"
        ? []
        : [
            ...(before
              ? [{ name: before.name, type: before.type, position: "before" as const, distance: 1 }]
              : []),
            ...(after
              ? [{ name: after.name, type: after.type, position: "after" as const, distance: 1 }]
              : []),
          ],
    imports: options.contextMode === "full" ? imports : [],
    parser: {
      name: "tree-sitter",
      language: language.id,
      ...(language.grammar ? { grammar: language.grammar } : {}),
      recovered: false,
    },
    sourceMetadata: {
      name: language.name,
      type: language.type,
      detectedBy: language.detectedBy,
    },
    syntaxNodes: syntaxNodes.map((node) => ({
      type: node.type,
      byteRange: { start: node.startIndex, end: node.endIndex },
      lineRange: { start: node.startPosition.row, end: node.endPosition.row },
      named: node.isNamed,
      missing: node.isMissing,
      hasError: node.hasError,
    })),
  };
}

function formatContext(
  text: string,
  context: CompatibleChunk["context"],
  overlapText?: string,
): string {
  const parts: string[] = [];
  if (context.filepath) parts.push(`# ${context.filepath.split("/").slice(-3).join("/")}`);
  if (context.language) parts.push(`# Language: ${context.language}`);
  if (context.scope.length)
    parts.push(
      `# Scope: ${context.scope
        .map((item) => item.name)
        .reverse()
        .join(" > ")}`,
    );
  const signatures = context.entities
    .filter((entity) => entity.signature && entity.type !== "import")
    .map((entity) => entity.signature);
  if (signatures.length) parts.push(`# Defines: ${signatures.join(", ")}`);
  if (context.imports.length)
    parts.push(`# Uses: ${context.imports.map((item) => item.name).join(", ")}`);
  const before = context.siblings
    .filter((item) => item.position === "before")
    .map((item) => item.name);
  const after = context.siblings
    .filter((item) => item.position === "after")
    .map((item) => item.name);
  if (before.length) parts.push(`# After: ${before.join(", ")}`);
  if (after.length) parts.push(`# Before: ${after.join(", ")}`);
  if (parts.length) parts.push("");
  if (overlapText) parts.push("# ...", overlapText, "# ---");
  parts.push(text);
  return parts.join("\n");
}

/** Select the closest-to-target contiguous suffix without splitting a source line. */
export function selectLineOverlap(text: string, targetTokens: number): string | undefined {
  if (targetTokens <= 0 || !text) return undefined;
  const lines = text.match(/[^\n]*(?:\n|$)/g)?.filter(Boolean) ?? [];
  if (lines.length === 0) return undefined;

  let bestStart = lines.length - 1;
  let bestDistance = Number.POSITIVE_INFINITY;
  let tokens = 0;
  for (let index = lines.length - 1; index >= 0; index--) {
    tokens += Buffer.byteLength(lines[index]!, "utf8");
    const distance = Math.abs(targetTokens - tokens);
    if (distance <= bestDistance) {
      bestStart = index;
      bestDistance = distance;
    }
    if (tokens >= targetTokens && distance > bestDistance) break;
  }
  return lines.slice(bestStart).join("");
}

function overlapTarget(options: FallbackOptions): number {
  return Math.round(options.maxChunkSize * options.overlapRatio);
}

const LOGICAL_DECLARATION =
  /^\s*(?:(?:pub|public|private|protected|static|async|export|abstract|final|open|internal)\s+)*(class|interface|enum|struct|record|trait|module|namespace|function|fn|func|def|method|sub|proc|procedure|type|impl|entity|architecture|package|component)\s+([A-Za-z_$][\w$.-]*)/i;
const LOGICAL_IMPORT = /^\s*(?:import|include|require|use|using|from|with)\b/i;

function logicalEntityType(keyword: string): CompatibleEntityType {
  const normalized = keyword.toLowerCase();
  if (
    [
      "class",
      "struct",
      "record",
      "module",
      "namespace",
      "trait",
      "impl",
      "entity",
      "architecture",
      "package",
      "component",
    ].includes(normalized)
  ) {
    return "class";
  }
  if (normalized === "interface") return "interface";
  if (normalized === "enum") return "enum";
  if (normalized === "type") return "type";
  if (normalized === "method") return "method";
  return "function";
}

function logicalUnits(source: Buffer, maxSize: number): Span[] {
  const starts = lineStarts(source);
  const lines = starts.map((start, index) => ({
    start,
    end: starts[index + 1] ?? source.length,
    text: source.subarray(start, starts[index + 1] ?? source.length).toString("utf8"),
  }));
  const blocks: Span[] = [];
  let blockStart = 0;
  let depth = 0;
  for (const line of lines) {
    const declaration = LOGICAL_DECLARATION.test(line.text);
    if (line.start > blockStart && depth <= 0 && declaration) {
      blocks.push({ start: blockStart, end: line.start });
      blockStart = line.start;
    }
    const opens = line.text.match(/[({[]/g)?.length ?? 0;
    const closes = line.text.match(/[)}\]]/g)?.length ?? 0;
    depth = Math.max(0, depth + opens - closes);
    if (depth === 0 && !line.text.trim() && line.end > blockStart) {
      blocks.push({ start: blockStart, end: line.end });
      blockStart = line.end;
    }
  }
  if (blockStart < source.length) blocks.push({ start: blockStart, end: source.length });
  return mergeAdjacent(
    blocks.flatMap((block) =>
      block.end - block.start > maxSize
        ? splitAtLines(source, block.start, block.end, maxSize)
        : [block],
    ),
    maxSize,
  );
}

function logicalMetadata(source: Buffer): {
  entities: (EntityInfo & { start: number; end: number })[];
  imports: { name: string; source: string; start: number; end: number }[];
} {
  const entities: (EntityInfo & { start: number; end: number })[] = [];
  const imports: { name: string; source: string; start: number; end: number }[] = [];
  const starts = lineStarts(source);
  for (const [index, start] of starts.entries()) {
    const end = starts[index + 1] ?? source.length;
    const text = source.subarray(start, end).toString("utf8").trim();
    const declaration = text.match(LOGICAL_DECLARATION);
    if (declaration?.[1] && declaration[2]) {
      entities.push({
        name: declaration[2],
        type: logicalEntityType(declaration[1]),
        signature: text.slice(0, 240),
        lineRange: { start: index, end: index },
        start,
        end,
      });
    }
    if (LOGICAL_IMPORT.test(text)) {
      const imported = text.match(/["'<]([^"'>]+)["'>]/)?.[1] ?? text.slice(0, 120);
      imports.push({ name: imported, source: imported, start, end });
    }
  }
  return { entities, imports };
}

export function chunkLogically(
  filepath: string,
  code: string,
  language: DetectedSource,
  options: FallbackOptions,
  fallbackReason?: string,
): CompatibleChunk[] {
  const source = Buffer.from(code, "utf8");
  const spans = logicalUnits(source, options.maxChunkSize);
  const starts = lineStarts(source);
  const metadata = logicalMetadata(source);
  let previousText: string | undefined;
  return spans.map((span, index) => {
    const text = source.subarray(span.start, span.end).toString("utf8");
    const entities = metadata.entities.filter(
      (entity) => entity.start >= span.start && entity.start < span.end,
    );
    const before = metadata.entities.filter((entity) => entity.end <= span.start).at(-1);
    const after = metadata.entities.find((entity) => entity.start >= span.end);
    const context: CompatibleChunk["context"] =
      options.contextMode === "none"
        ? { scope: [], entities: [], siblings: [], imports: [] }
        : {
            filepath,
            language: language.id,
            scope: [],
            entities: entities.map(({ start: _start, end: _end, ...entity }) => entity),
            siblings:
              options.siblingDetail === "none"
                ? []
                : [
                    ...(before
                      ? [
                          {
                            name: before.name,
                            type: before.type,
                            position: "before" as const,
                            distance: 1,
                          },
                        ]
                      : []),
                    ...(after
                      ? [
                          {
                            name: after.name,
                            type: after.type,
                            position: "after" as const,
                            distance: 1,
                          },
                        ]
                      : []),
                  ],
            imports:
              options.contextMode === "full"
                ? metadata.imports
                    .filter((item) => !options.filterImports || text.includes(item.name))
                    .map(({ name, source: importSource }) => ({ name, source: importSource }))
                : [],
            parser: {
              name: "logical-boundary",
              language: language.id,
              ...(fallbackReason ? { fallbackReason } : {}),
            },
            sourceMetadata: {
              name: language.name,
              type: language.type,
              detectedBy: language.detectedBy,
            },
          };
    const overlapText = previousText
      ? selectLineOverlap(previousText, overlapTarget(options))
      : undefined;
    const chunk: CompatibleChunk = {
      text,
      contextualizedText: formatContext(text, context, overlapText),
      byteRange: span,
      lineRange: {
        start: lineAt(starts, span.start),
        end: lineAt(starts, Math.max(span.start, span.end - 1)),
      },
      context,
      index,
      totalChunks: spans.length,
    };
    previousText = text;
    return chunk;
  });
}

function plainTextUnits(source: Buffer, maxSize: number): Span[] {
  const starts = lineStarts(source);
  const blocks: Span[] = [];
  let blockStart = 0;
  for (const [index, start] of starts.entries()) {
    const end = starts[index + 1] ?? source.length;
    if (!source.subarray(start, end).toString("utf8").trim() && end > blockStart) {
      blocks.push({ start: blockStart, end });
      blockStart = end;
    }
  }
  if (blockStart < source.length) blocks.push({ start: blockStart, end: source.length });
  return mergeAdjacent(
    blocks.flatMap((block) =>
      block.end - block.start > maxSize
        ? splitAtLines(source, block.start, block.end, maxSize)
        : [block],
    ),
    maxSize,
  );
}

export function chunkPlainText(
  filepath: string,
  code: string,
  language: DetectedSource,
  options: FallbackOptions,
  fallbackReason?: string,
): CompatibleChunk[] {
  const source = Buffer.from(code, "utf8");
  const spans = plainTextUnits(source, options.maxChunkSize);
  const starts = lineStarts(source);
  let previousText: string | undefined;
  return spans.map((span, index) => {
    const text = source.subarray(span.start, span.end).toString("utf8");
    const context: CompatibleChunk["context"] =
      options.contextMode === "none"
        ? { scope: [], entities: [], siblings: [], imports: [] }
        : {
            filepath,
            language: language.id,
            scope: [],
            entities: [],
            siblings: [],
            imports: [],
            parser: {
              name: "plain-text",
              language: language.id,
              ...(fallbackReason ? { fallbackReason } : {}),
            },
            sourceMetadata: {
              name: language.name,
              type: language.type,
              detectedBy: language.detectedBy,
            },
          };
    const overlapText = previousText
      ? selectLineOverlap(previousText, overlapTarget(options))
      : undefined;
    const chunk: CompatibleChunk = {
      text,
      contextualizedText: formatContext(text, context, overlapText),
      byteRange: span,
      lineRange: {
        start: lineAt(starts, span.start),
        end: lineAt(starts, Math.max(span.start, span.end - 1)),
      },
      context,
      index,
      totalChunks: spans.length,
    };
    previousText = text;
    return chunk;
  });
}

export async function chunkWithTreeSitter(
  filepath: string,
  code: string,
  language: DetectedSource,
  options: FallbackOptions,
): Promise<CompatibleChunk[]> {
  const source = Buffer.from(code, "utf8");
  if (!language.grammar) throw new Error(`missing_tree_sitter_grammar:${language.id}`);
  const treeSitterLanguage = await loadLanguage(language.grammar);
  const parser = new Parser();
  try {
    parser.setLanguage(treeSitterLanguage);
    const tree = parser.parse(code);
    if (!tree) throw new Error("tree_sitter_parse_failed");
    try {
      const rootUnits = structuralUnits(tree.rootNode, source, options.maxChunkSize);
      const units = [
        ...(tree.rootNode.startIndex > 0
          ? splitAtLines(source, 0, tree.rootNode.startIndex, options.maxChunkSize)
          : []),
        ...rootUnits,
        ...(tree.rootNode.endIndex < source.length
          ? splitAtLines(source, tree.rootNode.endIndex, source.length, options.maxChunkSize)
          : []),
      ];
      const spans = mergeAdjacent(units, options.maxChunkSize);
      const taggedEntities = await extractTaggedEntities(
        tree.rootNode,
        source,
        treeSitterLanguage,
        language.grammar,
      );
      const entities = [
        ...taggedEntities,
        ...extractEntities(tree.rootNode, source).filter(
          (entity) =>
            !taggedEntities.some(
              (tagged) =>
                tagged.start === entity.start &&
                tagged.end === entity.end &&
                tagged.type === entity.type,
            ),
        ),
      ];
      const starts = lineStarts(source);
      const chunks: CompatibleChunk[] = [];
      let previousText: string | undefined;
      for (const [index, span] of spans.entries()) {
        const text = source.subarray(span.start, span.end).toString("utf8");
        const context = contextFor(
          filepath,
          language,
          span,
          entities,
          tree.rootNode,
          source,
          options,
        );
        if (tree.rootNode.hasError) {
          context.parseError = {
            message: "Tree-sitter recovered from syntax errors",
            recoverable: true,
          };
          if (context.parser) context.parser.recovered = true;
        }
        const overlapText = previousText
          ? selectLineOverlap(previousText, overlapTarget(options))
          : undefined;
        chunks.push({
          text,
          contextualizedText: formatContext(text, context, overlapText),
          byteRange: span,
          lineRange: {
            start: lineAt(starts, span.start),
            end: lineAt(starts, Math.max(span.start, span.end - 1)),
          },
          context,
          index,
          totalChunks: spans.length,
        });
        previousText = text;
      }
      return chunks;
    } finally {
      tree.delete();
    }
  } finally {
    parser.delete();
  }
}
