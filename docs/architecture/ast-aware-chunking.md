# AST-Aware Chunking

After a source snapshot becomes durable, ingestion creates a separate PostgreSQL chunking job. A
chunking failure never changes snapshot success and can be retried without downloading the
repository again.

## Runtime boundary

Python owns job leases, retry policy, snapshot materialization, artifact persistence, and the
embedding handoff. A warm Node 22 companion under `apps/chunker` owns the pinned
`code-chunk@0.1.14` integration and a `web-tree-sitter@0.26.3` fallback backed by the pinned
`tree-sitter-wasm@1.1.6` grammar bundle. The processes exchange
versioned NDJSON over local standard streams; this keeps the vendor package replaceable and avoids
per-file process startup. Protocol v3 emits each normalized chunk as its own bounded record followed
by a file-completion record. The Python client reassembles these records before handing them to the
service layer, preserving the existing artifact contract without relying on an unbounded NDJSON
line for large files.

The snapshot provider materializes an archive into a private temporary workspace and verifies every
file against the stored manifest. The worker accepts only manifest-listed relative paths, rejects
links and path escapes, and never receives cloud credentials.

## Processing

`code-chunk` detects and parses TypeScript, JavaScript, Python, Rust, Go, and Java. GitHub Linguist
metadata recognizes the wider source-file set by extension, exact filename, or shebang. When one of
the 112 bundled Tree-sitter grammars matches, the worker parses it offline. Both AST paths split at
semantic boundaries and produce the same scope, entity, import, sibling, range, and recoverable
parse-error context contract. The fallback also records the detected language, source kind,
detection method, Tree-sitter grammar, recovery state, and syntax-node types/ranges.

Recognized programming, markup, and repository-data files without a bundled grammar are not
discarded. They use a deterministic logical-boundary splitter that groups declarations and
blank-line-delimited blocks, extracts declaration signatures and import-like statements where the
syntax exposes them textually, and omits AST-only fields such as scope or syntax nodes. Its output
uses the identical normalized chunk contract and therefore follows the same persistence, embedding,
and vector-storage path.

Every remaining non-empty UTF-8 file uses a final plain-text splitter. It preserves paragraph and
line boundaries, combines adjacent small passages up to `maxChunkSize`, and records only the file,
text language, ranges, and plain-text parser metadata. It deliberately leaves code-specific entity,
scope, import, sibling, and syntax-node collections empty rather than fabricating metadata.
Tree-sitter failures degrade to logical chunks, and logical-analysis failures degrade again to
plain text; the selected parser metadata records the fallback reason.
ShowcaseHost stores both raw `text` and the worker-generated `contextualizedText`; the latter is the
canonical input for embedding providers.

The fallback recursively keeps a syntax node intact whenever it fits. Oversized nodes are divided
at named-child boundaries, with line boundaries used only when the AST has no smaller usable
structure. A greedy merge then combines adjacent small blocks up to `maxChunkSize`. Byte ranges are
computed against the original UTF-8 buffer, so every raw chunk is within the configured byte limit
and concatenating all chunks reconstructs the complete file.

Ordinary `code-chunk` files use `chunkBatchStream`; fallback files use bounded concurrent grammar or
logical parses. Both operate inside byte- and memory-bounded groups. Above the streaming threshold,
`code-chunk` uses its streaming API while the fallback emits chunks after analysis. Tree-sitter
requires the complete source file and AST. Generated, minified, empty, invalid UTF-8, binary assets,
and over-limit files receive explicit skip records.

## Profiles and reuse

A chunking profile records the embedding input limit, sizing counter, safety margin, context mode,
sibling detail, overlap, file limits, and concurrency controls. UTF-8 byte length is the conservative
default token estimate until an exact tokenizer is registered. Every contextualized input is checked
against the profile budget; oversized files are retried with a smaller byte target and finally with
minimal context, sibling names, and no overlap.

Overlap targets 20% of the resolved raw chunk size (about 205 conservative tokens at the default
1,024-token target). The worker selects the closest contiguous suffix made only of complete source
lines, so overlap never begins in the middle of a line. Adaptive sizing recomputes the target from
the smaller resolved chunk size.

Every chunk also receives an explainable quality grade and numeric rating. Preferred `code-chunk`
output is always in the A/90-100 band, Tree-sitter fallback output is B/75-89, logical-boundary
output is C/55-74, and plain text is D/35-54. Metadata completeness determines position within a
strategy's band; parser recovery and recorded fallback failures lower the rating without crossing
strategy bands. The grade, rating, strategy, completeness, and missing-field list are retained with
the chunk, while the grade and rating are also copied into vector payloads. Semantic similarity
remains the primary retrieval order, with quality rating used only to break equal-score ties.

Artifacts are fingerprinted from repository, path, content hash, engine version, profile, resolved
options, and normalization version. Unchanged files reuse artifacts across snapshots. A larger-window
embedding model can use existing chunks; a smaller window gets another versioned profile.

Vector data is intentionally separate. `EmbeddingInput` exposes stable chunk identity,
`contextualizedText`, raw source, citations, language, entities, scope, fingerprints, and sizing
metadata for the next pipeline stage. The complete normalized context is also copied into vector
payloads while the existing top-level `entities` and `scope` keys remain available for compatible
filters and consumers.

## Operational defaults

- Worker concurrency: 8
- Batch input budget: 16 MiB
- Streaming threshold: 1 MiB
- Maximum parsed source file: 5 MiB
- Worker memory budget: 256 MiB
- Raw-code target: 1,024 conservative tokens
- Full-line overlap target: 20% (about 205 conservative tokens by default)
- Embedding-window safety margin: 10%

Chunking job status is available at `GET /chunking-jobs/{id}`. The endpoint returns counts and
timings but never private source text.
