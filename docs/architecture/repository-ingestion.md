# Repository Ingestion

ShowcaseHost ingests the default branch of selected GitHub repositories through a GitHub App.
The app needs only read access to repository Contents and Metadata. User OAuth tokens are used
only while validating an installation and synchronizing the repository list; they are never
stored. Ingestion uses a short-lived installation token restricted to one repository.

## Snapshot flow

1. A durable PostgreSQL job is queued and claimed with a renewable lease.
2. GitHub repository metadata and the default branch are resolved to an exact commit SHA.
3. The SHA-pinned tar archive is streamed with configured size limits.
4. A safe extractor rejects traversal and special files, skips links, and hashes regular files.
5. The archive and versioned manifest are atomically stored under
   `github/{github_repository_id}/{commit_sha}`.
6. The latest two successful snapshots are retained by default.
7. A separate, durable AST-aware chunking job is atomically ensured for the snapshot.

The local snapshot provider is suitable for one API/worker host. A multi-instance deployment must
replace it with shared object storage through the `SnapshotStore` boundary.

## Incremental synchronization provision

The snapshot manifest contains the path, byte size, and SHA-256 digest of every regular file. A
future verified `push` webhook will enqueue the new head SHA. GitHub's compare API can prioritize
changed paths, while comparing the old and new manifests remains the correctness fallback.

AST-aware chunking reprocesses added and changed file hashes, removes snapshot membership for
deleted paths, and identifies artifacts using repository, path, content hash, engine version,
resolved options, symbol/span identity, and chunking profile. Matching fingerprints reuse existing
chunks. See [AST-Aware Chunking](ast-aware-chunking.md).

Webhook work is intentionally not implemented yet. It must validate `X-Hub-Signature-256` in
constant time and deduplicate `X-GitHub-Delivery` before enqueuing work.

## Known boundaries

- Only GitHub.com and the current default branch are supported.
- Git LFS pointer files remain pointers.
- Submodule contents are not included.
- Snapshots are private application data and have no static-file route.
