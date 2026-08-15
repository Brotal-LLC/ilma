# Embedder identity enforcement

## Why

When an operator swaps the embedding model (e.g. `text-embedding-3-small` →
`bge-m3` → `nomic-embed-text-v1.5`), the existing vectors in Postgres are
dimensionally incompatible with the new model's output. The current code
silently accepts the mismatch and returns nonsense on recall — high cosine
similarity between unrelated memories because the dimensions get padded or
truncated by pgvector.

We need to record which embedder produced each memory row so the system can
refuse to serve a `MemoryRepo` whose model identity doesn't match the
ingested corpus, and so `ilma doctor` can detect mixed-corpus drift.

## What changes

- Every memory row records `embedder_model` (str) and `embedder_dim` (int) at
  write time. Default: `"unknown"` / `0` for legacy rows.
- `MemoryRepo.remember()` rejects writes if the caller hasn't pinned an
  embedder identity via the new `embedder=` constructor arg.
- `MemoryRepo.recall()` raises `EmbedderMismatchError` when the live embedder
  identity doesn't match the corpus identity stored in the
  `ilma_corpus_meta` table.
- New CLI command `ilma doctor --embedder-audit` lists memories with missing
  or mixed embedder identities and offers to re-embed in batches.
- New spec `openspec/specs/memory/embedder-identity.md` is the source of
  truth going forward.

## Impact

- **Affected specs**: none yet — this is a *new* area (`memory/embedder-identity`).
- **Affected code**:
  - `src/ilma/core/embeddings/` — add `EmbedderIdentity` dataclass.
  - `src/ilma/core/memory.py` — constructor takes `embedder: EmbedderIdentity`,
    `remember()` validates, `recall()` checks corpus meta.
  - `src/ilma/storage/` — schema migration adds two columns + a meta table.
  - `src/ilma/cli.py` — adds `ilma doctor --embedder-audit`.
  - `tests/unit/` + `tests/integration/` — TDD coverage.
- **Breaking change**: yes. `MemoryRepo` constructor gains a required
  parameter. Migration path: `ilma doctor --embedder-audit` flags legacy
  rows; operator sets the new embedder explicitly and runs the audit
  command to confirm the corpus is homogeneous before deploying the
  updated server.

## Out of scope

- Re-embedding legacy corpora automatically — that's a separate
  `ilma repair re-embed` change (P5 row 4).
- Multi-model routing (different embeddings for different memories).
  One embedder per `MemoryRepo` for now.
- Vector dimension coercion / padding. Mismatch is a hard error, not
  silent recovery.
