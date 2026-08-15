# Spec: memory/embedder-identity

## Purpose

Ensure every memory row can be traced to the embedder that produced its
vector, and refuse to serve a `MemoryRepo` whose embedder identity does
not match the corpus recorded for the database. Prevents silent
cross-model contamination when an operator swaps embedding providers
without re-embedding the corpus.

## Requirements

### Requirement: Write-time identity stamping

The system SHALL record the embedder identity (`embedder_model`,
`embedder_dim`) on every row written via `MemoryRepo.remember()`. The
identity SHALL be taken from the `EmbedderIdentity` passed to the
`MemoryRepo` constructor, not from caller-supplied arguments.

#### Scenario: First write stamps identity

- **WHEN** a `MemoryRepo` constructed with
  `EmbedderIdentity("text-embedding-3-small", 1536)` calls `remember()`
- **THEN** the persisted row SHALL have `embedder_model =
  "text-embedding-3-small"` and `embedder_dim = 1536`

#### Scenario: Legacy rows keep the default identity

- **WHEN** a memory row exists from before this spec shipped
- **THEN** the row SHALL have `embedder_model = "unknown"` and
  `embedder_dim = 0` and SHALL NOT block reads as long as the operator
  has not pinned a real identity that conflicts

### Requirement: Read-time mismatch error

The system SHALL raise `EmbedderMismatchError` from `MemoryRepo.recall()`
when the embedder identity recorded in `ilma_corpus_meta` does not match
the `EmbedderIdentity` bound to the `MemoryRepo` instance.

#### Scenario: Matched identity serves reads

- **WHEN** `ilma_corpus_meta.embedder_model = "bge-m3"` AND
  `ilma_corpus_meta.embedder_dim = 1024` AND the repo was constructed
  with `EmbedderIdentity("bge-m3", 1024)`
- **THEN** `recall()` SHALL return results normally

#### Scenario: Mismatched identity refuses reads

- **WHEN** `ilma_corpus_meta` records `"text-embedding-3-small"` / `1536`
  but the repo was constructed with `EmbedderIdentity("bge-m3", 1024)`
- **THEN** `recall()` SHALL raise `EmbedderMismatchError` and SHALL NOT
  return any results

### Requirement: Audit command

The system SHALL provide an `ilma doctor --embedder-audit` command that
reports the count of memory rows grouped by `(embedder_model,
embedder_dim)` and explicitly surfaces the `unknown` / `0` bucket as a
legacy warning.

#### Scenario: Homogeneous corpus

- **WHEN** all rows have `embedder_model = "bge-m3"` AND
  `embedder_dim = 1024`
- **THEN** `ilma doctor --embedder-audit` SHALL exit 0 and report a
  single row in the count table

#### Scenario: Mixed corpus

- **WHEN** rows have more than one distinct `(embedder_model,
  embedder_dim)` pair
- **THEN** `ilma doctor --embedder-audit` SHALL exit non-zero and print
  a warning identifying the mixed identities
