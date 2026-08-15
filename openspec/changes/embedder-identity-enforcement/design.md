# Design: Embedder identity enforcement

## Approach

Add an `EmbedderIdentity` value object to `ilma.core.embeddings` that pairs
`(model: str, dim: int, version: str | None = None)`. Every `MemoryRepo`
binds one identity at construction time. Writes stamp the identity onto
the row. Reads compare the repo's identity against a single row in the new
`ilma_corpus_meta` table and refuse to serve on mismatch.

The corpus meta table holds exactly one row per `memory_repo` logical
database, set on first successful write. Reads fail closed if the row is
absent (cold start) — a migration step seeds it from the most common
identity in the existing corpus, gated by `ilma doctor --embedder-audit`
which the operator must run before deploying.

## Key decisions

- **Single identity per repo, not per row lookups on every read.**
  The corpus meta table is one row. Reads do one indexed lookup, not a
  `SELECT DISTINCT embedder_model FROM memories` on every call.
- **Hard error on mismatch, not coercion.** Padded vectors give false
  positives. The user's stance: "fail loudly, never silently degrade."
- **Migration is operator-driven, not auto.** `ilma doctor --embedder-audit`
  is the audit; the operator sets the embedder identity explicitly when
  starting the new server version. No surprise re-embedding.
- **Legacy rows get `"unknown"` / `0`.** They are still readable as long
  as the operator's pinned identity matches a meta row seeded by the
  audit. If the operator pins a *real* identity, legacy rows surface in
  the audit output but don't block reads.
- **Rejected alternative**: tag-based filtering on each read. Too slow,
  and a model swap would still produce garbage from unfiltered rows.

## API / interface changes

```python
# src/ilma/core/embeddings/identity.py
@dataclass(frozen=True)
class EmbedderIdentity:
    model: str                       # e.g. "text-embedding-3-small"
    dim: int                         # e.g. 1536
    version: str | None = None       # e.g. provider-side revision

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if not self.model:
            raise ValueError("model must be non-empty")

# src/ilma/core/memory.py
class EmbedderMismatchError(RuntimeError):
    """MemoryRepo was constructed with an embedder identity that does not
    match the corpus recorded in ilma_corpus_meta."""

class MemoryRepo:
    def __init__(self, ..., *, embedder: EmbedderIdentity) -> None: ...
    def remember(self, ...) -> None: ...   # stamps identity
    def recall(self, ...) -> list[...]: ...  # raises on mismatch
```

```sql
-- migration
ALTER TABLE memories
    ADD COLUMN embedder_model TEXT NOT NULL DEFAULT 'unknown',
    ADD COLUMN embedder_dim   INT  NOT NULL DEFAULT 0;

CREATE TABLE ilma_corpus_meta (
    id              INT  PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    embedder_model  TEXT NOT NULL,
    embedder_dim    INT  NOT NULL,
    seeded_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (id = 1)
);
```

```bash
# CLI
ilma doctor --embedder-audit
# Output: count of rows per (model, dim), warnings for mixed corpora,
# suggested seed identity.
```

## Risks

- **Existing deployments break on upgrade** if the operator doesn't run
  the audit first. Mitigation: ship `ilma doctor --embedder-audit` in the
  same release and surface a clear "run me first" warning in CHANGELOG.
- **Tests using `FakeMemoryRepo`** need a default identity. Add a
  `_DEFAULT_TEST_IDENTITY = EmbedderIdentity("test-model", 4)` constant
  in `tests/conftest.py` so all existing tests keep passing.
- **Single-row `ilma_corpus_meta` is a hot row.** Reads happen on every
  `recall()`. Mitigation: cache the identity in process memory with a
  short TTL (e.g. 30s) and a manual `clear_cache()` method for operators
  who change the identity mid-process (shouldn't happen in prod, but
  tests need it).

## Test plan

- **Unit**:
  - `EmbedderIdentity.__post_init__` rejects empty model / non-positive dim.
  - `MemoryRepo` constructor rejects missing `embedder=`.
  - `MemoryRepo.remember()` stamps identity from the repo on each row.
  - `MemoryRepo.recall()` raises `EmbedderMismatchError` when meta row
    embedder != repo embedder.
  - `ilma doctor --embedder-audit` correctly buckets legacy (`unknown`/`0`)
    rows separately from real rows.
- **Integration** (Testcontainers Postgres + pgvector):
  - Migration applies cleanly on a fresh DB.
  - Migration applies cleanly on an existing DB with legacy rows; audit
    reports them as `unknown`.
  - End-to-end: write → recall succeeds; swap identity → next recall
    raises; restore identity → recall succeeds.
- **Manual**:
  - Run the new migration on a staging DB copy; confirm `ilma doctor`
    output matches `SELECT embedder_model, embedder_dim, COUNT(*) FROM
    memories GROUP BY 1, 2`.
