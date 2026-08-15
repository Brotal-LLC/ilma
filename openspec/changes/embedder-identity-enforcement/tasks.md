# Tasks — Embedder identity enforcement

Each task = one commit. Run the test suite after each.

## 1. Specs (frozen first — everything else targets these)

- [ ] 1.1 Add `openspec/specs/memory/embedder-identity.md` with the three
      requirements (write-time stamping, read-time mismatch error, audit
      command). Verify: `openspec validate . --strict` passes.
- [ ] 1.2 Add a delta at
      `openspec/changes/embedder-identity-enforcement/specs/memory/spec.md`
      marked `## ADDED Requirements` mirroring the spec from 1.1. File
      MUST be named `spec.md` or strict validation fails. Verify:
      `openspec validate embedder-identity-enforcement --strict` is green.

## 2. EmbedderIdentity value object

- [ ] 2.1 Write failing unit tests for `EmbedderIdentity` constructor
      (rejects empty model, rejects `dim <= 0`, equality + hashability,
      frozen). Verify: tests fail on import (module doesn't exist).
- [ ] 2.2 Create `src/ilma/core/embeddings/identity.py` with the dataclass
      + `__post_init__`. Verify: tests pass.
- [ ] 2.3 Export from `src/ilma/core/embeddings/__init__.py`. Verify:
      `from ilma.core.embeddings import EmbedderIdentity` works in a REPL.

## 3. MemoryRepo changes (TDD)

- [ ] 3.1 Add `_DEFAULT_TEST_IDENTITY` to `tests/conftest.py` and patch
      all existing `FakeMemoryRepo` / `MemoryRepo` constructions in the
      test suite to use it. Verify: `pytest tests/unit` is green (no
      regressions from the new required arg).
- [ ] 3.2 Write failing unit tests for `MemoryRepo` constructor rejecting
      missing `embedder=`. Verify: test fails for the right reason
      (TypeError).
- [ ] 3.3 Add `embedder: EmbedderIdentity` constructor parameter and the
      `EmbedderMismatchError` exception. Verify: test passes.
- [ ] 3.4 Write failing unit tests for `remember()` stamping `embedder_model`
      and `embedder_dim` from `self.embedder` onto every row. Verify: fails
      because columns don't exist yet.
- [ ] 3.5 Write failing unit tests for `recall()` raising `EmbedderMismatchError`
      when `ilma_corpus_meta` row's embedder ≠ `self.embedder`. Verify: fails.

## 4. Storage migration

- [ ] 4.1 Add migration under `migrations/` adding the two columns +
      `ilma_corpus_meta` table. Verify: `alembic upgrade head` runs on a
      fresh DB.
- [ ] 4.2 Add migration downgrade path. Verify: `alembic downgrade -1`
      cleanly reverses the change.
- [ ] 4.3 Write integration test (Testcontainers Postgres) applying the
      migration to an existing DB with legacy rows and asserting the
      `unknown` / `0` defaults land. Verify: integration test passes.

## 5. MemoryRepo implementation (continue 3)

- [ ] 5.1 Wire `remember()` to stamp identity. Verify: 3.4 tests pass.
- [ ] 5.2 Wire `recall()` to read `ilma_corpus_meta` and raise on mismatch.
      Verify: 3.5 tests pass.
- [ ] 5.3 Add 30s in-process cache for the corpus identity with a
      `clear_cache()` method. Verify: unit test with a mock clock.

## 6. CLI: `ilma doctor --embedder-audit`

- [ ] 6.1 Write failing test for the audit command bucketing rows by
      `(embedder_model, embedder_dim)` and surfacing legacy `unknown`
      counts. Verify: fails (command not registered).
- [ ] 6.2 Wire the command in `src/ilma/cli.py` with the SQL aggregation.
      Verify: test passes.
- [ ] 6.3 Add human-readable output: count table + warning if > 1 distinct
      identity present. Verify: manual `ilma doctor --embedder-audit` on
      the staging DB.

## 7. Integration test (full loop)

- [ ] 7.1 End-to-end Testcontainers test: write with identity A → recall
      succeeds → swap repo to identity B → recall raises → swap back →
      recall succeeds. Verify: passes.

## 8. Final validation

- [ ] 8.1 Run full test suite (`make test` or equivalent). Verify: green.
- [ ] 8.2 Run `openspec validate embedder-identity-enforcement --strict`.
      Verify: green.
- [ ] 8.3 Run `openspec archive embedder-identity-enforcement`. Verify:
      delta specs merged into `openspec/specs/memory/embedder-identity.md`,
      change folder moved to `openspec/changes/archive/<date>-embedder-identity-enforcement/`.
