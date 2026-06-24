"""Tests for the post-migration data repair module."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from ilma.repair import (
    _chunk_content,
    _is_zero_vec,
    _looks_corrupt_tags,
    _parse_proper_tags,
    _to_vec_literal,
    repair_migrated_memories,
)


class FakeCursor:
    """Minimal psycopg cursor mock for testing the repair flow.

    Records every execute() and fetchall() so tests can assert what
    SQL was issued and what rows came back. Implements the minimal
    context-manager protocol used by psycopg cursors.
    """

    def __init__(self, rows: list[tuple] | None = None) -> None:
        self._rows = rows or []
        self._executed: list[tuple[str, tuple[Any, ...]]] = []
        self.executed_sql: list[str] = []
        self.executed_params: list[tuple[Any, ...]] = []

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        self._executed.append((sql, params))
        self.executed_sql.append(sql)
        self.executed_params.append(params)

    def fetchall(self) -> list[tuple]:
        return self._rows


class FakeConnection:
    def __init__(self, rows_by_query: list[list[tuple]] | None = None) -> None:
        # rows_by_query[i] is the result set returned by the i-th execute().
        # - "tags"    : SELECT id, tags, content FROM ilma.memories WHERE deleted_at IS NULL ORDER BY id
        # - "vectors" : SELECT id, content FROM ilma.memories WHERE deleted_at IS NULL AND (...)
        # - "chunks"  : SELECT m.id, m.content, COUNT(c.id) ...
        self._queries = list(rows_by_query or [])
        self.commits = 0
        self.cursors: list[FakeCursor] = []

    def cursor(self) -> FakeCursor:
        cur = FakeCursor(self._queries.pop(0) if self._queries else [])
        self.cursors.append(cur)
        return cur

    def commit(self) -> None:
        self.commits += 1


def make_embedder(dim: int = 1024) -> MagicMock:
    emb = MagicMock()
    emb.default_dim = dim

    def fake_embed(text: str, *, dim: int) -> list[float]:
        # Return a unique non-zero vector per text so the repair loop
        # sees each call produce a "real" embedding.
        # Use a simple deterministic hash → vector mapping.
        h = abs(hash(text)) % (10**6) or 1
        return [(h + i) / 100.0 for i in range(dim)]

    emb.embed.side_effect = fake_embed
    return emb


# -----------------------------------------------------------------------
# Pure helpers — no DB
# -----------------------------------------------------------------------


def test_looks_corrupt_tags_recognises_degenerate_arrays() -> None:
    assert _looks_corrupt_tags(["i", "d", "e", "n", "t", "i", "t", "y", ",", " "])
    assert _looks_corrupt_tags(["a", "b"] * 20)  # 40 elements


def test_looks_corrupt_tags_passes_well_formed_arrays() -> None:
    assert not _looks_corrupt_tags(["identity", "roleplay", "loyalty"])
    assert not _looks_corrupt_tags(["infra", "postgres", "embeddings"])
    assert not _looks_corrupt_tags([])


def test_parse_proper_tags_recovers_known_anchors() -> None:
    content = (
        'User identity: Rezaur Rahman is "Bruce Wayne" (Master Wayne) in '
        'roleplay. Fluffy takes the persona of "Alfred".'
    )
    tags = _parse_proper_tags(content)
    assert "user-identity" in tags
    assert "roleplay" in tags
    assert "alfred" in tags
    assert "bruce-wayne" in tags


def test_parse_proper_tags_falls_back_to_uncategorized() -> None:
    tags = _parse_proper_tags("hello world this has no anchors")
    assert tags == ["uncategorized"]


def test_chunk_content_short_text_returns_one_chunk() -> None:
    chunks = _chunk_content("hello world")
    assert chunks == ["hello world"]


def test_chunk_content_splits_long_text_on_sentence_boundary() -> None:
    text = "First sentence. " * 100 + "Last sentence."
    chunks = _chunk_content(text, target_chars=512, max_chars=1024)
    # Should split into more than one chunk
    assert len(chunks) > 1
    # No empty chunks
    assert all(c.strip() for c in chunks)


def test_is_zero_vec_detects_all_zero() -> None:
    assert _is_zero_vec([0.0] * 1024)
    assert not _is_zero_vec([0.001] + [0.0] * 1023)
    assert not _is_zero_vec([0.5] * 1024)


def test_to_vec_literal_round_trip() -> None:
    vec = [0.1, 0.2, 0.3]
    literal = _to_vec_literal(vec)
    assert literal.startswith("[")
    assert literal.endswith("]")
    # round-trip parse
    parsed = [float(x) for x in literal.strip("[]").split(",")]
    assert parsed == vec


# -----------------------------------------------------------------------
# Integration — repair_migrated_memories against a fake connection
# -----------------------------------------------------------------------


def test_repair_skips_when_no_rows_need_attention() -> None:
    # Empty results from all three SELECTs → no UPDATEs/INSERTs.
    conn = FakeConnection(rows_by_query=[[], [], []])
    stats = repair_migrated_memories(conn, embedder_registry=make_embedder())

    assert stats["tags_cleaned"] == 0
    assert stats["memories_embedded"] == 0
    assert stats["chunks_created"] == 0
    # Only 3 SELECTs issued, no writes
    for cur in conn.cursors:
        for sql in cur.executed_sql:
            assert sql.lstrip().upper().startswith("SELECT")
    # Commit still called (so the SELECTs' implicit txn closes cleanly)
    assert conn.commits == 1


def test_repair_cleans_corrupt_tags() -> None:
    corrupt_tags_row = (
        1,
        ["i", "d", "e", "n", "t", "i", "t", "y", ",", " "],
        "User identity: Rezaur Rahman is Bruce Wayne roleplay with Alfred persona",
    )
    conn = FakeConnection(rows_by_query=[[corrupt_tags_row], [], []])
    # The first cursor should have issued the SELECT *plus* an UPDATE
    stats = repair_migrated_memories(conn, embedder_registry=make_embedder())
    assert stats["tags_cleaned"] == 1
    cur_tags = conn.cursors[0]
    update_sql = next(
        sql
        for sql, _params in zip(cur_tags.executed_sql, cur_tags.executed_params, strict=False)
        if "UPDATE" in sql.upper()
    )
    assert "ilma.memories SET tags" in update_sql
    # The new tags arg should be the parsed proper tags
    update_params = next(
        params
        for sql, params in zip(cur_tags.executed_sql, cur_tags.executed_params, strict=False)
        if "UPDATE" in sql.upper()
    )
    new_tags = update_params[0]
    assert "user-identity" in new_tags
    assert "roleplay" in new_tags
    assert "alfred" in new_tags


def test_repair_reembeds_zero_vector_memories() -> None:
    # The vector branch sees one memory whose vector is all-zero.
    # Note: the SELECT for the vector branch selects id, content
    vector_row = (5, "some content to embed")
    conn = FakeConnection(rows_by_query=[[], [vector_row], []])

    stats = repair_migrated_memories(conn, embedder_registry=make_embedder())
    assert stats["memories_embedded"] == 1

    # The second cursor should have issued an UPDATE with a vector literal
    cur_vec = conn.cursors[1]
    update_sql = next(sql for sql in cur_vec.executed_sql if "UPDATE" in sql.upper())
    assert "vector_1024" in update_sql
    assert "::vector" in update_sql


def test_repair_chunks_long_content_without_existing_chunks() -> None:
    # Long content row from the chunk-eligibility SELECT
    chunk_row = (7, "x" * 500, 0)  # id, content, count(existing chunks)
    conn = FakeConnection(rows_by_query=[[], [], [chunk_row]])

    stats = repair_migrated_memories(conn, embedder_registry=make_embedder())
    # The content is one big 500-char blob; we chunk it on word boundaries
    # (no sentence-end punctuation). Should produce at least 1 chunk.
    assert stats["chunks_created"] >= 1

    cur_chunks = conn.cursors[2]
    insert_sql = next(sql for sql in cur_chunks.executed_sql if "INSERT" in sql.upper())
    assert "ilma.memory_chunks" in insert_sql
    assert "ON CONFLICT" in insert_sql


def test_repair_is_idempotent_when_no_corruption_present() -> None:
    # Well-formed tags, no zero-vector memories, all memories already chunked.
    well_formed_row = (1, ["identity", "roleplay"], "Some short content.")
    # The chunk-eligibility SELECT filters length > 200, so empty means no rows.
    conn = FakeConnection(
        rows_by_query=[
            [well_formed_row],  # tags SELECT (will not match corrupt heuristic)
            [],  # zero-vector SELECT (no rows)
            [],  # chunk-eligibility SELECT (no rows)
        ]
    )
    stats = repair_migrated_memories(conn, embedder_registry=make_embedder())
    assert stats["tags_cleaned"] == 0
    assert stats["memories_embedded"] == 0
    assert stats["chunks_created"] == 0
