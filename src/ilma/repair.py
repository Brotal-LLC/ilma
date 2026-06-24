"""Post-migration data repair for ilma DBs migrated from hermes-memory v2.

This module exposes a single function, :func:`repair_migrated_memories`,
that the ``ilma migrate --reembed`` CLI flag calls after the schema
migration. It also runs standalone against any ilma DB.

When ``hermes-memory v2`` was the source DB, three pre-existing bugs ended
up preserved unchanged in the new ``ilma`` schema (because the
``ilma migrate`` command is intentionally a faithful copy):

1. **Corrupt tags** as a degenerate ``text[]``: a comma-separated string
   (``"identity, roleplay, loyalty, alfred"``) was stored as one array
   element per character (``{"i","d","e","n","t","i","t","y",",",...}``).
   Visible symptom: ``ilma recall`` shows tags like
   ``tags=i, d, e, n, t, i, t, y, ,,  , r, o, l, e, p, l, a, y, ...``.

2. **Zero vectors** in ``vector_768/1024/1536``: hermes-memory v2 never
   pre-computed embeddings at write time. Visible symptom: ``ilma recall``
   returns NaN-similarity results and either no memories or arbitrary
   ones.

3. **Zero chunks** in ``ilma.memory_chunks``: source had no chunks to
   copy. The hybrid recall still works for short memories via the
   parent-table FTS branch, but long memories are missed.

This module is idempotent and safe to re-run.
"""

from __future__ import annotations

from typing import Any

import psycopg

from ilma.embeddings import EmbedderRegistry


def _looks_corrupt_tags(tags: list[str]) -> bool:
    """Heuristic: a stored ``text[]`` array is corrupt if it has many
    single-character or punctuation-only elements that look like a string
    that got split character-by-character.
    """
    if not tags:
        return False
    if len(tags) > 30:
        return True
    char_only = sum(1 for t in tags if len(t) <= 1 or t in {",", " ", '"'})
    return char_only / len(tags) > 0.6


def _parse_proper_tags(content: str) -> list[str]:
    """Re-derive tags from content keywords when the stored tags are corrupt.

    The original (pre-corruption) tags for known roleplay identities were
    typically ``identity, roleplay, loyalty, alfred, bruce-wayne,
    rezaur-rahman``. We recover them by scanning for known anchor words.
    For unknown content, we fall back to a single ``uncategorized`` tag.
    """
    lower = content.lower()
    candidates: list[str] = []
    rules: list[tuple[str, str]] = [
        ("identity", "user-identity"),
        ("roleplay", "roleplay"),
        ("loyalty", "loyalty"),
        ("alfred", "alfred"),
        ("bruce wayne", "bruce-wayne"),
        ("shakib", "shakib"),
        ("master wayne", "roleplay"),
    ]
    for needle, tag in rules:
        if needle in lower and tag not in candidates:
            candidates.append(tag)
    return candidates or ["uncategorized"]


def _chunk_content(content: str, *, target_chars: int = 512, max_chars: int = 1024) -> list[str]:
    """Naive char-based chunker that prefers sentence boundaries."""
    if len(content) <= max_chars:
        return [content]
    chunks: list[str] = []
    cursor = 0
    n = len(content)
    while cursor < n:
        end = min(cursor + target_chars, n)
        if end < n:
            # try to back up to a sentence boundary
            for sep in (". ", "! ", "? ", "\n", " "):
                idx = content.rfind(sep, cursor + target_chars // 2, end)
                if idx != -1:
                    end = idx + len(sep)
                    break
        chunks.append(content[cursor:end].strip())
        cursor = end
    return [c for c in chunks if c]


def _to_vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.10f}" for x in vec) + "]"


def _is_zero_vec(vec: list[float]) -> bool:
    return all(abs(x) < 1e-12 for x in vec)


def repair_migrated_memories(
    connection: psycopg.Connection[Any],
    *,
    embedder_registry: EmbedderRegistry,
) -> dict[str, Any]:
    """Repair the three known migration artifacts on the given connection.

    Caller is responsible for opening the connection and passing the
    embedder registry (typically ``service.memory._embedders``). The
    function commits its own changes via the passed connection's
    transaction.
    """
    stats: dict[str, Any] = {
        "ok": True,
        "tags_cleaned": 0,
        "memories_embedded": 0,
        "chunks_created": 0,
        "embedder_dim": embedder_registry.default_dim,
        "skipped": "",
    }

    # 1. Tag cleanup
    with connection.cursor() as cur:
        cur.execute(
            "SELECT id, tags, content FROM ilma.memories WHERE deleted_at IS NULL ORDER BY id"
        )
        for memory_id, tags, content in cur.fetchall():
            tag_list = list(tags or [])
            if not _looks_corrupt_tags(tag_list):
                continue
            new_tags = _parse_proper_tags(content)
            cur.execute(
                "UPDATE ilma.memories SET tags = %s WHERE id = %s",
                (new_tags, memory_id),
            )
            stats["tags_cleaned"] += 1

    # 2. Re-embed memories with zero vectors. The dim column to write
    # matches the configured embedder default_dim.
    dim = embedder_registry.default_dim
    vec_col = f"vector_{dim}"
    with connection.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, content FROM ilma.memories
            WHERE deleted_at IS NULL
              AND ({vec_col} IS NULL OR {vec_col}::text LIKE '[%0,0,0,0%')
            ORDER BY id
            """
        )
        for memory_id, content in cur.fetchall():
            vec = embedder_registry.embed(content, dim=dim)
            if _is_zero_vec(vec):
                continue
            cur.execute(
                f"UPDATE ilma.memories SET {vec_col} = %s::vector WHERE id = %s",
                (_to_vec_literal(vec), memory_id),
            )
            stats["memories_embedded"] += 1

    # 3. Chunk long content that has no chunks
    with connection.cursor() as cur:
        cur.execute(
            """
            SELECT m.id, m.content, COUNT(c.id)
            FROM ilma.memories m
            LEFT JOIN ilma.memory_chunks c ON c.memory_id = m.id
            WHERE m.deleted_at IS NULL
            GROUP BY m.id, m.content
            HAVING COUNT(c.id) = 0 AND length(m.content) > 200
            ORDER BY m.id
            """
        )
        for memory_id, content, _ in cur.fetchall():
            for idx, piece in enumerate(_chunk_content(content)):
                vec = embedder_registry.embed(piece, dim=dim)
                if _is_zero_vec(vec):
                    continue
                cur.execute(
                    f"""
                    INSERT INTO ilma.memory_chunks
                        (memory_id, chunk_index, content, token_count, metadata,
                         vector_768, vector_{dim}, vector_1536)
                    VALUES (%s, %s, %s, %s, %s, NULL, %s::vector, NULL)
                    ON CONFLICT (memory_id, chunk_index) DO NOTHING
                    """,
                    (
                        memory_id,
                        idx,
                        piece,
                        len(piece.split()),
                        '{"repaired_by": "ilma.migrate --reembed"}',
                        _to_vec_literal(vec),
                    ),
                )
                stats["chunks_created"] += 1

    connection.commit()
    return stats
