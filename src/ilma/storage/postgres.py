"""Postgres storage backend and MemoryRepo implementation.

Framework-agnostic port of hermes-memory's PgMemoryRepo. Uses psycopg3,
psycopg_pool, Postgres full-text search, and pgvector chunk embeddings.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from ilma.chunking import chunk_text
from ilma.core.memory import (
    MEMORY_MAX_CHARS,
    Memory,
    MemoryNotFoundError,
    MemoryRepo,
    RoutingRuleViolationError,
    routing_rule_message,
)
from ilma.embeddings import DEFAULT_DIM, SUPPORTED_DIMS, EmbedderRegistry
from ilma.storage.base import StorageBackend

_POOL_CACHE: dict[tuple[str, int, int], ConnectionPool] = {}
_POOL_LOCK = threading.Lock()


def _get_pool(dsn: str, *, min_size: int = 1, max_size: int = 8) -> ConnectionPool:
    """Return a shared psycopg3 connection pool for a DSN/size tuple."""
    key = (dsn, min_size, max_size)
    pool = _POOL_CACHE.get(key)
    if pool is None:
        with _POOL_LOCK:
            pool = _POOL_CACHE.get(key)
            if pool is None:
                pool = ConnectionPool(
                    conninfo=dsn,
                    min_size=min_size,
                    max_size=max_size,
                    kwargs={"autocommit": False},
                    open=True,
                )
                pool.wait(timeout=30.0)
                _POOL_CACHE[key] = pool
    return pool


def close_all_pools() -> None:
    """Close and clear all process-global Postgres pools."""
    with _POOL_LOCK:
        pools = list(_POOL_CACHE.values())
        _POOL_CACHE.clear()
    for pool in pools:
        pool.close()


@contextmanager
def _conn(dsn: str, *, min_size: int = 1, max_size: int = 8):
    pool = _get_pool(dsn, min_size=min_size, max_size=max_size)
    with pool.connection() as connection:
        yield connection


def _vector_literal(vector: Sequence[float]) -> str:
    """Serialize a Python float sequence to pgvector's '[...]' text form."""
    values: list[str] = []
    for value in vector:
        f = float(value)
        if not math.isfinite(f):
            f = 0.0
        values.append(repr(f))
    return "[" + ",".join(values) + "]"


def _vector_col(dim: int) -> str:
    if dim not in SUPPORTED_DIMS:
        msg = f"unsupported vector dim {dim}; choose one of {SUPPORTED_DIMS}"
        raise ValueError(msg)
    return f"vector_{dim}"


def _jsonb(value: Any) -> Jsonb:
    return Jsonb(value if value is not None else {})


class PgMemoryRepo(MemoryRepo):
    """Postgres + pgvector implementation of :class:`ilma.core.memory.MemoryRepo`."""

    def __init__(
        self,
        dsn: str,
        *,
        embedders: Any | None = None,
        min_pool_size: int = 1,
        max_pool_size: int = 8,
        initialize: bool = True,
    ) -> None:
        self._dsn = dsn
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._embedders = embedders or EmbedderRegistry.from_env()
        self.default_dim = int(getattr(self._embedders, "default_dim", DEFAULT_DIM))
        _vector_col(self.default_dim)
        self._pool = _get_pool(dsn, min_size=min_pool_size, max_size=max_pool_size)
        if initialize:
            self.initialize_schema()
        with self._pool.connection() as connection:
            connection.execute("SELECT 1")

    def initialize_schema(self) -> None:
        """Create the pgvector extension and ilma memory tables if absent."""
        with self._pool.connection() as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.execute("CREATE SCHEMA IF NOT EXISTS ilma")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ilma.memories (
                    id bigserial PRIMARY KEY,
                    content text NOT NULL,
                    tags text[] NOT NULL DEFAULT '{}',
                    category text,
                    source text,
                    metadata jsonb NOT NULL DEFAULT '{}',
                    vector_768 vector(768),
                    vector_1024 vector(1024),
                    vector_1536 vector(1536),
                    content_tsv tsvector GENERATED ALWAYS AS
                        (to_tsvector('english', coalesce(content, ''))) STORED,
                    created_at timestamptz NOT NULL DEFAULT now(),
                    deleted_at timestamptz
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ilma.memory_chunks (
                    id bigserial PRIMARY KEY,
                    memory_id bigint NOT NULL REFERENCES ilma.memories(id) ON DELETE CASCADE,
                    chunk_index integer NOT NULL,
                    content text NOT NULL,
                    token_count integer NOT NULL,
                    metadata jsonb NOT NULL DEFAULT '{}',
                    vector_768 vector(768),
                    vector_1024 vector(1024),
                    vector_1536 vector(1536),
                    content_tsv tsvector GENERATED ALWAYS AS
                        (to_tsvector('english', coalesce(content, ''))) STORED,
                    created_at timestamptz NOT NULL DEFAULT now(),
                    UNIQUE(memory_id, chunk_index)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS memories_content_tsv_idx "
                "ON ilma.memories USING gin(content_tsv)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS memory_chunks_content_tsv_idx "
                "ON ilma.memory_chunks USING gin(content_tsv)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS memory_chunks_memory_id_idx "
                "ON ilma.memory_chunks(memory_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS memories_live_idx "
                "ON ilma.memories(id) WHERE deleted_at IS NULL"
            )

    def remember(
        self,
        content: str,
        *,
        tags: Sequence[str] | None = None,
        category: str | None = None,
        source: str | None = None,
    ) -> int:
        """Store a memory. Returns the new id, or 0 if duplicate."""
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string")
        if len(content) > MEMORY_MAX_CHARS:
            raise RoutingRuleViolationError(
                routing_rule_message(size_bytes=len(content), cap=MEMORY_MAX_CHARS)
            )
        return self._insert_memory(
            content,
            tags=list(tags) if tags else [],
            category=category,
            source=source,
            embedding_dim=self.default_dim,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        hybrid_text_weight: float = 0.5,
    ) -> list[Memory]:
        """Run hybrid search over parent FTS, parent vectors, and chunk vectors."""
        if not isinstance(query, str) or not query.strip() or top_k <= 0:
            return []
        text_weight = max(0.0, min(1.0, float(hybrid_text_weight)))
        query_embedding = self._embed_query(query)
        return self._search(
            query_embedding,
            query,
            top_k=top_k,
            hybrid_text_weight=text_weight,
        )

    def forget(self, memory_id: int) -> bool:
        """Soft-delete a memory."""
        if not isinstance(memory_id, int) or memory_id <= 0:
            raise MemoryNotFoundError(f"invalid memory id: {memory_id}")
        with self._pool.connection() as connection:
            row = connection.execute(
                """
                UPDATE ilma.memories
                SET deleted_at = now()
                WHERE id = %s AND deleted_at IS NULL
                RETURNING id
                """,
                (memory_id,),
            ).fetchone()
            return row is not None

    def status(self) -> dict[str, Any]:
        """Return repository statistics."""
        with self._pool.connection() as connection:
            row = connection.execute(
                """
                SELECT
                    (SELECT count(*) FROM ilma.memories) AS total,
                    (SELECT count(*) FROM ilma.memories WHERE deleted_at IS NULL) AS live,
                    (SELECT count(*) FROM ilma.memory_chunks) AS chunks
                """
            ).fetchone()
        assert row is not None
        return {
            "total_memories": row[0],
            "live_memories": row[1],
            "total_chunks": row[2],
            "default_dim": self.default_dim,
        }

    def close(self) -> None:
        """Instance close hook. Pools are process-shared and owned globally."""

    def _embed_query(self, query: str) -> list[float]:
        return [float(x) for x in self._embedders.embed(query, dim=self.default_dim)]

    def _insert_memory(
        self,
        content: str,
        *,
        tags: list[str],
        category: str | None,
        source: str | None,
        embedding_dim: int,
    ) -> int:
        vector_col = _vector_col(embedding_dim)
        with self._pool.connection() as connection:
            duplicate = connection.execute(
                """
                SELECT id FROM ilma.memories
                WHERE content = %s
                  AND source IS NOT DISTINCT FROM %s
                  AND deleted_at IS NULL
                LIMIT 1
                """,
                (content, source),
            ).fetchone()
            if duplicate is not None:
                return 0

            row = connection.execute(
                """
                INSERT INTO ilma.memories (content, tags, category, source)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (content, tags, category, source),
            ).fetchone()
            assert row is not None
            memory_id = int(row[0])

            chunks = chunk_text(content)
            for chunk in chunks:
                embedding = self._embedders.embed(chunk.text, dim=embedding_dim)
                connection.execute(
                    f"""
                    INSERT INTO ilma.memory_chunks
                        (memory_id, chunk_index, content, token_count, {vector_col})
                    VALUES (%s, %s, %s, %s, %s::vector)
                    ON CONFLICT (memory_id, chunk_index) DO NOTHING
                    """,
                    (
                        memory_id,
                        chunk.index,
                        chunk.text,
                        chunk.token_count,
                        _vector_literal(embedding),
                    ),
                )

            if not chunks or len(content) <= 2048:
                embedding = self._embedders.embed(content, dim=embedding_dim)
                connection.execute(
                    f"UPDATE ilma.memories SET {vector_col} = %s::vector WHERE id = %s",
                    (_vector_literal(embedding), memory_id),
                )
            return memory_id

    def _search(
        self,
        query_embedding: Sequence[float],
        query_text: str,
        *,
        top_k: int,
        hybrid_text_weight: float,
    ) -> list[Memory]:
        vector_col = _vector_col(self.default_dim)
        query_vector = _vector_literal(query_embedding)
        limit = max(top_k * 4, top_k)
        text_weight = hybrid_text_weight
        vector_weight = 1.0 - text_weight

        with (
            self._pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute(
                """
                SELECT id, ts_rank_cd(content_tsv, plainto_tsquery('english', %s)) AS score
                FROM ilma.memories
                WHERE deleted_at IS NULL
                  AND content_tsv @@ plainto_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s
                """,
                (query_text, query_text, limit),
            )
            parent_fts = {int(row["id"]): float(row["score"] or 0.0) for row in cursor.fetchall()}

            cursor.execute(
                """
                SELECT c.memory_id,
                       max(ts_rank_cd(c.content_tsv, plainto_tsquery('english', %s))) AS score
                FROM ilma.memory_chunks c
                JOIN ilma.memories m ON m.id = c.memory_id
                WHERE m.deleted_at IS NULL
                  AND c.content_tsv @@ plainto_tsquery('english', %s)
                GROUP BY c.memory_id
                ORDER BY score DESC
                LIMIT %s
                """,
                (query_text, query_text, limit),
            )
            chunk_fts = {
                int(row["memory_id"]): float(row["score"] or 0.0) for row in cursor.fetchall()
            }

            cursor.execute(
                f"""
                SELECT id, 1 - ({vector_col} <=> %s::vector) AS score
                FROM ilma.memories
                WHERE deleted_at IS NULL AND {vector_col} IS NOT NULL
                ORDER BY {vector_col} <=> %s::vector
                LIMIT %s
                """,
                (query_vector, query_vector, limit),
            )
            parent_vectors = {
                int(row["id"]): float(row["score"] or 0.0) for row in cursor.fetchall()
            }

            cursor.execute(
                f"""
                SELECT c.memory_id, max(1 - (c.{vector_col} <=> %s::vector)) AS score
                FROM ilma.memory_chunks c
                JOIN ilma.memories m ON m.id = c.memory_id
                WHERE m.deleted_at IS NULL AND c.{vector_col} IS NOT NULL
                GROUP BY c.memory_id
                ORDER BY min(c.{vector_col} <=> %s::vector)
                LIMIT %s
                """,
                (query_vector, query_vector, limit),
            )
            chunk_vectors = {
                int(row["memory_id"]): float(row["score"] or 0.0) for row in cursor.fetchall()
            }

            memory_ids = set(parent_fts) | set(chunk_fts) | set(parent_vectors) | set(chunk_vectors)
            scored: dict[int, float] = {}
            for memory_id in memory_ids:
                score = 0.0
                score += text_weight * parent_fts.get(memory_id, 0.0)
                score += text_weight * 0.75 * chunk_fts.get(memory_id, 0.0)
                score += vector_weight * parent_vectors.get(memory_id, 0.0)
                score += vector_weight * 0.75 * chunk_vectors.get(memory_id, 0.0)
                if score > 0.0:
                    scored[memory_id] = score

            top = sorted(scored.items(), key=lambda item: (-item[1], item[0]))[:top_k]
            if not top:
                return []

            cursor.execute(
                """
                SELECT id, content, tags, category, source, metadata,
                       deleted_at IS NOT NULL AS deleted, created_at
                FROM ilma.memories
                WHERE id = ANY(%s)
                """,
                ([memory_id for memory_id, _score in top],),
            )
            by_id = {int(row["id"]): row for row in cursor.fetchall()}

        results: list[Memory] = []
        for memory_id, _score in top:
            row = by_id.get(memory_id)
            if row is None:
                continue
            created_at = row["created_at"]
            if created_at is not None and not isinstance(created_at, datetime):
                created_at = None
            results.append(
                Memory(
                    id=memory_id,
                    content=str(row["content"]),
                    tags=tuple(row["tags"] or ()),
                    category=row["category"],
                    source=row["source"],
                    embedding_dim=self.default_dim,
                    deleted=bool(row["deleted"]),
                    created_at=created_at,
                    metadata=dict(row["metadata"] or {}),
                )
            )
        return results


class PgBackend(StorageBackend):
    """Small generic pgvector backend implementing :class:`StorageBackend`."""

    name = "postgres"

    def __init__(
        self,
        dsn: str,
        *,
        min_pool_size: int = 1,
        max_pool_size: int = 8,
        initialize: bool = True,
    ) -> None:
        self._dsn = dsn
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._pool = _get_pool(dsn, min_size=min_pool_size, max_size=max_pool_size)
        if initialize:
            self.initialize_schema()

    def initialize_schema(self) -> None:
        with self._pool.connection() as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.execute("CREATE SCHEMA IF NOT EXISTS ilma")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ilma.backend_items (
                    collection text NOT NULL,
                    id text NOT NULL,
                    document text NOT NULL,
                    embedding vector,
                    metadata jsonb NOT NULL DEFAULT '{}',
                    created_at timestamptz NOT NULL DEFAULT now(),
                    updated_at timestamptz NOT NULL DEFAULT now(),
                    PRIMARY KEY(collection, id)
                )
                """
            )

    def memory_repo(self, *, embedders: Any | None = None) -> PgMemoryRepo:
        return PgMemoryRepo(
            self._dsn,
            embedders=embedders,
            min_pool_size=self._min_pool_size,
            max_pool_size=self._max_pool_size,
        )

    def add(
        self,
        collection: str,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        self._validate_batch(ids, documents, embeddings)
        metadata_batch = metadatas or [{} for _ in ids]
        with self._pool.connection() as connection:
            for item_id, document, embedding, metadata in zip(
                ids, documents, embeddings, metadata_batch, strict=True
            ):
                connection.execute(
                    """
                    INSERT INTO ilma.backend_items
                        (collection, id, document, embedding, metadata)
                    VALUES (%s, %s, %s, %s::vector, %s)
                    """,
                    (collection, item_id, document, _vector_literal(embedding), _jsonb(metadata)),
                )

    def upsert(
        self,
        collection: str,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        self._validate_batch(ids, documents, embeddings)
        metadata_batch = metadatas or [{} for _ in ids]
        with self._pool.connection() as connection:
            for item_id, document, embedding, metadata in zip(
                ids, documents, embeddings, metadata_batch, strict=True
            ):
                connection.execute(
                    """
                    INSERT INTO ilma.backend_items
                        (collection, id, document, embedding, metadata)
                    VALUES (%s, %s, %s, %s::vector, %s)
                    ON CONFLICT (collection, id) DO UPDATE
                    SET document = EXCLUDED.document,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        updated_at = now()
                    """,
                    (collection, item_id, document, _vector_literal(embedding), _jsonb(metadata)),
                )

    def query(
        self,
        collection: str,
        query_embeddings: list[list[float]],
        *,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not query_embeddings or n_results <= 0:
            return []
        query_vector = _vector_literal(query_embeddings[0])
        sql = [
            "SELECT id, document, metadata, embedding <=> %s::vector AS distance ",
            "FROM ilma.backend_items WHERE collection = %s ",
        ]
        params: list[Any] = [query_vector, collection]
        self._append_metadata_filter(sql, params, where)
        sql.append("ORDER BY embedding <=> %s::vector LIMIT %s")
        params.extend([query_vector, n_results])
        with (
            self._pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute("".join(sql), params)
            return [dict(row) for row in cursor.fetchall()]

    def get(
        self,
        collection: str,
        ids: list[str] | None = None,
        *,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        sql = ["SELECT id, document, metadata FROM ilma.backend_items WHERE collection = %s "]
        params: list[Any] = [collection]
        if ids is not None:
            sql.append("AND id = ANY(%s) ")
            params.append(ids)
        self._append_metadata_filter(sql, params, where)
        sql.append("ORDER BY id")
        with (
            self._pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute("".join(sql), params)
            return [dict(row) for row in cursor.fetchall()]

    def delete(
        self,
        collection: str,
        ids: list[str] | None = None,
        *,
        where: dict[str, Any] | None = None,
    ) -> None:
        sql = ["DELETE FROM ilma.backend_items WHERE collection = %s "]
        params: list[Any] = [collection]
        if ids is not None:
            sql.append("AND id = ANY(%s) ")
            params.append(ids)
        self._append_metadata_filter(sql, params, where)
        with self._pool.connection() as connection:
            connection.execute("".join(sql), params)

    def count(self, collection: str) -> int:
        with self._pool.connection() as connection:
            row = connection.execute(
                "SELECT count(*) FROM ilma.backend_items WHERE collection = %s",
                (collection,),
            ).fetchone()
        assert row is not None
        return int(row[0])

    def health(self) -> dict[str, Any]:
        with self._pool.connection() as connection:
            row = connection.execute(
                """
                SELECT current_database(),
                       EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')
                """
            ).fetchone()
        assert row is not None
        return {"ok": True, "database": row[0], "pgvector": bool(row[1])}

    @staticmethod
    def _validate_batch(
        ids: list[str], documents: list[str], embeddings: list[list[float]]
    ) -> None:
        if not (len(ids) == len(documents) == len(embeddings)):
            raise ValueError("ids, documents, and embeddings must have the same length")

    @staticmethod
    def _append_metadata_filter(
        sql: list[str], params: list[Any], where: dict[str, Any] | None
    ) -> None:
        if not where:
            return
        for key, value in where.items():
            sql.append("AND metadata ->> %s = %s ")
            params.extend([key, str(value)])


__all__ = ["PgBackend", "PgMemoryRepo", "close_all_pools"]
