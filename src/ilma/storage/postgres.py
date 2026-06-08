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
from ilma.core.journal import JournalEntry, JournalRepo
from ilma.core.kanban import KanbanRepo, Task
from ilma.core.memory import (
    MEMORY_MAX_CHARS,
    Memory,
    MemoryNotFoundError,
    MemoryRepo,
    RoutingRuleViolationError,
    routing_rule_message,
)
from ilma.core.metrics import Metric, MetricsRepo
from ilma.core.observability import ObservabilityRepo, Observation
from ilma.core.sessions import SessionMessage, SessionsRepo
from ilma.core.skills import Skill, SkillsRepo
from ilma.core.wiki import WikiDoc, WikiRepo
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
def _conn(dsn: str, *, min_size: int = 1, max_size: int = 8) -> Any:
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


class _PgRepoMixin:
    """Shared constructor/schema helper for the surface-specific PG repos."""

    def initialize_schema(self) -> None:
        raise NotImplementedError

    def _init_pg_repo(
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
        with self._pool.connection() as connection:
            connection.execute("SELECT 1")

    def _ensure_schema(self, *, vector: bool = False) -> None:
        with self._pool.connection() as connection:
            if vector:
                connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.execute("CREATE SCHEMA IF NOT EXISTS ilma")


def _as_datetime(value: Any) -> datetime | None:
    return value if isinstance(value, datetime) else None


def _bucket_to_pg_unit(bucket: str) -> str:
    """Translate human bucket strings to Postgres date_trunc units."""
    b = bucket.strip().lower()
    if b.endswith(("minute", "minutes", "min", "mins")):
        return "minute"
    if b.endswith(("hour", "hours", "hr", "hrs")):
        return "hour"
    if b.endswith(("day", "days")):
        return "day"
    if b.endswith(("week", "weeks")):
        return "week"
    if b.endswith(("month", "months")):
        return "month"
    return "hour"


class PgWikiRepo(_PgRepoMixin, WikiRepo):
    """Postgres + pgvector implementation of the wiki surface."""

    def __init__(
        self,
        dsn: str,
        *,
        embedders: Any | None = None,
        min_pool_size: int = 1,
        max_pool_size: int = 8,
        initialize: bool = True,
    ) -> None:
        self._embedders = embedders or EmbedderRegistry.from_env()
        self.default_dim = int(getattr(self._embedders, "default_dim", DEFAULT_DIM))
        _vector_col(self.default_dim)
        self._init_pg_repo(
            dsn,
            min_pool_size=min_pool_size,
            max_pool_size=max_pool_size,
            initialize=initialize,
        )

    def initialize_schema(self) -> None:
        self._ensure_schema(vector=True)
        with self._pool.connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ilma.wiki_docs (
                    id bigserial PRIMARY KEY,
                    slug text NOT NULL UNIQUE,
                    title text NOT NULL,
                    body_md text NOT NULL,
                    category text,
                    tags text[] NOT NULL DEFAULT '{}',
                    source_uri text,
                    version integer NOT NULL DEFAULT 1,
                    metadata jsonb NOT NULL DEFAULT '{}',
                    content_tsv tsvector GENERATED ALWAYS AS
                        (to_tsvector('english', coalesce(title, '') || ' ' || coalesce(body_md, ''))) STORED,
                    created_at timestamptz NOT NULL DEFAULT now(),
                    updated_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ilma.wiki_chunks (
                    id bigserial PRIMARY KEY,
                    doc_id bigint NOT NULL REFERENCES ilma.wiki_docs(id) ON DELETE CASCADE,
                    chunk_index integer NOT NULL,
                    content text NOT NULL,
                    token_count integer NOT NULL,
                    vector_768 vector(768),
                    vector_1024 vector(1024),
                    vector_1536 vector(1536),
                    content_tsv tsvector GENERATED ALWAYS AS
                        (to_tsvector('english', coalesce(content, ''))) STORED,
                    created_at timestamptz NOT NULL DEFAULT now(),
                    UNIQUE(doc_id, chunk_index)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS wiki_docs_slug_idx ON ilma.wiki_docs(slug)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS wiki_docs_content_tsv_idx ON ilma.wiki_docs USING gin(content_tsv)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS wiki_chunks_doc_id_idx ON ilma.wiki_chunks(doc_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS wiki_chunks_content_tsv_idx ON ilma.wiki_chunks USING gin(content_tsv)"
            )

    def ingest(
        self,
        slug: str,
        title: str,
        body_md: str,
        *,
        category: str | None = None,
        tags: list[str] | None = None,
        source_uri: str | None = None,
    ) -> dict[str, Any]:
        if not slug.strip() or not title.strip():
            raise ValueError("slug and title are required")
        vector_col = _vector_col(self.default_dim)
        with self._pool.connection() as connection:
            row = connection.execute(
                """
                INSERT INTO ilma.wiki_docs (slug, title, body_md, category, tags, source_uri)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (slug) DO UPDATE
                SET title = EXCLUDED.title,
                    body_md = EXCLUDED.body_md,
                    category = EXCLUDED.category,
                    tags = EXCLUDED.tags,
                    source_uri = EXCLUDED.source_uri,
                    version = ilma.wiki_docs.version + 1,
                    updated_at = now()
                RETURNING id, version
                """,
                (slug, title, body_md, category, tags or [], source_uri),
            ).fetchone()
            assert row is not None
            doc_id = int(row[0])
            version = int(row[1])
            connection.execute("DELETE FROM ilma.wiki_chunks WHERE doc_id = %s", (doc_id,))
            chunks = chunk_text(f"{title}\n\n{body_md}")
            for chunk in chunks:
                embedding = self._embedders.embed(chunk.text, dim=self.default_dim)
                connection.execute(
                    f"""
                    INSERT INTO ilma.wiki_chunks
                        (doc_id, chunk_index, content, token_count, {vector_col})
                    VALUES (%s, %s, %s, %s, %s::vector)
                    """,
                    (
                        doc_id,
                        chunk.index,
                        chunk.text,
                        chunk.token_count,
                        _vector_literal(embedding),
                    ),
                )
        return {"document_id": doc_id, "version_id": version, "chunks": len(chunks)}

    def get(self, slug: str) -> WikiDoc | None:
        with (
            self._pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute(
                """
                SELECT id, slug, title, body_md, category, tags, source_uri,
                       version, created_at, updated_at, metadata
                FROM ilma.wiki_docs
                WHERE slug = %s
                """,
                (slug,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return WikiDoc(
            id=int(row["id"]),
            slug=str(row["slug"]),
            title=str(row["title"]),
            body_md=str(row["body_md"]),
            category=row["category"],
            tags=tuple(row["tags"] or ()),
            source_uri=row["source_uri"],
            version=int(row["version"]),
            created_at=_as_datetime(row["created_at"]),
            updated_at=_as_datetime(row["updated_at"]),
            metadata=dict(row["metadata"] or {}),
        )

    def search(self, query: str, *, top_k: int = 5) -> list[dict[str, Any]]:
        if not query.strip() or top_k <= 0:
            return []
        vector_col = _vector_col(self.default_dim)
        query_vector = _vector_literal(self._embedders.embed(query, dim=self.default_dim))
        limit = max(top_k * 4, top_k)
        with (
            self._pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute(
                """
                SELECT id, ts_rank_cd(content_tsv, plainto_tsquery('english', %s)) AS score
                FROM ilma.wiki_docs
                WHERE content_tsv @@ plainto_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s
                """,
                (query, query, limit),
            )
            doc_fts = {int(row["id"]): float(row["score"] or 0.0) for row in cursor.fetchall()}
            cursor.execute(
                """
                SELECT doc_id, max(ts_rank_cd(content_tsv, plainto_tsquery('english', %s))) AS score
                FROM ilma.wiki_chunks
                WHERE content_tsv @@ plainto_tsquery('english', %s)
                GROUP BY doc_id
                ORDER BY score DESC
                LIMIT %s
                """,
                (query, query, limit),
            )
            chunk_fts = {
                int(row["doc_id"]): float(row["score"] or 0.0) for row in cursor.fetchall()
            }
            cursor.execute(
                f"""
                SELECT doc_id, max(1 - ({vector_col} <=> %s::vector)) AS score
                FROM ilma.wiki_chunks
                WHERE {vector_col} IS NOT NULL
                GROUP BY doc_id
                ORDER BY min({vector_col} <=> %s::vector)
                LIMIT %s
                """,
                (query_vector, query_vector, limit),
            )
            vectors = {int(row["doc_id"]): float(row["score"] or 0.0) for row in cursor.fetchall()}
            doc_ids = set(doc_fts) | set(chunk_fts) | set(vectors)
            scored = {
                doc_id: doc_fts.get(doc_id, 0.0)
                + 0.75 * chunk_fts.get(doc_id, 0.0)
                + vectors.get(doc_id, 0.0)
                for doc_id in doc_ids
            }
            top = sorted(scored.items(), key=lambda item: (-item[1], item[0]))[:top_k]
            if not top:
                return []
            cursor.execute(
                """
                SELECT id, slug, title, body_md, category, tags, source_uri, version,
                       created_at, updated_at, metadata
                FROM ilma.wiki_docs
                WHERE id = ANY(%s)
                """,
                ([doc_id for doc_id, _score in top],),
            )
            docs = {int(row["id"]): row for row in cursor.fetchall()}
        results: list[dict[str, Any]] = []
        for doc_id, score in top:
            row = docs.get(doc_id)
            if row is None:
                continue
            results.append(
                {
                    "id": doc_id,
                    "document_id": doc_id,
                    "slug": row["slug"],
                    "title": row["title"],
                    "category": row["category"],
                    "tags": tuple(row["tags"] or ()),
                    "source_uri": row["source_uri"],
                    "version": int(row["version"]),
                    "score": float(score),
                    "snippet": str(row["body_md"])[:240],
                    "metadata": dict(row["metadata"] or {}),
                }
            )
        return results

    def suggest_links(self, doc_id: int, *, top_k: int = 5) -> list[dict[str, Any]]:
        if doc_id <= 0 or top_k <= 0:
            return []
        vector_col = _vector_col(self.default_dim)
        with (
            self._pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute(
                f"""
                WITH query_chunk AS (
                    SELECT {vector_col} AS embedding
                    FROM ilma.wiki_chunks
                    WHERE doc_id = %s AND {vector_col} IS NOT NULL
                    ORDER BY chunk_index
                    LIMIT 1
                )
                SELECT d.id, d.slug, d.title,
                       max(1 - (c.{vector_col} <=> q.embedding)) AS score
                FROM query_chunk q
                JOIN ilma.wiki_chunks c ON c.{vector_col} IS NOT NULL
                JOIN ilma.wiki_docs d ON d.id = c.doc_id
                WHERE d.id <> %s
                GROUP BY d.id, d.slug, d.title
                ORDER BY min(c.{vector_col} <=> q.embedding), d.id
                LIMIT %s
                """,
                (doc_id, doc_id, top_k),
            )
            return [dict(row) for row in cursor.fetchall()]


class PgJournalRepo(_PgRepoMixin, JournalRepo):
    """Postgres implementation of the journal surface."""

    def __init__(
        self,
        dsn: str,
        *,
        min_pool_size: int = 1,
        max_pool_size: int = 8,
        initialize: bool = True,
    ) -> None:
        self._init_pg_repo(
            dsn,
            min_pool_size=min_pool_size,
            max_pool_size=max_pool_size,
            initialize=initialize,
        )

    def initialize_schema(self) -> None:
        self._ensure_schema()
        with self._pool.connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ilma.journal_entries (
                    id bigserial PRIMARY KEY,
                    content text NOT NULL,
                    tags text[] NOT NULL DEFAULT '{}',
                    content_tsv tsvector GENERATED ALWAYS AS
                        (to_tsvector('english', coalesce(content, ''))) STORED,
                    created_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS journal_entries_content_tsv_idx ON ilma.journal_entries USING gin(content_tsv)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS journal_entries_created_at_idx ON ilma.journal_entries(created_at DESC)"
            )

    def append(self, content: str, *, tags: list[str] | None = None) -> int:
        if not content.strip():
            raise ValueError("content is required")
        with self._pool.connection() as connection:
            row = connection.execute(
                "INSERT INTO ilma.journal_entries (content, tags) VALUES (%s, %s) RETURNING id",
                (content, tags or []),
            ).fetchone()
        assert row is not None
        return int(row[0])

    def search(self, query: str, *, top_k: int = 10) -> list[JournalEntry]:
        if not query.strip() or top_k <= 0:
            return []
        with (
            self._pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute(
                """
                SELECT id, content, tags, created_at,
                       ts_rank_cd(content_tsv, plainto_tsquery('english', %s)) AS score
                FROM ilma.journal_entries
                WHERE content_tsv @@ plainto_tsquery('english', %s)
                ORDER BY score DESC, created_at DESC
                LIMIT %s
                """,
                (query, query, top_k),
            )
            return [
                JournalEntry(
                    id=int(row["id"]),
                    content=str(row["content"]),
                    tags=tuple(row["tags"] or ()),
                    created_at=_as_datetime(row["created_at"]),
                )
                for row in cursor.fetchall()
            ]

    def recent(self, *, limit: int = 10) -> list[JournalEntry]:
        with (
            self._pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute(
                """
                SELECT id, content, tags, created_at
                FROM ilma.journal_entries
                ORDER BY created_at DESC, id DESC
                LIMIT %s
                """,
                (limit,),
            )
            return [
                JournalEntry(
                    id=int(row["id"]),
                    content=str(row["content"]),
                    tags=tuple(row["tags"] or ()),
                    created_at=_as_datetime(row["created_at"]),
                )
                for row in cursor.fetchall()
            ]


class PgSkillsRepo(_PgRepoMixin, SkillsRepo):
    """Postgres implementation of the skills surface."""

    def __init__(
        self,
        dsn: str,
        *,
        min_pool_size: int = 1,
        max_pool_size: int = 8,
        initialize: bool = True,
    ) -> None:
        self._init_pg_repo(
            dsn,
            min_pool_size=min_pool_size,
            max_pool_size=max_pool_size,
            initialize=initialize,
        )

    def initialize_schema(self) -> None:
        self._ensure_schema()
        with self._pool.connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ilma.skills (
                    id bigserial PRIMARY KEY,
                    name text NOT NULL UNIQUE,
                    content text NOT NULL,
                    category text,
                    tags text[] NOT NULL DEFAULT '{}',
                    body_tsv tsvector GENERATED ALWAYS AS
                        (to_tsvector('english', coalesce(name, '') || ' ' || coalesce(content, '') || ' ' || coalesce(category, ''))) STORED,
                    created_at timestamptz NOT NULL DEFAULT now(),
                    updated_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS skills_name_idx ON ilma.skills(name)")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS skills_body_tsv_idx ON ilma.skills USING gin(body_tsv)"
            )

    def upsert(
        self, name: str, content: str, *, category: str | None = None, tags: list[str] | None = None
    ) -> int:
        if not name.strip() or not content.strip():
            raise ValueError("name and content are required")
        with self._pool.connection() as connection:
            row = connection.execute(
                """
                INSERT INTO ilma.skills (name, content, category, tags)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (name) DO UPDATE
                SET content = EXCLUDED.content,
                    category = EXCLUDED.category,
                    tags = EXCLUDED.tags,
                    updated_at = now()
                RETURNING id
                """,
                (name, content, category, tags or []),
            ).fetchone()
        assert row is not None
        return int(row[0])

    def get(self, name: str) -> Skill | None:
        with (
            self._pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute(
                "SELECT id, name, content, category, tags, updated_at FROM ilma.skills WHERE name = %s",
                (name,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return Skill(
            id=int(row["id"]),
            name=str(row["name"]),
            content=str(row["content"]),
            category=row["category"],
            tags=tuple(row["tags"] or ()),
            updated_at=_as_datetime(row["updated_at"]),
        )

    def search(self, query: str, *, top_k: int = 5) -> list[Skill]:
        if not query.strip() or top_k <= 0:
            return []
        with (
            self._pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute(
                """
                SELECT id, name, content, category, tags, updated_at,
                       ts_rank_cd(body_tsv, plainto_tsquery('english', %s)) AS score
                FROM ilma.skills
                WHERE body_tsv @@ plainto_tsquery('english', %s)
                ORDER BY score DESC, updated_at DESC
                LIMIT %s
                """,
                (query, query, top_k),
            )
            return [
                Skill(
                    id=int(row["id"]),
                    name=str(row["name"]),
                    content=str(row["content"]),
                    category=row["category"],
                    tags=tuple(row["tags"] or ()),
                    updated_at=_as_datetime(row["updated_at"]),
                )
                for row in cursor.fetchall()
            ]


class PgMetricsRepo(_PgRepoMixin, MetricsRepo):
    """Postgres implementation of the metrics surface."""

    def __init__(
        self,
        dsn: str,
        *,
        min_pool_size: int = 1,
        max_pool_size: int = 8,
        initialize: bool = True,
    ) -> None:
        self._init_pg_repo(
            dsn,
            min_pool_size=min_pool_size,
            max_pool_size=max_pool_size,
            initialize=initialize,
        )

    def initialize_schema(self) -> None:
        self._ensure_schema()
        with self._pool.connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ilma.metrics (
                    id bigserial PRIMARY KEY,
                    name text NOT NULL,
                    value double precision NOT NULL,
                    labels jsonb NOT NULL DEFAULT '{}',
                    recorded_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS metrics_name_time_idx ON ilma.metrics(name, recorded_at DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS metrics_labels_idx ON ilma.metrics USING gin(labels)"
            )

    def record(self, name: str, value: float, *, labels: dict[str, str] | None = None) -> int:
        if not name.strip():
            raise ValueError("name is required")
        with self._pool.connection() as connection:
            row = connection.execute(
                "INSERT INTO ilma.metrics (name, value, labels) VALUES (%s, %s, %s) RETURNING id",
                (name, float(value), _jsonb(labels or {})),
            ).fetchone()
        assert row is not None
        return int(row[0])

    def query(
        self,
        name: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 100,
    ) -> list[Metric]:
        sql = ["SELECT id, name, value, labels, recorded_at FROM ilma.metrics WHERE name = %s "]
        params: list[Any] = [name]
        if start is not None:
            sql.append("AND recorded_at >= %s ")
            params.append(start)
        if end is not None:
            sql.append("AND recorded_at <= %s ")
            params.append(end)
        sql.append("ORDER BY recorded_at DESC, id DESC LIMIT %s")
        params.append(limit)
        with (
            self._pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute("".join(sql), params)
            return [
                Metric(
                    id=int(row["id"]),
                    name=str(row["name"]),
                    value=float(row["value"]),
                    labels={str(k): str(v) for k, v in dict(row["labels"] or {}).items()},
                    recorded_at=row["recorded_at"],
                )
                for row in cursor.fetchall()
            ]

    def aggregate(self, name: str, *, window: str = "1 hour") -> list[dict[str, Any]]:
        unit = _bucket_to_pg_unit(window)
        with (
            self._pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute(
                """
                SELECT date_trunc(%s, recorded_at) AS window_start,
                       count(*) AS count,
                       avg(value) AS avg,
                       min(value) AS min,
                       max(value) AS max,
                       percentile_cont(0.5) WITHIN GROUP (ORDER BY value) AS p50
                FROM ilma.metrics
                WHERE name = %s
                GROUP BY window_start
                ORDER BY window_start
                """,
                (unit, name),
            )
            return [
                {
                    "window_start": row["window_start"],
                    "count": int(row["count"]),
                    "avg": float(row["avg"] or 0.0),
                    "min": float(row["min"] or 0.0),
                    "max": float(row["max"] or 0.0),
                    "p50": float(row["p50"] or 0.0),
                }
                for row in cursor.fetchall()
            ]


class PgKanbanRepo(_PgRepoMixin, KanbanRepo):
    """Postgres implementation of the kanban surface."""

    def __init__(
        self,
        dsn: str,
        *,
        min_pool_size: int = 1,
        max_pool_size: int = 8,
        initialize: bool = True,
    ) -> None:
        self._init_pg_repo(
            dsn,
            min_pool_size=min_pool_size,
            max_pool_size=max_pool_size,
            initialize=initialize,
        )

    def initialize_schema(self) -> None:
        self._ensure_schema()
        with self._pool.connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ilma.kanban_tasks (
                    id bigserial PRIMARY KEY,
                    title text NOT NULL,
                    description text NOT NULL DEFAULT '',
                    status text NOT NULL DEFAULT 'todo',
                    priority integer NOT NULL DEFAULT 0,
                    tags text[] NOT NULL DEFAULT '{}',
                    parent_id bigint REFERENCES ilma.kanban_tasks(id) ON DELETE SET NULL,
                    metadata jsonb NOT NULL DEFAULT '{}',
                    content_tsv tsvector GENERATED ALWAYS AS
                        (to_tsvector('english', coalesce(title, '') || ' ' || coalesce(description, ''))) STORED,
                    created_at timestamptz NOT NULL DEFAULT now(),
                    updated_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS kanban_tasks_status_idx ON ilma.kanban_tasks(status)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS kanban_tasks_content_tsv_idx ON ilma.kanban_tasks USING gin(content_tsv)"
            )

    def create(
        self,
        title: str,
        *,
        description: str = "",
        status: str = "todo",
        priority: int = 0,
        tags: list[str] | None = None,
        parent_id: int | None = None,
    ) -> int:
        if not title.strip():
            raise ValueError("title is required")
        with self._pool.connection() as connection:
            row = connection.execute(
                """
                INSERT INTO ilma.kanban_tasks
                    (title, description, status, priority, tags, parent_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (title, description, status, int(priority), tags or [], parent_id),
            ).fetchone()
        assert row is not None
        return int(row[0])

    def get(self, task_id: int) -> Task | None:
        with (
            self._pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute(
                """
                SELECT id, title, description, status, priority, tags, parent_id,
                       created_at, updated_at, metadata
                FROM ilma.kanban_tasks WHERE id = %s
                """,
                (task_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return self._task_from_row(row)

    def update(self, task_id: int, **kwargs: Any) -> bool:
        allowed = {"title", "description", "status", "priority", "tags", "parent_id", "metadata"}
        updates: list[str] = []
        params: list[Any] = []
        for key, value in kwargs.items():
            if key not in allowed:
                continue
            updates.append(f"{key} = %s")
            if key == "metadata":
                params.append(_jsonb(value or {}))
            elif key == "tags":
                params.append(value or [])
            else:
                params.append(value)
        if not updates:
            return False
        updates.append("updated_at = now()")
        params.append(task_id)
        with self._pool.connection() as connection:
            row = connection.execute(
                f"UPDATE ilma.kanban_tasks SET {', '.join(updates)} WHERE id = %s RETURNING id",
                params,
            ).fetchone()
        return row is not None

    def complete(self, task_id: int) -> bool:
        return self.update(task_id, status="done")

    def list_by_status(self, status: str, *, limit: int = 50) -> list[Task]:
        with (
            self._pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute(
                """
                SELECT id, title, description, status, priority, tags, parent_id,
                       created_at, updated_at, metadata
                FROM ilma.kanban_tasks
                WHERE status = %s
                ORDER BY priority DESC, created_at, id
                LIMIT %s
                """,
                (status, limit),
            )
            return [self._task_from_row(row) for row in cursor.fetchall()]

    def search(self, query: str, *, top_k: int = 10) -> list[Task]:
        if not query.strip() or top_k <= 0:
            return []
        with (
            self._pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute(
                """
                SELECT id, title, description, status, priority, tags, parent_id,
                       created_at, updated_at, metadata,
                       ts_rank_cd(content_tsv, plainto_tsquery('english', %s)) AS score
                FROM ilma.kanban_tasks
                WHERE content_tsv @@ plainto_tsquery('english', %s)
                ORDER BY score DESC, priority DESC
                LIMIT %s
                """,
                (query, query, top_k),
            )
            return [self._task_from_row(row) for row in cursor.fetchall()]

    @staticmethod
    def _task_from_row(row: Any) -> Task:
        return Task(
            id=int(row["id"]),
            title=str(row["title"]),
            description=str(row["description"] or ""),
            status=str(row["status"]),
            priority=int(row["priority"]),
            tags=tuple(row["tags"] or ()),
            parent_id=row["parent_id"],
            created_at=_as_datetime(row["created_at"]),
            updated_at=_as_datetime(row["updated_at"]),
            metadata=dict(row["metadata"] or {}),
        )


class PgObservabilityRepo(_PgRepoMixin, ObservabilityRepo):
    """Postgres implementation of the observability surface."""

    def __init__(
        self,
        dsn: str,
        *,
        min_pool_size: int = 1,
        max_pool_size: int = 8,
        initialize: bool = True,
    ) -> None:
        self._init_pg_repo(
            dsn,
            min_pool_size=min_pool_size,
            max_pool_size=max_pool_size,
            initialize=initialize,
        )

    def initialize_schema(self) -> None:
        self._ensure_schema()
        with self._pool.connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ilma.observations (
                    id bigserial PRIMARY KEY,
                    level text NOT NULL,
                    message text NOT NULL,
                    source text,
                    context jsonb NOT NULL DEFAULT '{}',
                    recorded_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS observations_query_idx ON ilma.observations(level, source, recorded_at DESC)"
            )

    def log(
        self,
        level: str,
        message: str,
        *,
        source: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> int:
        if not level.strip() or not message.strip():
            raise ValueError("level and message are required")
        with self._pool.connection() as connection:
            row = connection.execute(
                """
                INSERT INTO ilma.observations (level, message, source, context)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (level, message, source, _jsonb(context or {})),
            ).fetchone()
        assert row is not None
        return int(row[0])

    def query(
        self,
        *,
        level: str | None = None,
        source: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 100,
    ) -> list[Observation]:
        sql = [
            "SELECT id, level, message, source, context, recorded_at FROM ilma.observations WHERE true "
        ]
        params: list[Any] = []
        if level is not None:
            sql.append("AND level = %s ")
            params.append(level)
        if source is not None:
            sql.append("AND source = %s ")
            params.append(source)
        if start is not None:
            sql.append("AND recorded_at >= %s ")
            params.append(start)
        if end is not None:
            sql.append("AND recorded_at <= %s ")
            params.append(end)
        sql.append("ORDER BY recorded_at DESC, id DESC LIMIT %s")
        params.append(limit)
        with (
            self._pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute("".join(sql), params)
            return [
                Observation(
                    id=int(row["id"]),
                    level=str(row["level"]),
                    message=str(row["message"]),
                    source=row["source"],
                    context=dict(row["context"] or {}),
                    recorded_at=_as_datetime(row["recorded_at"]),
                )
                for row in cursor.fetchall()
            ]


class PgSessionsRepo(_PgRepoMixin, SessionsRepo):
    """Postgres implementation of the sessions surface."""

    def __init__(
        self,
        dsn: str,
        *,
        min_pool_size: int = 1,
        max_pool_size: int = 8,
        initialize: bool = True,
    ) -> None:
        self._init_pg_repo(
            dsn,
            min_pool_size=min_pool_size,
            max_pool_size=max_pool_size,
            initialize=initialize,
        )

    def initialize_schema(self) -> None:
        self._ensure_schema()
        with self._pool.connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ilma.sessions (
                    session_id text PRIMARY KEY,
                    created_at timestamptz NOT NULL DEFAULT now(),
                    updated_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ilma.session_messages (
                    id bigserial PRIMARY KEY,
                    session_id text NOT NULL REFERENCES ilma.sessions(session_id) ON DELETE CASCADE,
                    role text NOT NULL,
                    content text NOT NULL,
                    content_tsv tsvector GENERATED ALWAYS AS
                        (to_tsvector('english', coalesce(content, ''))) STORED,
                    created_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS session_messages_session_idx ON ilma.session_messages(session_id, created_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS session_messages_content_tsv_idx ON ilma.session_messages USING gin(content_tsv)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS sessions_updated_at_idx ON ilma.sessions(updated_at DESC)"
            )

    def append(self, session_id: str, role: str, content: str) -> int:
        if not session_id.strip() or not role.strip() or not content.strip():
            raise ValueError("session_id, role, and content are required")
        with self._pool.connection() as connection:
            connection.execute(
                """
                INSERT INTO ilma.sessions (session_id)
                VALUES (%s)
                ON CONFLICT (session_id) DO UPDATE SET updated_at = now()
                """,
                (session_id,),
            )
            row = connection.execute(
                """
                INSERT INTO ilma.session_messages (session_id, role, content)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (session_id, role, content),
            ).fetchone()
        assert row is not None
        return int(row[0])

    def get_session(self, session_id: str, *, limit: int = 100) -> list[SessionMessage]:
        with (
            self._pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute(
                """
                SELECT id, session_id, role, content, created_at
                FROM ilma.session_messages
                WHERE session_id = %s
                ORDER BY created_at, id
                LIMIT %s
                """,
                (session_id, limit),
            )
            return [self._message_from_row(row) for row in cursor.fetchall()]

    def search(self, query: str, *, top_k: int = 10) -> list[SessionMessage]:
        if not query.strip() or top_k <= 0:
            return []
        with (
            self._pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute(
                """
                SELECT id, session_id, role, content, created_at,
                       ts_rank_cd(content_tsv, plainto_tsquery('english', %s)) AS score
                FROM ilma.session_messages
                WHERE content_tsv @@ plainto_tsquery('english', %s)
                ORDER BY score DESC, created_at DESC
                LIMIT %s
                """,
                (query, query, top_k),
            )
            return [self._message_from_row(row) for row in cursor.fetchall()]

    def recent_sessions(self, *, limit: int = 10) -> list[str]:
        with self._pool.connection() as connection:
            rows = connection.execute(
                "SELECT session_id FROM ilma.sessions ORDER BY updated_at DESC, session_id LIMIT %s",
                (limit,),
            ).fetchall()
        return [str(row[0]) for row in rows]

    @staticmethod
    def _message_from_row(row: Any) -> SessionMessage:
        return SessionMessage(
            id=int(row["id"]),
            session_id=str(row["session_id"]),
            role=str(row["role"]),
            content=str(row["content"]),
            created_at=row["created_at"],
        )


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

    def wiki_repo(self, *, embedders: Any | None = None) -> PgWikiRepo:
        return PgWikiRepo(
            self._dsn,
            embedders=embedders,
            min_pool_size=self._min_pool_size,
            max_pool_size=self._max_pool_size,
        )

    def journal_repo(self) -> PgJournalRepo:
        return PgJournalRepo(
            self._dsn,
            min_pool_size=self._min_pool_size,
            max_pool_size=self._max_pool_size,
        )

    def skills_repo(self) -> PgSkillsRepo:
        return PgSkillsRepo(
            self._dsn,
            min_pool_size=self._min_pool_size,
            max_pool_size=self._max_pool_size,
        )

    def metrics_repo(self) -> PgMetricsRepo:
        return PgMetricsRepo(
            self._dsn,
            min_pool_size=self._min_pool_size,
            max_pool_size=self._max_pool_size,
        )

    def kanban_repo(self) -> PgKanbanRepo:
        return PgKanbanRepo(
            self._dsn,
            min_pool_size=self._min_pool_size,
            max_pool_size=self._max_pool_size,
        )

    def observability_repo(self) -> PgObservabilityRepo:
        return PgObservabilityRepo(
            self._dsn,
            min_pool_size=self._min_pool_size,
            max_pool_size=self._max_pool_size,
        )

    def sessions_repo(self) -> PgSessionsRepo:
        return PgSessionsRepo(
            self._dsn,
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


__all__ = [
    "PgBackend",
    "PgJournalRepo",
    "PgKanbanRepo",
    "PgMemoryRepo",
    "PgMetricsRepo",
    "PgObservabilityRepo",
    "PgSessionsRepo",
    "PgSkillsRepo",
    "PgWikiRepo",
    "close_all_pools",
]
