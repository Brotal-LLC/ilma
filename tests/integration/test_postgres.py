"""Integration tests for the Postgres memory backend.

These tests exercise a real Postgres + pgvector instance via Testcontainers.
They are intentionally written against the framework-agnostic ilma interfaces
and must not depend on hermes-memory.
"""

from __future__ import annotations

from collections.abc import Iterator

import psycopg
import pytest
from testcontainers.postgres import PostgresContainer

from ilma.core.memory import MemoryRepo


class FakeEmbedderRegistry:
    """Small deterministic 1024-dim embedder for integration tests."""

    default_dim = 1024

    def embed(self, text: str, *, dim: int | None = None) -> list[float]:
        assert dim in (None, 1024)
        lower = text.lower()
        vec = [0.0] * 1024
        groups = [
            (("dark",), 0),
            (("mode",), 1),
            (("preference", "prefers"), 2),
            (("postgres", "pgvector"), 3),
            (("atlas", "cartography"), 4),
            (("python",), 5),
        ]
        for words, idx in groups:
            if any(word in lower for word in words):
                vec[idx] = 1.0
        # Keep every vector non-zero for cosine distance.
        vec[-1] = 0.001
        return vec


@pytest.fixture(scope="session")
def pg_dsn() -> Iterator[str]:
    with PostgresContainer(
        "pgvector/pgvector:pg16",
        username="test",
        password="test",
        dbname="test",
        driver=None,
    ) as postgres:
        yield postgres.get_connection_url()


@pytest.fixture
def clean_pg(pg_dsn: str) -> Iterator[str]:
    from ilma.storage.postgres import close_all_pools

    close_all_pools()
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        conn.execute("DROP SCHEMA IF EXISTS ilma CASCADE")
    yield pg_dsn
    close_all_pools()
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        conn.execute("DROP SCHEMA IF EXISTS ilma CASCADE")


def make_repo(dsn: str) -> MemoryRepo:
    from ilma.storage.postgres import PgMemoryRepo

    return PgMemoryRepo(dsn, embedders=FakeEmbedderRegistry(), min_pool_size=1, max_pool_size=4)


def test_pg_memory_repo_remember_search_forget_status(clean_pg: str) -> None:
    repo = make_repo(clean_pg)

    memory_id = repo.remember(
        "User prefers dark mode in every application",
        tags=["user", "preference"],
        category="identity",
        source="integration-test",
    )

    assert memory_id > 0
    assert (
        repo.remember(
            "User prefers dark mode in every application",
            tags=["duplicate-ignored"],
            category="identity",
            source="integration-test",
        )
        == 0
    )

    results = repo.search("dark mode preference", top_k=3)
    assert [m.id for m in results] == [memory_id]
    assert results[0].content == "User prefers dark mode in every application"
    assert results[0].tags == ("user", "preference")
    assert results[0].category == "identity"
    assert results[0].source == "integration-test"
    assert not results[0].deleted
    assert results[0].created_at is not None

    status = repo.status()
    assert status["total_memories"] == 1
    assert status["live_memories"] == 1
    assert status["total_chunks"] == 1
    assert status["default_dim"] == 1024

    assert repo.forget(memory_id) is True
    assert repo.search("dark mode", top_k=3) == []
    assert repo.status()["live_memories"] == 0


def test_pg_memory_repo_chunked_embeddings_enable_chunk_vector_search(clean_pg: str) -> None:
    repo = make_repo(clean_pg)
    long_content = (
        "This long memory starts with unrelated filler. "
        + ("x" * 2300)
        + " The buried concept is cartography and map making. "
        + ("y" * 2300)
    )

    memory_id = repo.remember(long_content, tags=["project"], category="research")

    with psycopg.connect(clean_pg) as conn:
        chunk_count = conn.execute(
            "SELECT count(*) FROM ilma.memory_chunks WHERE memory_id = %s",
            (memory_id,),
        ).fetchone()[0]
        parent_has_vector = conn.execute(
            "SELECT vector_1024 IS NOT NULL FROM ilma.memories WHERE id = %s",
            (memory_id,),
        ).fetchone()[0]

    assert chunk_count > 1
    # Long memories rely on chunk vectors; the parent vector remains empty.
    assert parent_has_vector is False

    # Query text does not occur in the memory, but the fake embedder maps
    # atlas <-> cartography to the same vector dimension. This proves the
    # chunk-level vector path participates in hybrid search.
    results = repo.search("atlas", top_k=3, hybrid_text_weight=0.0)
    assert [m.id for m in results] == [memory_id]


def test_pg_memory_repo_hybrid_search_uses_full_text_and_vector(clean_pg: str) -> None:
    repo = make_repo(clean_pg)
    postgres_id = repo.remember("Postgres pgvector stores durable memories", tags=["db"])
    python_id = repo.remember("Python testing notes for agents", tags=["code"])

    fts_results = repo.search("durable memories", top_k=2, hybrid_text_weight=1.0)
    assert fts_results[0].id == postgres_id

    vector_results = repo.search("python", top_k=2, hybrid_text_weight=0.0)
    assert vector_results[0].id == python_id


def test_pg_memory_repo_uses_shared_connection_pool(clean_pg: str) -> None:
    from ilma.storage import postgres as pg

    repo1 = make_repo(clean_pg)
    repo2 = make_repo(clean_pg)

    assert repo1.status()["live_memories"] == 0
    assert repo2.status()["live_memories"] == 0
    assert pg._get_pool(clean_pg, min_size=1, max_size=4) is pg._get_pool(  # noqa: SLF001
        clean_pg, min_size=1, max_size=4
    )


def test_postgres_backend_is_framework_agnostic(clean_pg: str) -> None:
    repo = make_repo(clean_pg)
    repo.remember("Framework agnostic Postgres backend", tags=["ilma"])

    import sys

    assert "hermes_memory" not in sys.modules
