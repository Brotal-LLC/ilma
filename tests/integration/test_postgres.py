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


def test_pg_wiki_repo_ingest_get_hybrid_search_and_suggest_links(clean_pg: str) -> None:
    from ilma.storage.postgres import PgWikiRepo

    repo = PgWikiRepo(clean_pg, embedders=FakeEmbedderRegistry(), min_pool_size=1, max_pool_size=4)
    first = repo.ingest(
        "postgres-memory",
        "Postgres memory",
        "Durable pgvector knowledge storage for agents.",
        category="db",
        tags=["postgres", "pgvector"],
        source_uri="https://example.test/postgres",
    )
    second = repo.ingest(
        "python-testing",
        "Python testing",
        "Testing notes for reliable Python agents.",
        category="code",
        tags=["python"],
    )

    assert first["document_id"] > 0
    assert first["version_id"] == 1
    assert first["chunks"] >= 1
    assert second["document_id"] > 0

    doc = repo.get("postgres-memory")
    assert doc is not None
    assert doc.slug == "postgres-memory"
    assert doc.tags == ("postgres", "pgvector")
    assert doc.source_uri == "https://example.test/postgres"

    updated = repo.ingest("postgres-memory", "Postgres memory", "Updated pgvector docs")
    assert updated["document_id"] == first["document_id"]
    assert updated["version_id"] == 2

    fts_results = repo.search("pgvector", top_k=2)
    assert fts_results[0]["slug"] == "postgres-memory"

    vector_results = repo.search("python", top_k=2)
    assert vector_results[0]["slug"] == "python-testing"

    suggestions = repo.suggest_links(first["document_id"], top_k=5)
    assert {s["slug"] for s in suggestions} >= {"python-testing"}


def test_pg_journal_repo_append_search_recent(clean_pg: str) -> None:
    from ilma.storage.postgres import PgJournalRepo

    repo = PgJournalRepo(clean_pg, min_pool_size=1, max_pool_size=4)
    first_id = repo.append("Today I debugged a Postgres integration", tags=["db"])
    second_id = repo.append("Wrote Python unit tests", tags=["code"])

    assert first_id > 0
    assert second_id > first_id
    assert repo.search("Postgres", top_k=3)[0].id == first_id
    recent = repo.recent(limit=2)
    assert [entry.id for entry in recent] == [second_id, first_id]
    assert recent[0].created_at is not None


def test_pg_skills_repo_upsert_get_search(clean_pg: str) -> None:
    from ilma.storage.postgres import PgSkillsRepo

    repo = PgSkillsRepo(clean_pg, min_pool_size=1, max_pool_size=4)
    skill_id = repo.upsert(
        "postgres-debugging",
        "Use psql and pgvector diagnostics to debug storage issues.",
        category="database",
        tags=["postgres"],
    )
    assert skill_id > 0
    assert repo.upsert("postgres-debugging", "Updated pgvector troubleshooting guide") == skill_id

    skill = repo.get("postgres-debugging")
    assert skill is not None
    assert skill.id == skill_id
    assert skill.content == "Updated pgvector troubleshooting guide"
    assert repo.search("troubleshooting", top_k=3)[0].name == "postgres-debugging"


def test_pg_metrics_repo_record_query_aggregate(clean_pg: str) -> None:
    from ilma.storage.postgres import PgMetricsRepo

    repo = PgMetricsRepo(clean_pg, min_pool_size=1, max_pool_size=4)
    first_id = repo.record("latency_ms", 10.0, labels={"route": "search"})
    second_id = repo.record("latency_ms", 30.0, labels={"route": "search"})
    repo.record("tokens", 100.0)

    rows = repo.query("latency_ms", limit=10)
    assert {row.id for row in rows} == {first_id, second_id}
    assert rows[0].labels == {"route": "search"}
    assert rows[0].recorded_at is not None

    aggregates = repo.aggregate("latency_ms", window="1 hour")
    assert len(aggregates) == 1
    assert aggregates[0]["count"] == 2
    assert aggregates[0]["avg"] == 20.0
    assert aggregates[0]["min"] == 10.0
    assert aggregates[0]["max"] == 30.0


def test_pg_kanban_repo_create_get_update_complete_list_search(clean_pg: str) -> None:
    from ilma.storage.postgres import PgKanbanRepo

    repo = PgKanbanRepo(clean_pg, min_pool_size=1, max_pool_size=4)
    parent_id = repo.create(
        "Build Postgres backend",
        description="Implement durable kanban storage",
        priority=5,
        tags=["db"],
    )
    child_id = repo.create(
        "Write tests", description="Integration tests for kanban", parent_id=parent_id, priority=10
    )

    child = repo.get(child_id)
    assert child is not None
    assert child.parent_id == parent_id
    assert child.status == "todo"

    assert repo.update(child_id, status="in_progress", metadata={"owner": "agent"}) is True
    assert repo.get(child_id).metadata == {"owner": "agent"}  # type: ignore[union-attr]
    assert [task.id for task in repo.list_by_status("in_progress")] == [child_id]
    assert repo.search("durable kanban", top_k=3)[0].id == parent_id
    assert repo.complete(child_id) is True
    assert repo.get(child_id).status == "done"  # type: ignore[union-attr]


def test_pg_observability_repo_log_and_query(clean_pg: str) -> None:
    from ilma.storage.postgres import PgObservabilityRepo

    repo = PgObservabilityRepo(clean_pg, min_pool_size=1, max_pool_size=4)
    info_id = repo.log("info", "started postgres integration", source="tests", context={"ok": True})
    error_id = repo.log("error", "failed tool call", source="tools")

    errors = repo.query(level="error")
    assert [obs.id for obs in errors] == [error_id]
    tests = repo.query(source="tests")
    assert [obs.id for obs in tests] == [info_id]
    assert tests[0].context == {"ok": True}
    assert tests[0].recorded_at is not None


def test_pg_sessions_repo_append_fetch_search_recent(clean_pg: str) -> None:
    from ilma.storage.postgres import PgSessionsRepo

    repo = PgSessionsRepo(clean_pg, min_pool_size=1, max_pool_size=4)
    first_id = repo.append("session-a", "user", "Please debug Postgres storage")
    second_id = repo.append("session-a", "assistant", "I will inspect pgvector tables")
    repo.append("session-b", "user", "Unrelated Python question")

    messages = repo.get_session("session-a")
    assert [m.id for m in messages] == [first_id, second_id]
    assert messages[0].session_id == "session-a"
    assert messages[0].created_at is not None

    assert repo.search("pgvector", top_k=3)[0].id == second_id
    assert repo.recent_sessions(limit=2)[0] == "session-b"
