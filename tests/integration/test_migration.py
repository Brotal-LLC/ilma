"""Integration tests for hermes-memory v2 migration."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
import yaml
from testcontainers.postgres import PostgresContainer
from typer.testing import CliRunner

from ilma.api.cli import app
from ilma.storage.postgres import close_all_pools

runner = CliRunner()


@pytest.fixture(scope="session")
def migration_pg_dsn() -> Iterator[str]:
    with PostgresContainer(
        "pgvector/pgvector:pg16",
        username="test",
        password="test",
        dbname="test",
        driver=None,
    ) as postgres:
        yield postgres.get_connection_url()


@pytest.fixture
def hermes_v2_db(migration_pg_dsn: str) -> Iterator[str]:
    close_all_pools()
    with psycopg.connect(migration_pg_dsn, autocommit=True) as conn:
        for schema in (
            "ilma",
            "agent_memory",
            "hermes_wiki",
            "hermes_journal",
            "hermes_skills",
            "hermes_metrics",
            "hermes_kanban",
            "hermes_observability",
            "hermes_sessions",
        ):
            conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        _create_hermes_v2_schema(conn)
        _insert_sample_hermes_v2_data(conn)
    yield migration_pg_dsn
    close_all_pools()
    with psycopg.connect(migration_pg_dsn, autocommit=True) as conn:
        for schema in (
            "ilma",
            "agent_memory",
            "hermes_wiki",
            "hermes_journal",
            "hermes_skills",
            "hermes_metrics",
            "hermes_kanban",
            "hermes_observability",
            "hermes_sessions",
        ):
            conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")


def _create_hermes_v2_schema(conn: psycopg.Connection) -> None:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute("CREATE EXTENSION IF NOT EXISTS ltree")
    conn.execute("CREATE SCHEMA agent_memory")
    conn.execute(
        """
        CREATE TABLE agent_memory.memories (
            id bigserial PRIMARY KEY,
            content text NOT NULL,
            tags text[] DEFAULT '{}',
            category ltree,
            metadata jsonb DEFAULT '{}',
            source text,
            vector_768 vector(768),
            vector_1024 vector(1024),
            vector_1536 vector(1536),
            created_at timestamptz DEFAULT now(),
            deleted_at timestamptz
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE agent_memory.memory_chunks (
            id bigserial PRIMARY KEY,
            memory_id bigint REFERENCES agent_memory.memories(id),
            chunk_index int NOT NULL,
            content text NOT NULL,
            token_count int NOT NULL,
            vector_768 vector(768),
            vector_1024 vector(1024),
            vector_1536 vector(1536),
            created_at timestamptz DEFAULT now(),
            UNIQUE(memory_id, chunk_index)
        )
        """
    )

    conn.execute("CREATE SCHEMA hermes_wiki")
    conn.execute(
        """
        CREATE TABLE hermes_wiki.documents (
            id bigserial PRIMARY KEY,
            slug text UNIQUE NOT NULL,
            title text NOT NULL,
            body_md text NOT NULL,
            category ltree,
            metadata jsonb DEFAULT '{}',
            source_uri text,
            created_at timestamptz DEFAULT now(),
            updated_at timestamptz DEFAULT now()
        )
        """
    )
    conn.execute("CREATE TABLE hermes_wiki.tags (id serial PRIMARY KEY, name text UNIQUE NOT NULL)")
    conn.execute(
        """
        CREATE TABLE hermes_wiki.document_tags (
            document_id bigint REFERENCES hermes_wiki.documents(id),
            tag_id int REFERENCES hermes_wiki.tags(id),
            PRIMARY KEY(document_id, tag_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE hermes_wiki.document_chunks (
            id bigserial PRIMARY KEY,
            document_id bigint REFERENCES hermes_wiki.documents(id),
            ordinal int NOT NULL,
            content text NOT NULL,
            token_count int,
            vector_768 vector(768),
            vector_1024 vector(1024),
            vector_1536 vector(1536),
            created_at timestamptz DEFAULT now(),
            UNIQUE(document_id, ordinal)
        )
        """
    )

    conn.execute("CREATE SCHEMA hermes_journal")
    conn.execute(
        """
        CREATE TABLE hermes_journal.sessions (
            id bigserial PRIMARY KEY,
            profile text NOT NULL,
            started_at timestamptz DEFAULT now(),
            metadata jsonb DEFAULT '{}'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE hermes_journal.messages (
            id bigserial PRIMARY KEY,
            session_id bigint REFERENCES hermes_journal.sessions(id),
            ts timestamptz DEFAULT now(),
            role text NOT NULL,
            content text NOT NULL,
            tool_calls jsonb
        )
        """
    )

    conn.execute("CREATE SCHEMA hermes_skills")
    conn.execute(
        """
        CREATE TABLE hermes_skills.skills (
            id bigserial PRIMARY KEY,
            name text UNIQUE NOT NULL,
            version text NOT NULL,
            owner text,
            description text,
            tags text[] DEFAULT '{}',
            metadata jsonb DEFAULT '{}',
            created_at timestamptz DEFAULT now(),
            updated_at timestamptz DEFAULT now()
        )
        """
    )

    conn.execute("CREATE SCHEMA hermes_metrics")
    conn.execute(
        """
        CREATE TABLE hermes_metrics.events (
            ts timestamptz NOT NULL DEFAULT now(),
            profile text NOT NULL,
            metric_name text NOT NULL,
            value double precision NOT NULL,
            tags jsonb DEFAULT '{}'
        )
        """
    )

    conn.execute("CREATE SCHEMA hermes_kanban")
    conn.execute(
        """
        CREATE TABLE hermes_kanban.tenants (
            id bigserial PRIMARY KEY,
            slug text UNIQUE NOT NULL,
            name text NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE hermes_kanban.tasks (
            id text PRIMARY KEY,
            tenant_id bigint REFERENCES hermes_kanban.tenants(id),
            title text NOT NULL,
            body text,
            assignee text,
            status text NOT NULL,
            priority int DEFAULT 0,
            created_by text,
            created_at timestamptz DEFAULT now(),
            skills jsonb DEFAULT '[]',
            result text,
            session_id text
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE hermes_kanban.task_links (
            parent_id text REFERENCES hermes_kanban.tasks(id),
            child_id text REFERENCES hermes_kanban.tasks(id),
            PRIMARY KEY(parent_id, child_id)
        )
        """
    )

    conn.execute("CREATE SCHEMA hermes_observability")
    conn.execute(
        """
        CREATE TABLE hermes_observability.logs (
            ts timestamptz NOT NULL DEFAULT now(),
            level text NOT NULL,
            logger text NOT NULL,
            message text NOT NULL,
            exception text,
            profile text,
            session_id text,
            task_id text,
            platform text,
            metadata jsonb DEFAULT '{}'
        )
        """
    )
    conn.execute(
        "CREATE TABLE hermes_observability.traces (ts timestamptz DEFAULT now(), trace_id text, profile text, session_id text, task_id text, name text, metadata jsonb DEFAULT '{}')"
    )
    conn.execute(
        "CREATE TABLE hermes_observability.spans (ts timestamptz DEFAULT now(), trace_id text, span_id text, parent_id text, name text, start_ts timestamptz DEFAULT now(), end_ts timestamptz, duration_ms double precision, metadata jsonb DEFAULT '{}')"
    )
    conn.execute(
        "CREATE TABLE hermes_observability.llm_calls (ts timestamptz DEFAULT now(), trace_id text, span_id text, profile text, session_id text, model text, provider text, prompt_tokens int, completion_tokens int, total_tokens int, latency_ms double precision, cost_usd double precision, metadata jsonb DEFAULT '{}')"
    )
    conn.execute(
        "CREATE TABLE hermes_observability.tool_calls (ts timestamptz DEFAULT now(), trace_id text, span_id text, profile text, session_id text, tool_name text, tool_call_id text, latency_ms double precision, success boolean, error text, metadata jsonb DEFAULT '{}')"
    )

    conn.execute("CREATE SCHEMA hermes_sessions")
    conn.execute(
        """
        CREATE TABLE hermes_sessions.sessions (
            id text PRIMARY KEY,
            profile text NOT NULL,
            started_at timestamptz DEFAULT now(),
            ended_at timestamptz,
            metadata jsonb DEFAULT '{}'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE hermes_sessions.messages (
            id bigserial PRIMARY KEY,
            session_id text REFERENCES hermes_sessions.sessions(id),
            timestamp timestamptz DEFAULT now(),
            role text NOT NULL,
            content text NOT NULL,
            tool_calls jsonb,
            metadata jsonb DEFAULT '{}'
        )
        """
    )


def _insert_sample_hermes_v2_data(conn: psycopg.Connection) -> None:
    first_memory = conn.execute(
        """
        INSERT INTO agent_memory.memories (content, tags, category, metadata, source)
        VALUES ('User prefers dark mode', ARRAY['user','preference'], 'identity', '{"mood":"calm"}', 'test')
        RETURNING id
        """
    ).fetchone()[0]
    conn.execute(
        """
        INSERT INTO agent_memory.memories (content, tags, category, source)
        VALUES ('User prefers dark mode', ARRAY['duplicate'], 'identity', 'test')
        """
    )
    conn.execute(
        """
        INSERT INTO agent_memory.memory_chunks (memory_id, chunk_index, content, token_count)
        VALUES (%s, 0, 'User prefers dark mode', 4)
        """,
        (first_memory,),
    )

    doc_id = conn.execute(
        """
        INSERT INTO hermes_wiki.documents (slug, title, body_md, category, source_uri)
        VALUES ('dark-mode', 'Dark mode', 'Dark mode preference notes', 'prefs', 'file://dark.md')
        RETURNING id
        """
    ).fetchone()[0]
    tag_id = conn.execute(
        "INSERT INTO hermes_wiki.tags (name) VALUES ('prefs') RETURNING id"
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO hermes_wiki.document_tags (document_id, tag_id) VALUES (%s, %s)",
        (doc_id, tag_id),
    )
    conn.execute(
        """
        INSERT INTO hermes_wiki.document_chunks (document_id, ordinal, content, token_count)
        VALUES (%s, 0, 'Dark mode preference notes', 4)
        """,
        (doc_id,),
    )

    journal_session = conn.execute(
        "INSERT INTO hermes_journal.sessions (profile) VALUES ('default') RETURNING id"
    ).fetchone()[0]
    conn.execute(
        """
        INSERT INTO hermes_journal.messages (session_id, role, content)
        VALUES (%s, 'user', 'Remember dark mode')
        """,
        (journal_session,),
    )

    conn.execute(
        """
        INSERT INTO hermes_skills.skills (name, version, owner, description, tags)
        VALUES ('postgres-debugging', '1.0.0', 'platform', 'Debug Postgres storage', ARRAY['postgres'])
        """
    )
    conn.execute(
        """
        INSERT INTO hermes_metrics.events (profile, metric_name, value, tags)
        VALUES ('default', 'latency_ms', 12.5, '{"route":"search"}')
        """
    )

    tenant_id = conn.execute(
        "INSERT INTO hermes_kanban.tenants (slug, name) VALUES ('core', 'Core') RETURNING id"
    ).fetchone()[0]
    conn.execute(
        """
        INSERT INTO hermes_kanban.tasks (id, tenant_id, title, body, assignee, status, priority, created_by)
        VALUES ('task-parent', %s, 'Build migration', 'Move data to ilma', 'agent', 'ready', 5, 'tester')
        """,
        (tenant_id,),
    )
    conn.execute(
        """
        INSERT INTO hermes_kanban.tasks (id, tenant_id, title, body, status, priority)
        VALUES ('task-child', %s, 'Verify migration', 'Check data', 'blocked', 1)
        """,
        (tenant_id,),
    )
    conn.execute(
        "INSERT INTO hermes_kanban.task_links (parent_id, child_id) VALUES ('task-parent', 'task-child')"
    )

    conn.execute(
        """
        INSERT INTO hermes_observability.logs (level, logger, message, profile, metadata)
        VALUES ('info', 'hermes', 'migration source log', 'default', '{"ok":true}')
        """
    )
    conn.execute(
        "INSERT INTO hermes_observability.llm_calls (profile, model, total_tokens) VALUES ('default', 'test-model', 7)"
    )

    conn.execute(
        "INSERT INTO hermes_sessions.sessions (id, profile) VALUES ('session-a', 'default')"
    )
    conn.execute(
        """
        INSERT INTO hermes_sessions.messages (session_id, role, content)
        VALUES ('session-a', 'assistant', 'Stored dark mode preference')
        """
    )


def test_ilma_migrate_from_hermes_v2_schema(hermes_v2_db: str) -> None:
    dry_run = runner.invoke(app, ["migrate", "--dsn", hermes_v2_db, "--dry-run", "--json"])
    assert dry_run.exit_code == 0, dry_run.output
    dry_payload = json.loads(dry_run.output)
    assert dry_payload["detected"] is True
    assert dry_payload["dry_run"] is True
    assert dry_payload["surfaces"]["memory"]["source_rows"] == 3

    result = runner.invoke(app, ["migrate", "--dsn", hermes_v2_db, "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["detected"] is True
    assert payload["conflicts"] == 1
    assert payload["surfaces"]["memory"]["inserted"] == 2

    with psycopg.connect(hermes_v2_db) as conn:
        assert conn.execute("SELECT count(*) FROM ilma.memories").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM ilma.memory_chunks").fetchone()[0] == 1
        assert conn.execute("SELECT slug FROM ilma.wiki_docs").fetchone()[0] == "dark-mode"
        assert conn.execute("SELECT count(*) FROM ilma.wiki_chunks").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM ilma.journal_entries").fetchone()[0] == 1
        assert (
            conn.execute(
                "SELECT content FROM ilma.skills WHERE name = 'postgres-debugging'"
            ).fetchone()[0]
            == "Debug Postgres storage"
        )
        assert (
            conn.execute(
                "SELECT labels ->> 'profile' FROM ilma.metrics WHERE name = 'latency_ms'"
            ).fetchone()[0]
            == "default"
        )
        child_parent = conn.execute(
            """
            SELECT child.parent_id IS NOT NULL
            FROM ilma.kanban_tasks child
            WHERE child.metadata ->> 'hermes_v2_id' = 'task-child'
            """
        ).fetchone()[0]
        assert child_parent is True
        assert conn.execute("SELECT count(*) FROM ilma.observations").fetchone()[0] >= 2
        assert (
            conn.execute(
                "SELECT count(*) FROM ilma.session_messages WHERE session_id = 'session-a'"
            ).fetchone()[0]
            == 1
        )


def test_migrate_config_updates_provider_dsn_and_writes_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "memory": {"provider": "postgres"},
                "env": {"PG_MEM_DB_CONN_STR": "postgresql://old/db"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("ILMA_DSN", raising=False)
    monkeypatch.delenv("HERMES_PG_CONN_STR", raising=False)
    monkeypatch.delenv("PG_MEM_DB_CONN_STR", raising=False)

    result = runner.invoke(app, ["migrate-config", "--config", str(config_path), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["changed"] is True
    assert payload["backup_path"]
    assert Path(payload["backup_path"]).exists()

    updated = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert updated["memory"]["provider"] == "ilma"
    assert updated["env"]["ILMA_DSN"] == "postgresql://old/db"
    assert updated["env"]["PG_MEM_DB_CONN_STR"] == "postgresql://old/db"
