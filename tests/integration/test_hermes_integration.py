"""End-to-end integration tests for the Hermes Agent adapter.

These tests simulate a Hermes plugin context while exercising a real
Postgres + pgvector backend through Testcontainers. They intentionally avoid
importing hermes-memory so the adapter remains framework-agnostic.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest
from testcontainers.postgres import PostgresContainer

from ilma.adapters.hermes.register import PROVIDER_ILMA, PROVIDER_LOCAL, PROVIDER_POSTGRES, register


class FakeEmbedderRegistry:
    """Small deterministic 1024-dim embedder for adapter integration tests."""

    default_dim = 1024

    def embed(self, text: str, *, dim: int | None = None) -> list[float]:
        assert dim in (None, 1024)
        lower = text.lower()
        vec = [0.0] * 1024
        groups = [
            (("dark",), 0),
            (("mode",), 1),
            (("preference", "preferences", "prefers"), 2),
            (("user", "identity"), 3),
            (("project", "context"), 4),
        ]
        for words, idx in groups:
            if any(word in lower for word in words):
                vec[idx] = 1.0
        vec[-1] = 0.001  # Keep every vector non-zero for cosine distance.
        return vec


class FakeCtx:
    """Minimal Hermes Agent plugin context used by the adapter."""

    def __init__(self) -> None:
        self.tools: dict[str, dict[str, Any]] = {}
        self.hooks: dict[str, list[Any]] = {}

    def register_tool(self, name: str, **kwargs: Any) -> None:
        self.tools[name] = {"name": name, **kwargs}

    def register_hook(self, name: str, fn: Any) -> None:
        self.hooks.setdefault(name, []).append(fn)


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


def _write_hermes_config(hermes_home: Path, provider: str) -> None:
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "config.yaml").write_text(f"memory:\n  provider: {provider}\n", encoding="utf-8")


def _configure_adapter(
    monkeypatch: pytest.MonkeyPatch,
    *,
    hermes_home: Path,
    provider: str,
    dsn: str,
) -> None:
    _write_hermes_config(hermes_home, provider)
    monkeypatch.delenv("MEMORY_PROVIDER", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("ILMA_DSN", dsn)
    monkeypatch.setenv("ILMA_PG_POOL_MIN", "1")
    monkeypatch.setenv("ILMA_PG_POOL_MAX", "4")

    # The E2E target is the Hermes adapter + real Postgres. Patch only the
    # embedder so the test does not depend on an external Ollama/OpenAI service.
    import ilma.storage.postgres as pg

    monkeypatch.setattr(
        pg.EmbedderRegistry,
        "from_env",
        classmethod(lambda cls: FakeEmbedderRegistry()),
    )


def test_hermes_adapter_memory_crud_hooks_and_postgres_persistence(
    clean_pg: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_adapter(
        monkeypatch,
        hermes_home=tmp_path / "hermes-ilma",
        provider=PROVIDER_ILMA,
        dsn=clean_pg,
    )
    ctx = FakeCtx()

    register(ctx)

    assert "hermes_memory" not in sys.modules
    assert "memory" in ctx.tools
    assert ctx.tools["memory"]["override"] is True
    assert "pre_tool_call" in ctx.hooks
    assert "on_session_end" in ctx.hooks
    assert "ilma_remember" in ctx.tools
    assert "ilma_recall" in ctx.tools

    memory = ctx.tools["memory"]["handler"]
    add = json.loads(
        memory(
            action="add",
            content="User prefers dark mode in every Hermes application",
            tags=["user", "preference"],
            category="identity",
            source="hermes-adapter-integration",
        )
    )
    assert add["ok"] is True
    memory_id = add["memory_id"]
    assert memory_id > 0

    with psycopg.connect(clean_pg) as conn:
        row = conn.execute(
            """
            SELECT content, tags, category, source, deleted_at IS NULL AS live
            FROM ilma.memories
            WHERE id = %s
            """,
            (memory_id,),
        ).fetchone()
    assert row == (
        "User prefers dark mode in every Hermes application",
        ["user", "preference"],
        "identity",
        "hermes-adapter-integration",
        True,
    )

    search = json.loads(memory(action="search", query="dark mode preference", top_k=3))
    assert search["ok"] is True
    assert [result["id"] for result in search["results"]] == [memory_id]
    assert search["results"][0]["content"] == "User prefers dark mode in every Hermes application"

    listed = json.loads(memory(action="list", limit=10))
    assert listed["ok"] is True
    assert [result["id"] for result in listed["results"]] == [memory_id]

    pre_tool_call = ctx.hooks["pre_tool_call"][0]
    for tool_name in ("system_prompt_refresh", "memory"):
        hook_result = pre_tool_call(tool_name)
        assert hook_result is not None
        block = hook_result["memory_block"]
        assert "MEMORY (your personal notes)" in block
        assert "User prefers dark mode" in block
        assert "live: 1" in block
    assert pre_tool_call("unrelated_tool") is None

    removed = json.loads(memory(action="remove", memory_id=memory_id))
    assert removed["ok"] is True
    assert removed["deleted"] is True

    with psycopg.connect(clean_pg) as conn:
        deleted_row = conn.execute(
            "SELECT deleted_at FROM ilma.memories WHERE id = %s",
            (memory_id,),
        ).fetchone()
    assert deleted_row is not None
    assert deleted_row[0] is not None
    assert json.loads(memory(action="search", query="dark mode", top_k=3))["results"] == []


def test_hermes_adapter_provider_noop_routing(
    clean_pg: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for provider in (PROVIDER_POSTGRES, PROVIDER_LOCAL):
        _configure_adapter(
            monkeypatch,
            hermes_home=tmp_path / f"hermes-{provider}",
            provider=provider,
            dsn=clean_pg,
        )
        ctx = FakeCtx()

        register(ctx)

        assert ctx.tools == {}
        assert ctx.hooks == {}
