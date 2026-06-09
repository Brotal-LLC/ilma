from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

from ilma.adapters.hermes.memory_provider import IlmaMemoryProvider
from ilma.service import IlmaService


class FakeRepo:
    def status(self) -> dict[str, Any]:
        return {"ok": True}


class FakeBackend:
    def memory_repo(self) -> FakeRepo:
        return FakeRepo()

    def wiki_repo(self) -> FakeRepo:
        return FakeRepo()

    def journal_repo(self) -> FakeRepo:
        return FakeRepo()

    def skills_repo(self) -> FakeRepo:
        return FakeRepo()

    def kanban_repo(self) -> FakeRepo:
        return FakeRepo()

    def metrics_repo(self) -> FakeRepo:
        return FakeRepo()

    def observability_repo(self) -> FakeRepo:
        return FakeRepo()

    def sessions_repo(self) -> FakeRepo:
        return FakeRepo()


class FakePgBackend:
    def __init__(self, dsn: str, **kwargs: Any) -> None:
        self.dsn = dsn
        self.kwargs = kwargs

    def memory_repo(self) -> FakeRepo:
        return FakeRepo()

    def wiki_repo(self) -> FakeRepo:
        return FakeRepo()

    def journal_repo(self) -> FakeRepo:
        return FakeRepo()

    def skills_repo(self) -> FakeRepo:
        return FakeRepo()

    def kanban_repo(self) -> FakeRepo:
        return FakeRepo()

    def metrics_repo(self) -> FakeRepo:
        return FakeRepo()

    def observability_repo(self) -> FakeRepo:
        return FakeRepo()

    def sessions_repo(self) -> FakeRepo:
        return FakeRepo()


class ToolService:
    def ilma_recall(self, query: str, limit: int = 5) -> dict[str, Any]:
        """Recall relevant memories."""
        return {"ok": True, "results": [{"content": query, "limit": limit}]}

    def ilma_remember(
        self,
        content: str,
        tags: list[str] | None = None,
        category: str | None = None,
        source: str | None = "mcp",
    ) -> dict[str, Any]:
        """Remember durable content."""
        return {
            "ok": True,
            "memory_id": 1,
            "content": content,
            "tags": tags,
            "category": category,
            "source": source,
        }


def test_name_is_ilma() -> None:
    provider = IlmaMemoryProvider()
    assert provider.name == "ilma"


def test_is_available_with_no_dsn_returns_false(monkeypatch) -> None:
    monkeypatch.delenv("ILMA_DSN", raising=False)
    monkeypatch.delenv("PG_MEM_DB_CONN_STR", raising=False)
    monkeypatch.delenv("HERMES_PG_CONN_STR", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", "/tmp/nonexistent-ilma-config")

    provider = IlmaMemoryProvider()

    assert provider.is_available() is False


def test_is_available_with_dsn_returns_true(monkeypatch) -> None:
    monkeypatch.delenv("ILMA_DSN", raising=False)
    monkeypatch.delenv("PG_MEM_DB_CONN_STR", raising=False)
    monkeypatch.setenv("HERMES_PG_CONN_STR", "postgresql://example/ilma")

    provider = IlmaMemoryProvider()

    assert provider.is_available() is True


def test_initialize_builds_service(monkeypatch) -> None:
    import ilma.adapters.hermes.memory_provider as module

    monkeypatch.setenv("HERMES_PG_CONN_STR", "postgresql://example/ilma")
    monkeypatch.setattr(module, "PgBackend", FakePgBackend)

    provider = IlmaMemoryProvider()
    provider.initialize(hermes_home="/tmp/fake", platform="cli")

    assert isinstance(provider._service, IlmaService)


def test_get_tool_schemas_returns_openai_format() -> None:
    provider = IlmaMemoryProvider()
    provider._service = ToolService()

    schemas = provider.get_tool_schemas()

    assert {schema["name"] for schema in schemas} == {"ilma_recall", "ilma_remember"}
    for schema in schemas:
        assert schema["name"].startswith("ilma_")
        assert schema["description"]
        assert schema["parameters"]["type"] == "object"
        assert "properties" in schema["parameters"]
        assert "required" in schema["parameters"]


def test_handle_tool_call_ilma_recall() -> None:
    service = MagicMock()
    service.ilma_recall.return_value = {"ok": True, "results": ["memory"]}
    provider = IlmaMemoryProvider()
    provider._service = service

    result = provider.handle_tool_call("ilma_recall", {"query": "test", "limit": 3})

    data = json.loads(result)
    assert data == {"ok": True, "results": ["memory"]}
    service.ilma_recall.assert_called_once_with(query="test", limit=3)


def test_prefetch_returns_formatted_context() -> None:
    service = MagicMock()
    service.ilma_recall.return_value = {"ok": True, "results": [{"content": "mocked memory"}]}
    provider = IlmaMemoryProvider()
    provider._service = service

    result = provider.prefetch("hello")

    assert "ilma recalled memory context" in result
    assert "mocked memory" in result
    service.ilma_recall.assert_called_once_with(query="hello", limit=5)


def test_sync_turn_persists_via_ilma_remember() -> None:
    service = MagicMock()
    provider = IlmaMemoryProvider()
    provider._service = service

    provider.sync_turn("user msg", "assistant reply", session_id="abc123")

    service.ilma_remember.assert_called_once_with(
        content="User: user msg\nAssistant: assistant reply",
        category="turn",
        source="hermes:abc123",
    )


def test_sync_turn_skips_for_cron_context() -> None:
    service = MagicMock()
    provider = IlmaMemoryProvider()
    provider._service = service
    provider._agent_context = "cron"

    provider.sync_turn("user msg", "assistant reply", session_id="abc123")

    service.ilma_remember.assert_not_called()


def test_shutdown_clears_service() -> None:
    provider = IlmaMemoryProvider()
    provider._service = ToolService()

    provider.shutdown()

    assert provider._service is None
