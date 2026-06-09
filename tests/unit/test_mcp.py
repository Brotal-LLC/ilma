from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from mcp.types import TextContent

from ilma.api.mcp import (
    TOOL_COUNT,
    WRITE_TOOLS,
    IlmaConfigError,
    IlmaMcpService,
    InMemoryAuditLogger,
    _dsn_from_env,
    create_mcp_server,
)


@dataclass
class Item:
    id: int
    name: str = "item"
    content: str = "content"
    created_at: datetime = datetime(2026, 1, 1, tzinfo=UTC)


class FakeMemoryRepo:
    def __init__(self) -> None:
        self.items = [Item(1, content="User prefers dark mode")]
        self.initialized = False

    def initialize_schema(self) -> None:
        self.initialized = True

    def remember(self, content: str, **kwargs: Any) -> int:
        if content == "boom":
            raise ValueError("cannot remember boom")
        item_id = len(self.items) + 1
        self.items.append(Item(item_id, content=content))
        return item_id

    def search(self, query: str, **kwargs: Any) -> list[Item]:
        return [item for item in self.items if query.lower() in item.content.lower()]

    def forget(self, memory_id: int) -> bool:
        return memory_id == 1

    def status(self) -> dict[str, Any]:
        return {
            "total_memories": len(self.items),
            "live_memories": len(self.items),
            "total_chunks": 0,
        }

    def recent(self, *, limit: int = 10) -> list[Item]:
        return self.items[-limit:]

    def get(self, memory_id: int) -> Item | None:
        return next((item for item in self.items if item.id == memory_id), None)

    def list(self, **kwargs: Any) -> list[Item]:
        return self.items


class FakeWikiRepo:
    def __init__(self) -> None:
        self.docs = {"intro": {"id": 1, "slug": "intro", "title": "Intro", "body_md": "Hello"}}
        self.initialized = False

    def initialize_schema(self) -> None:
        self.initialized = True

    def ingest(self, slug: str, title: str, body_md: str, **kwargs: Any) -> dict[str, Any]:
        self.docs[slug] = {
            "id": len(self.docs) + 1,
            "slug": slug,
            "title": title,
            "body_md": body_md,
        }
        return {"document_id": self.docs[slug]["id"], "version_id": 1, "chunks": 1}

    def get(self, slug: str) -> dict[str, Any] | None:
        return self.docs.get(slug)

    def search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        return [doc for doc in self.docs.values() if query.lower() in doc["title"].lower()]

    def list(self, **kwargs: Any) -> list[dict[str, Any]]:
        return list(self.docs.values())


class FakeJournalRepo:
    def __init__(self) -> None:
        self.initialized = False

    def initialize_schema(self) -> None:
        self.initialized = True

    def search(self, query: str, **kwargs: Any) -> list[Item]:
        return [Item(1, content=f"journal {query}")]

    def recent(self, **kwargs: Any) -> list[Item]:
        return [Item(1, content="journal recent")]


class FakeSkillsRepo:
    def __init__(self) -> None:
        self.initialized = False

    def initialize_schema(self) -> None:
        self.initialized = True

    def search(self, query: str, **kwargs: Any) -> list[Item]:
        return [Item(1, name="python", content=query)]

    def get(self, name: str) -> Item | None:
        return Item(1, name=name, content="skill")


class FakeKanbanRepo:
    def __init__(self) -> None:
        self.initialized = False

    def initialize_schema(self) -> None:
        self.initialized = True

    def create(self, title: str, **kwargs: Any) -> int:
        return 7

    def get(self, task_id: int) -> dict[str, Any]:
        return {"id": task_id, "title": "task"}

    def update(self, task_id: int, **kwargs: Any) -> bool:
        return bool(kwargs)

    def complete(self, task_id: int) -> bool:
        return True

    def list_by_status(self, status: str, **kwargs: Any) -> list[dict[str, Any]]:
        return [{"id": 7, "status": status}]


class FakeMetricsRepo:
    def __init__(self) -> None:
        self.initialized = False

    def initialize_schema(self) -> None:
        self.initialized = True

    def record(self, name: str, value: float, **kwargs: Any) -> int:
        return 9

    def query(self, name: str, **kwargs: Any) -> list[dict[str, Any]]:
        return [{"id": 9, "name": name, "value": 1.0, "recorded_at": datetime.now(UTC)}]

    def aggregate(self, name: str, **kwargs: Any) -> list[dict[str, Any]]:
        return [{"name": name, "count": 1}]


class FakeObservabilityRepo:
    def __init__(self) -> None:
        self.initialized = False
        self.logged: list[dict[str, Any]] = []

    def initialize_schema(self) -> None:
        self.initialized = True

    def log(self, level: str, message: str, **kwargs: Any) -> int:
        self.logged.append({"level": level, "message": message, **kwargs})
        return len(self.logged)

    def query(self, **kwargs: Any) -> list[dict[str, Any]]:
        return [{"id": 11, "level": kwargs.get("level") or "info"}]


class FakeSessionsRepo:
    def __init__(self) -> None:
        self.initialized = False

    def initialize_schema(self) -> None:
        self.initialized = True

    def search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        return [{"id": 13, "session_id": "s1", "content": query}]

    def get_session(self, session_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        return [{"id": 13, "session_id": session_id, "content": "hello"}]


class FakeBackend:
    def __init__(self) -> None:
        self.memory = FakeMemoryRepo()
        self.wiki = FakeWikiRepo()
        self.journal = FakeJournalRepo()
        self.skills = FakeSkillsRepo()
        self.kanban = FakeKanbanRepo()
        self.metrics = FakeMetricsRepo()
        self.observability = FakeObservabilityRepo()
        self.sessions = FakeSessionsRepo()
        self.initialized = False

    def initialize_schema(self) -> None:
        self.initialized = True

    def health(self) -> dict[str, Any]:
        return {"ok": True, "database": "fake", "pgvector": True}

    def memory_repo(self) -> FakeMemoryRepo:
        return self.memory

    def wiki_repo(self) -> FakeWikiRepo:
        return self.wiki

    def journal_repo(self) -> FakeJournalRepo:
        return self.journal

    def skills_repo(self) -> FakeSkillsRepo:
        return self.skills

    def kanban_repo(self) -> FakeKanbanRepo:
        return self.kanban

    def metrics_repo(self) -> FakeMetricsRepo:
        return self.metrics

    def observability_repo(self) -> FakeObservabilityRepo:
        return self.observability

    def sessions_repo(self) -> FakeSessionsRepo:
        return self.sessions


@pytest.fixture
def service() -> IlmaMcpService:
    return IlmaMcpService(FakeBackend(), audit=InMemoryAuditLogger())


def test_dsn_from_env_prefers_ilma_and_supports_migration_fallback() -> None:
    preferred = {"ILMA_DSN": "postgresql://new", "PG_MEM_DB_CONN_STR": "postgresql://legacy"}
    fallback = {"PG_MEM_DB_CONN_STR": "postgresql://legacy"}
    assert _dsn_from_env(preferred) == "postgresql://new"
    assert _dsn_from_env(fallback) == "postgresql://legacy"
    with pytest.raises(IlmaConfigError):
        _dsn_from_env({})


@pytest.mark.asyncio
async def test_mcp_server_registration_is_driven_by_tools_dict_loop(
    service: IlmaMcpService,
) -> None:
    source = Path("src/ilma/api/mcp.py").read_text()
    assert "@server.tool" not in source

    server = create_mcp_server(service)
    tools = {tool.name: tool for tool in await server.list_tools()}

    assert set(tools) == set(WRITE_TOOLS) | {
        "ilma_status",
        "ilma_recall",
        "ilma_recent",
        "ilma_get_memory",
        "ilma_list_memories",
        "ilma_get_wiki",
        "ilma_wiki_search",
        "ilma_list_wiki",
        "ilma_journal_search",
        "ilma_journal_recent",
        "ilma_skills_search",
        "ilma_skills_get",
        "ilma_kanban_list",
        "ilma_kanban_get",
        "ilma_metrics_query",
        "ilma_obs_query",
        "ilma_session_search",
        "ilma_session_get",
        "ilma_doctor",
    }
    assert tools["ilma_recall"].inputSchema["required"] == ["query"]
    assert "hybrid_text_weight" in tools["ilma_recall"].inputSchema["properties"]

    import json
    result = await server.call_tool("ilma_recall", {"query": "dark"})
    # FastMCP >=1.0 returns a list of TextContent blocks; the JSON
    # payload is in result[0].text. Cast to TextContent for type-checkers.
    structured = json.loads(cast(TextContent, result[0]).text)
    assert isinstance(structured, dict)
    assert structured["ok"] is True
    assert structured["results"][0]["content"] == "User prefers dark mode"
@pytest.mark.asyncio
async def test_mcp_server_write_tool_audits_once(service: IlmaMcpService) -> None:
    server = create_mcp_server(service)

    import json
    result = await server.call_tool("ilma_remember", {"content": "from mcp"})
    structured = json.loads(cast(TextContent, result[0]).text)

    assert isinstance(structured, dict)
    assert structured["ok"] is True
    audit_logger = service.audit
    assert isinstance(audit_logger, InMemoryAuditLogger)
    assert [record["tool_name"] for record in audit_logger.records] == ["ilma_remember"]


@pytest.mark.asyncio
async def test_mcp_server_registers_expected_29_tools(service: IlmaMcpService) -> None:
    server = create_mcp_server(service)
    tools = await server.list_tools()
    names = {tool.name for tool in tools}
    assert len(names) == TOOL_COUNT
    assert names == {
        "ilma_status",
        "ilma_recall",
        "ilma_recent",
        "ilma_get_memory",
        "ilma_list_memories",
        "ilma_remember",
        "ilma_forget",
        "ilma_get_wiki",
        "ilma_wiki_search",
        "ilma_list_wiki",
        "ilma_wiki_create",
        "ilma_wiki_update",
        "ilma_journal_search",
        "ilma_journal_recent",
        "ilma_skills_search",
        "ilma_skills_get",
        "ilma_kanban_list",
        "ilma_kanban_get",
        "ilma_kanban_create",
        "ilma_kanban_update",
        "ilma_kanban_complete",
        "ilma_metrics_record",
        "ilma_metrics_query",
        "ilma_obs_log",
        "ilma_obs_query",
        "ilma_session_search",
        "ilma_session_get",
        "ilma_repair",
        "ilma_doctor",
        "ilma_migrate",
    }


def test_read_tools_return_structured_success(service: IlmaMcpService) -> None:
    assert service.ilma_status()["ok"] is True
    assert service.ilma_recall("dark")["ok"] is True
    assert service.ilma_recent()["ok"] is True
    assert service.ilma_get_memory(1)["memory"]["content"] == "User prefers dark mode"
    assert service.ilma_list_memories()["ok"] is True
    assert service.ilma_get_wiki("intro")["document"]["title"] == "Intro"
    assert service.ilma_wiki_search("Intro")["ok"] is True
    assert service.ilma_list_wiki()["ok"] is True
    assert service.ilma_journal_search("x")["ok"] is True
    assert service.ilma_journal_recent()["ok"] is True
    assert service.ilma_skills_search("py")["ok"] is True
    assert service.ilma_skills_get("python")["skill"]["name"] == "python"
    assert service.ilma_kanban_list("todo")["ok"] is True
    assert service.ilma_kanban_get(7)["task"]["id"] == 7
    assert service.ilma_metrics_query("latency")["aggregate"] is False
    assert service.ilma_metrics_query("latency", aggregate_window="1 hour")["aggregate"] is True
    assert service.ilma_obs_query(level="info")["ok"] is True
    assert service.ilma_session_search("hello")["ok"] is True
    assert service.ilma_session_get("s1")["messages"][0]["session_id"] == "s1"


def test_write_tools_are_audited_before_success(service: IlmaMcpService) -> None:
    assert set(WRITE_TOOLS) == {
        "ilma_remember",
        "ilma_forget",
        "ilma_wiki_create",
        "ilma_wiki_update",
        "ilma_kanban_create",
        "ilma_kanban_update",
        "ilma_kanban_complete",
        "ilma_metrics_record",
        "ilma_obs_log",
        "ilma_migrate",
        "ilma_repair",
    }
    calls = [
        service.ilma_remember("new memory"),
        service.ilma_forget(1),
        service.ilma_wiki_create("new", "New", "Body"),
        service.ilma_wiki_update("new", "New 2", "Body 2"),
        service.ilma_kanban_create("task"),
        service.ilma_kanban_update(7, status="done"),
        service.ilma_kanban_complete(7),
        service.ilma_metrics_record("latency", 1.2),
        service.ilma_obs_log("info", "message"),
        service.ilma_migrate(),
        service.ilma_repair(),
    ]
    assert all(call["ok"] for call in calls)
    audit_logger = service.audit
    assert isinstance(audit_logger, InMemoryAuditLogger)
    assert [record["tool_name"] for record in audit_logger.records] == list(WRITE_TOOLS)
    assert all(record["status"] == "succeeded" for record in audit_logger.records)


def test_errors_are_structured_and_failed_writes_are_audited(service: IlmaMcpService) -> None:
    result = service.ilma_remember("boom")
    assert result == {
        "ok": False,
        "error": {"type": "ValueError", "message": "cannot remember boom"},
    }
    audit_logger = service.audit
    assert isinstance(audit_logger, InMemoryAuditLogger)
    assert audit_logger.records[-1]["tool_name"] == "ilma_remember"
    assert audit_logger.records[-1]["status"] == "failed"
    assert audit_logger.records[-1]["error_type"] == "ValueError"


def test_tool_calls_are_structured_logged(service: IlmaMcpService) -> None:
    result = service.ilma_recall("dark")
    assert result["ok"] is True
    assert service.backend.observability.logged
    record = service.backend.observability.logged[-1]
    assert record["source"] == "mcp.tool"
    assert record["context"]["tool_name"] == "ilma_recall"
    assert record["context"]["success"] is True


def test_maintenance_initializes_all_surfaces(service: IlmaMcpService) -> None:
    result = service.ilma_migrate()
    assert result["ok"] is True
    backend = service.backend
    assert backend.initialized is True
    assert backend.memory.initialized is True
    assert backend.wiki.initialized is True
    assert backend.journal.initialized is True
    assert backend.skills.initialized is True
    assert backend.kanban.initialized is True
    assert backend.metrics.initialized is True
    assert backend.observability.initialized is True
    assert backend.sessions.initialized is True
