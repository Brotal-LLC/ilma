from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from typer.testing import CliRunner

from ilma.api import cli


@dataclass
class MemoryItem:
    id: int
    content: str
    tags: tuple[str, ...] = ()
    category: str | None = None
    created_at: datetime = datetime(2026, 1, 1, tzinfo=UTC)


class FakeService:
    def __init__(self) -> None:
        self.calls: list[Any] = []
        self.memories = [MemoryItem(1, "User prefers dark mode", ("prefs",), "profile")]

    def ilma_status(self) -> dict[str, Any]:
        self.calls.append(("status", ()))
        return {
            "ok": True,
            "backend": {"ok": True, "database": "ilma_test", "pgvector": True},
            "memory": {"total_memories": 1, "live_memories": 1, "total_chunks": 1},
            "surfaces": list(cli.SURFACES),
            "tool_count": 29,
        }

    def ilma_recall(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.0,
        hybrid_text_weight: float = 0.5,
    ) -> dict[str, Any]:
        self.calls.append(("recall", (query, limit, threshold, hybrid_text_weight)))
        return {
            "ok": True,
            "results": self.memories if query else [],
            "count": len(self.memories) if query else 0,
            "query": query,
            "limit": limit,
        }

    def ilma_remember(
        self,
        content: str,
        tags: list[str] | None = None,
        category: str | None = None,
        source: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(("remember", (content, tags or [], category, source)))
        return {"ok": True, "memory_id": 42}

    def ilma_forget(self, memory_id: int) -> dict[str, Any]:
        self.calls.append(("forget", (memory_id,)))
        return {"ok": True, "deleted": memory_id == 1}

    def ilma_doctor(self) -> dict[str, Any]:
        self.calls.append(("doctor", ()))
        return {
            "ok": True,
            "healthy": True,
            "checks": {
                "backend": {"ok": True},
                "surfaces": {surface: {"ok": True} for surface in cli.SURFACES},
                "audit_log": {"ok": True},
            },
        }

    def ilma_repair(self, force: bool = False) -> dict[str, Any]:
        self.calls.append(("repair", (force,)))
        return {
            "ok": True,
            "repaired": force,
            "force": force,
            "message": "repair complete" if force else "repair dry-run complete",
            "findings": {
                "orphaned_chunks": {"count": 0},
                "duplicate_memories": {"count": 0},
                "fts_indexes": {"missing": []},
            },
        }

    def ilma_audit(
        self,
        *,
        tool: str | None = None,
        status: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        self.calls.append(("audit", (tool, status, start, end, limit, offset)))
        return {
            "ok": True,
            "results": [
                {
                    "id": 1,
                    "operation_id": "op-1",
                    "tool_name": tool or "ilma_remember",
                    "surface": "memory",
                    "action": "remember",
                    "status": status or "succeeded",
                    "payload": {"content": "hello"},
                    "result": {"memory_id": 1},
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "completed_at": "2026-01-01T00:00:01+00:00",
                    "error_type": None,
                    "error_message": None,
                }
            ],
        }

    def ilma_migrate(self) -> dict[str, Any]:
        self.calls.append(("migrate", ()))
        return {"ok": True, "migrated": True, "surfaces": 8, "audit_log": True}


class FakeConnection:
    def __init__(self, statements: list[str]) -> None:
        self.statements = statements

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def execute(self, statement: str) -> None:
        self.statements.append(statement)


class FakeBackend:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self.initialized = False

    def initialize_schema(self) -> None:
        self.initialized = True


class FakeAudit:
    def __init__(self) -> None:
        self.initialized = False

    def initialize_schema(self) -> None:
        self.initialized = True


class InitService:
    created: list[InitService] = []

    def __init__(self, backend: FakeBackend) -> None:
        self.backend = backend
        self.audit = FakeAudit()
        self.migrated = False
        InitService.created.append(self)

    def ilma_migrate(self) -> dict[str, Any]:
        self.migrated = True
        return {"ok": True, "migrated": True, "surfaces": 8, "audit_log": True}


runner = CliRunner()


def test_help_lists_required_commands() -> None:
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    for command in [
        "init",
        "status",
        "recall",
        "remember",
        "forget",
        "doctor",
        "serve",
        "mcp",
        "repair",
        "audit",
        "migrate",
        "migrate-config",
    ]:
        assert command in result.output


def test_status_and_doctor_support_json(monkeypatch: Any) -> None:
    service = FakeService()
    monkeypatch.setattr(cli, "_service_from_env", lambda: service)

    status_result = runner.invoke(cli.app, ["status", "--json"])
    assert status_result.exit_code == 0
    status_payload = json.loads(status_result.output)
    assert status_payload["backend"]["database"] == "ilma_test"
    assert service.calls[-1] == ("status", ())

    doctor_result = runner.invoke(cli.app, ["doctor", "--json"])
    assert doctor_result.exit_code == 0
    doctor_payload = json.loads(doctor_result.output)
    assert doctor_payload["healthy"] is True
    assert service.calls[-1] == ("doctor", ())


def test_memory_commands_call_service_and_render_human_output(monkeypatch: Any) -> None:
    service = FakeService()
    monkeypatch.setattr(cli, "_service_from_env", lambda: service)

    recall_result = runner.invoke(
        cli.app,
        ["recall", "dark", "--limit", "3", "--hybrid-text-weight", "0.25"],
    )
    assert recall_result.exit_code == 0
    assert "[1] User prefers dark mode" in recall_result.output
    assert service.calls[-1] == ("recall", ("dark", 3, 0.0, 0.25))

    remember_result = runner.invoke(
        cli.app,
        ["remember", "new memory", "--tag", "one", "--tags", "two", "--category", "notes"],
    )
    assert remember_result.exit_code == 0
    assert "Remembered memory 42" in remember_result.output
    assert service.calls[-1] == ("remember", ("new memory", ["one", "two"], "notes", "cli"))

    forget_result = runner.invoke(cli.app, ["forget", "1"])
    assert forget_result.exit_code == 0
    assert "Deleted memory 1" in forget_result.output
    assert service.calls[-1] == ("forget", (1,))


def test_repair_migrate_and_audit_json(monkeypatch: Any) -> None:
    service = FakeService()
    monkeypatch.setattr(cli, "_service_from_env", lambda: service)
    monkeypatch.setattr(
        cli, "_dsn_from_env", lambda: (_ for _ in ()).throw(cli.IlmaConfigError("unset"))
    )

    repair_result = runner.invoke(cli.app, ["repair", "--force", "--json"])
    assert repair_result.exit_code == 0
    assert json.loads(repair_result.output)["repaired"] is True

    audit_result = runner.invoke(
        cli.app,
        ["audit", "--tool", "ilma_remember", "--status", "succeeded", "--format", "json"],
    )
    assert audit_result.exit_code == 0
    audit_payload = json.loads(audit_result.output)
    assert audit_payload["results"][0]["tool_name"] == "ilma_remember"

    migrate_result = runner.invoke(cli.app, ["migrate", "--json"])
    assert migrate_result.exit_code == 0
    assert json.loads(migrate_result.output)["migrated"] is True
    assert service.calls[-3:] == [
        ("repair", (True,)),
        ("audit", ("ilma_remember", "succeeded", None, None, 100, 0)),
        ("migrate", ()),
    ]


def test_failed_service_result_exits_nonzero(monkeypatch: Any) -> None:
    class BrokenService(FakeService):
        def ilma_recall(
            self,
            query: str,
            limit: int = 10,
            threshold: float = 0.0,
            hybrid_text_weight: float = 0.5,
        ) -> dict[str, Any]:
            return {"ok": False, "error": {"type": "RuntimeError", "message": "boom"}}

    monkeypatch.setattr(cli, "_service_from_env", BrokenService)
    result = runner.invoke(cli.app, ["recall", "x"])
    assert result.exit_code == 1
    assert "RuntimeError: boom" in result.output


def test_init_runs_nine_steps_and_initializes_schema(monkeypatch: Any) -> None:
    statements: list[str] = []
    InitService.created.clear()
    monkeypatch.setattr(cli, "_connect_for_init", lambda dsn: FakeConnection(statements))
    monkeypatch.setattr(cli, "PgBackend", FakeBackend)
    monkeypatch.setattr(cli, "IlmaMcpService", InitService)
    monkeypatch.setattr(
        cli, "_verify_embedder", lambda: {"default_dim": 1024, "vector_length": 1024}
    )

    result = runner.invoke(cli.app, ["init", "--dsn", "postgresql://test/ilma", "--yes"])

    assert result.exit_code == 0
    assert "[9/9] Print environment hints" in result.output
    assert "ilma initialized successfully" in result.output
    assert statements == [
        "SELECT 1",
        "CREATE EXTENSION IF NOT EXISTS vector",
        "CREATE SCHEMA IF NOT EXISTS ilma",
    ]
    assert InitService.created
    assert InitService.created[0].backend.initialized is True
    assert InitService.created[0].migrated is True
    assert InitService.created[0].audit.initialized is True


def test_init_json_can_skip_embedder_check(monkeypatch: Any) -> None:
    statements: list[str] = []
    monkeypatch.setattr(cli, "_connect_for_init", lambda dsn: FakeConnection(statements))
    monkeypatch.setattr(cli, "PgBackend", FakeBackend)
    monkeypatch.setattr(cli, "IlmaMcpService", InitService)

    result = runner.invoke(
        cli.app,
        ["init", "--dsn", "postgresql://test/ilma", "--yes", "--skip-embedder-check", "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert len(payload["steps"]) == 9
    assert payload["embedder"] == {"skipped": True}


def test_serve_and_mcp_start_servers(monkeypatch: Any) -> None:
    uvicorn_calls: list[dict[str, Any]] = []

    def fake_uvicorn_run(app_ref: str, **kwargs: Any) -> None:
        uvicorn_calls.append({"app_ref": app_ref, **kwargs})

    monkeypatch.setattr("uvicorn.run", fake_uvicorn_run)
    serve_result = runner.invoke(
        cli.app, ["serve", "--host", "0.0.0.0", "--port", "9000", "--reload"]
    )
    assert serve_result.exit_code == 0
    assert uvicorn_calls == [
        {
            "app_ref": "ilma.api.http:app_factory",
            "factory": True,
            "host": "0.0.0.0",
            "port": 9000,
            "reload": True,
        }
    ]

    class FakeMcpServer:
        def __init__(self) -> None:
            self.ran = False

        def run(self) -> None:
            self.ran = True

    server = FakeMcpServer()
    monkeypatch.setattr(cli, "create_mcp_server", lambda: server)
    mcp_result = runner.invoke(cli.app, ["mcp"])
    assert mcp_result.exit_code == 0
    assert server.ran is True
