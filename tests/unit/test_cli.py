from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest
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
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append(("recall", (query, limit, threshold, hybrid_text_weight, kwargs)))
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

    def ilma_list_memories(
        self,
        limit: int = 50,
        offset: int = 0,
        include_deleted: bool = False,
    ) -> dict[str, Any]:
        self.calls.append(("list", (limit, offset, include_deleted)))
        # Mirror the real service: order by created_at DESC, id DESC.
        ordered = sorted(
            self.memories,
            key=lambda m: (m.created_at, m.id),
            reverse=True,
        )
        rows = [
            {
                "id": m.id,
                "content": m.content,
                "tags": list(m.tags),
                "category": m.category,
                "source": "test",
                "metadata": {},
                "deleted": False,
                "created_at": m.created_at,
            }
            for m in ordered
        ]
        return {"ok": True, "results": rows, "count": len(rows)}

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

    def ilma_migrate(self, reembed: bool = False) -> dict[str, Any]:
        self.calls.append(("migrate", (reembed,)))
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
    assert service.calls[-1] == (
        "recall",
        ("dark", 3, 0.0, 0.25, {"expand_graph": False, "graph_hops": 1}),
    )

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


def test_list_memories_renders_human_output_and_honors_filters(
    monkeypatch: Any,
) -> None:
    service = FakeService()
    service.memories = [
        MemoryItem(
            id=1,
            content="User prefers dark mode",
            tags=("prefs", "ui"),
            category="profile",
            created_at=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
        ),
        MemoryItem(
            id=2,
            content="Rezaur = Bruce Wayne roleplay\nAlfred persona when addressed as Master Wayne",
            tags=("identity",),
            category="user-identity",
            created_at=datetime(2026, 6, 2, 9, 30, tzinfo=UTC),
        ),
        MemoryItem(
            id=3,
            content="Soft-deleted test memory",
            tags=(),
            category=None,
            created_at=datetime(2026, 5, 15, 8, 0, tzinfo=UTC),
        ),
    ]
    # Mark id=3 as soft-deleted by extending the FakeService to return it with
    # deleted=True. FakeService.ilma_list_memories always returns deleted=False,
    # so for the --all branch we override.
    monkeypatch.setattr(cli, "_service_from_env", lambda: service)

    # Default: live memories only (id=3 is filtered at the service layer, but
    # FakeService returns all rows unchanged. The CLI itself doesn't filter, it
    # forwards include_deleted to the service. Verify forwarding here.)
    result = runner.invoke(cli.app, ["list", "--limit", "2"])
    assert result.exit_code == 0
    assert "Memories: showing" in result.output
    assert service.calls[-1] == ("list", (2, 0, False))
    # Newest first: id=2 should appear before id=1
    assert result.output.index("[2]") < result.output.index("[1]")

    # --all forwards include_deleted=True
    result_all = runner.invoke(cli.app, ["list", "--all", "--offset", "1", "--limit", "10"])
    assert result_all.exit_code == 0
    assert service.calls[-1] == ("list", (10, 1, True))

    # --json works
    result_json = runner.invoke(cli.app, ["list", "--json"])
    assert result_json.exit_code == 0
    payload = json.loads(result_json.output)
    assert payload["ok"] is True
    assert len(payload["results"]) == 3

    # --csv works and is mutually exclusive with --json
    result_csv = runner.invoke(cli.app, ["list", "--csv"])
    assert result_csv.exit_code == 0
    assert "id,created_at,deleted,category,tags,content" in result_csv.output

    result_both = runner.invoke(cli.app, ["list", "--csv", "--json"])
    assert result_both.exit_code == 2

    # Empty result
    empty_service = FakeService()
    empty_service.memories = []
    monkeypatch.setattr(cli, "_service_from_env", lambda: empty_service)
    empty_result = runner.invoke(cli.app, ["list"])
    assert empty_result.exit_code == 0
    assert "No memories found." in empty_result.output


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
        ("migrate", (False,)),
    ]

    migrate_reembed_result = runner.invoke(cli.app, ["migrate", "--reembed", "--json"])
    assert migrate_reembed_result.exit_code == 0
    payload = json.loads(migrate_reembed_result.output)
    assert payload["migrated"] is True
    assert service.calls[-1] == ("migrate", (True,))

    class BrokenService(FakeService):
        def ilma_recall(
            self,
            query: str,
            limit: int = 10,
            threshold: float = 0.0,
            hybrid_text_weight: float = 0.5,
            **kwargs: Any,
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


# ---------------------------------------------------------------------------
# Graph CLI command tests
# ---------------------------------------------------------------------------


def test_graph_command_rebuild_invokes_service_and_renders_human_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ilma graph rebuild` calls ilma_graph_rebuild and prints stats."""
    from typer.testing import CliRunner

    class GraphService:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        def ilma_graph_rebuild(self, *, min_shared_tags: int) -> dict[str, Any]:
            self.calls.append(("rebuild", min_shared_tags))
            return {
                "ok": True,
                "stats": {
                    "memory_vertices": 43,
                    "wiki_vertices": 5,
                    "skill_vertices": 1,
                    "shares_tag_edges": 232,
                    "co_occurs_edges": 0,
                    "references_wiki_edges": 6,
                    "uses_skill_edges": 0,
                },
            }

        def ilma_traverse(self, **kwargs: Any) -> dict[str, Any]:
            raise AssertionError("traverse should not be called for rebuild")

    svc = GraphService()
    monkeypatch.setattr(cli, "_service_from_env", lambda: svc)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["graph", "rebuild", "--min-shared-tags", "3"])
    assert result.exit_code == 0
    assert svc.calls == [("rebuild", 3)]
    assert "Graph rebuilt:" in result.output
    assert "memory_vertices: 43" in result.output
    assert "shares_tag_edges: 232" in result.output


def test_graph_command_rebuild_json_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typer.testing import CliRunner

    class GraphService:
        def ilma_graph_rebuild(self, *, min_shared_tags: int) -> dict[str, Any]:
            return {
                "ok": True,
                "stats": {
                    "memory_vertices": 10,
                    "wiki_vertices": 1,
                    "skill_vertices": 0,
                    "shares_tag_edges": 0,
                    "co_occurs_edges": 0,
                    "references_wiki_edges": 0,
                    "uses_skill_edges": 0,
                },
            }

        def ilma_traverse(self, **kwargs: Any) -> dict[str, Any]:
            raise AssertionError

    monkeypatch.setattr(cli, "_service_from_env", lambda: GraphService())
    runner = CliRunner()
    result = runner.invoke(cli.app, ["graph", "rebuild", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["stats"]["memory_vertices"] == 10


def test_graph_command_traverse_requires_src_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typer.testing import CliRunner

    class GraphService:
        def ilma_traverse(self, **kwargs: Any) -> dict[str, Any]:
            raise AssertionError("must not be called without --src-id")

        def ilma_graph_rebuild(self, **kwargs: Any) -> dict[str, Any]:
            raise AssertionError

    monkeypatch.setattr(cli, "_service_from_env", lambda: GraphService())
    runner = CliRunner()
    result = runner.invoke(cli.app, ["graph", "traverse"])
    assert result.exit_code == 2
    assert "--src-id is required" in result.output


def test_graph_command_traverse_invokes_service_and_renders_human_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typer.testing import CliRunner

    class GraphService:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        def ilma_traverse(
            self,
            *,
            kind: str,
            src_id: int,
            max_hops: int,
            edge_types: list[str] | None,
            limit: int,
        ) -> dict[str, Any]:
            self.calls.append(
                {
                    "kind": kind,
                    "src_id": src_id,
                    "max_hops": max_hops,
                    "edge_types": edge_types,
                    "limit": limit,
                }
            )
            return {
                "ok": True,
                "subgraph": {
                    "nodes": [
                        {
                            "kind": "Memory",
                            "src_id": 99,
                            "vertex_id": 1,
                            "properties": {"id": 99, "category": "fact"},
                        }
                    ],
                    "edges": [
                        {
                            "edge_id": 7,
                            "label": "SHARES_TAG",
                            "start_id": 1,
                            "end_id": 99,
                            "properties": {"tags": ["a", "b"]},
                        }
                    ],
                },
            }

        def ilma_graph_rebuild(self, **kwargs: Any) -> dict[str, Any]:
            raise AssertionError

    svc = GraphService()
    monkeypatch.setattr(cli, "_service_from_env", lambda: svc)
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "graph",
            "traverse",
            "--kind",
            "Memory",
            "--src-id",
            "1",
            "--max-hops",
            "2",
            "--edge-type",
            "SHARES_TAG",
            "--edge-type",
            "REFERENCES_WIKI",
            "--limit",
            "10",
        ],
    )
    assert result.exit_code == 0
    assert svc.calls == [
        {
            "kind": "Memory",
            "src_id": 1,
            "max_hops": 2,
            "edge_types": ["SHARES_TAG", "REFERENCES_WIKI"],
            "limit": 10,
        }
    ]
    assert "1 node(s), 1 edge(s)" in result.output
    assert "node: kind=Memory src_id=99" in result.output
    assert "edge: SHARES_TAG 1->99" in result.output


def test_graph_command_traverse_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typer.testing import CliRunner

    class GraphService:
        def ilma_traverse(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "subgraph": {"nodes": [], "edges": []},
            }

        def ilma_graph_rebuild(self, **kwargs: Any) -> dict[str, Any]:
            raise AssertionError

    monkeypatch.setattr(cli, "_service_from_env", lambda: GraphService())
    runner = CliRunner()
    result = runner.invoke(cli.app, ["graph", "traverse", "--src-id", "42", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["subgraph"]["nodes"] == []


def test_graph_command_unknown_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typer.testing import CliRunner

    class GraphService:
        def ilma_graph_rebuild(self, **kwargs: Any) -> dict[str, Any]:
            raise AssertionError

        def ilma_traverse(self, **kwargs: Any) -> dict[str, Any]:
            raise AssertionError

    monkeypatch.setattr(cli, "_service_from_env", lambda: GraphService())
    runner = CliRunner()
    result = runner.invoke(cli.app, ["graph", "delete"])
    assert result.exit_code == 2
    assert "Unknown graph action" in result.output
