from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from ilma.api.http import SURFACES, create_app
from ilma.api.mcp import set_service


class FakeHttpService:
    def __init__(self) -> None:
        self.memories = [
            {"id": 1, "content": "User prefers dark mode", "tags": [], "category": None}
        ]
        self.wiki_docs = {"intro": {"id": 1, "slug": "intro", "title": "Intro", "body_md": "Hello"}}
        self.observability = self
        self.access_logs: list[dict[str, Any]] = []

    def ilma_status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "backend": {"ok": True, "database": "fake", "pgvector": True},
            "memory": {"total_memories": len(self.memories)},
            "surfaces": SURFACES,
            "tool_count": 29,
        }

    def ilma_recall(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.0,
        hybrid_text_weight: float = 0.5,
    ) -> dict[str, Any]:
        results = [m for m in self.memories if query.lower() in m["content"].lower()][:limit]
        return {
            "ok": True,
            "results": results,
            "count": len(results),
            "query": query,
            "limit": limit,
        }

    def ilma_remember(
        self,
        content: str,
        tags: list[str] | None = None,
        category: str | None = None,
        source: str | None = "http",
    ) -> dict[str, Any]:
        if content == "boom":
            return {
                "ok": False,
                "error": {"type": "ValueError", "message": "cannot remember boom"},
            }
        memory_id = len(self.memories) + 1
        self.memories.append(
            {"id": memory_id, "content": content, "tags": tags or [], "category": category}
        )
        return {"ok": True, "memory_id": memory_id}

    def ilma_forget(self, memory_id: int) -> dict[str, Any]:
        return {"ok": True, "deleted": memory_id == 1}

    def ilma_get_memory(self, memory_id: int) -> dict[str, Any]:
        return {"ok": True, "memory": next(m for m in self.memories if m["id"] == memory_id)}

    def ilma_list_memories(
        self, limit: int = 50, offset: int = 0, include_deleted: bool = False
    ) -> dict[str, Any]:
        return {"ok": True, "results": self.memories[offset : offset + limit]}

    def ilma_wiki_search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        return {
            "ok": True,
            "results": [
                doc for doc in self.wiki_docs.values() if query.lower() in doc["title"].lower()
            ][:top_k],
        }

    def ilma_get_wiki(self, slug: str) -> dict[str, Any]:
        return {"ok": True, "document": self.wiki_docs.get(slug)}

    def ilma_list_wiki(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        docs = list(self.wiki_docs.values())
        return {"ok": True, "results": docs[offset : offset + limit]}

    def ilma_wiki_create(
        self,
        slug: str,
        title: str,
        body_md: str,
        category: str | None = None,
        tags: list[str] | None = None,
        source_uri: str | None = None,
    ) -> dict[str, Any]:
        self.wiki_docs[slug] = {
            "id": len(self.wiki_docs) + 1,
            "slug": slug,
            "title": title,
            "body_md": body_md,
        }
        return {"ok": True, "document_id": self.wiki_docs[slug]["id"], "version_id": 1, "chunks": 1}

    def ilma_wiki_update(
        self,
        slug: str,
        title: str,
        body_md: str,
        category: str | None = None,
        tags: list[str] | None = None,
        source_uri: str | None = None,
    ) -> dict[str, Any]:
        self.wiki_docs[slug] = {
            "id": self.wiki_docs.get(slug, {}).get("id", 2),
            "slug": slug,
            "title": title,
            "body_md": body_md,
        }
        return {"ok": True, "document_id": self.wiki_docs[slug]["id"], "version_id": 1, "chunks": 1}

    def ilma_journal_search(self, query: str, top_k: int = 10) -> dict[str, Any]:
        return {"ok": True, "results": [{"id": 1, "content": f"journal {query}"}]}

    def ilma_journal_recent(self, limit: int = 10) -> dict[str, Any]:
        return {"ok": True, "results": [{"id": 1, "content": "journal recent"}]}

    def ilma_skills_search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        return {"ok": True, "results": [{"id": 1, "name": "python", "content": query}]}

    def ilma_skills_get(self, name: str) -> dict[str, Any]:
        return {"ok": True, "skill": {"id": 1, "name": name, "content": "skill"}}

    def ilma_kanban_list(self, status: str = "todo", limit: int = 50) -> dict[str, Any]:
        return {"ok": True, "results": [{"id": 7, "status": status}]}

    def ilma_kanban_get(self, task_id: int) -> dict[str, Any]:
        return {"ok": True, "task": {"id": task_id, "title": "task"}}

    def ilma_kanban_create(
        self,
        title: str,
        description: str = "",
        status: str = "todo",
        priority: int = 0,
        tags: list[str] | None = None,
        parent_id: int | None = None,
    ) -> dict[str, Any]:
        return {"ok": True, "task_id": 7}

    def ilma_kanban_update(
        self,
        task_id: int,
        title: str | None = None,
        description: str | None = None,
        status: str | None = None,
        priority: int | None = None,
        tags: list[str] | None = None,
        parent_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {"ok": True, "updated": True}

    def ilma_kanban_complete(self, task_id: int) -> dict[str, Any]:
        return {"ok": True, "completed": True}

    def ilma_metrics_record(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> dict[str, Any]:
        return {"ok": True, "metric_id": 9}

    def ilma_metrics_query(
        self,
        name: str,
        start: str | None = None,
        end: str | None = None,
        limit: int = 100,
        aggregate_window: str | None = None,
    ) -> dict[str, Any]:
        return {
            "ok": True,
            "results": [{"id": 9, "name": name}],
            "aggregate": bool(aggregate_window),
        }

    def ilma_obs_log(
        self,
        level: str,
        message: str,
        source: str | None = "http",
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {"ok": True, "observation_id": 11}

    def log(
        self,
        level: str,
        message: str,
        *,
        source: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> int:
        self.access_logs.append(
            {"level": level, "message": message, "source": source, "context": context or {}}
        )
        return len(self.access_logs)

    def ilma_obs_query(
        self,
        level: str | None = None,
        source: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        return {"ok": True, "results": [{"id": 11, "level": level or "info"}]}

    def ilma_session_search(self, query: str, top_k: int = 10) -> dict[str, Any]:
        return {"ok": True, "results": [{"id": 13, "session_id": "s1", "content": query}]}

    def ilma_session_get(self, session_id: str, limit: int = 100) -> dict[str, Any]:
        return {"ok": True, "messages": [{"id": 13, "session_id": session_id, "content": "hello"}]}

    def ilma_repair(self) -> dict[str, Any]:
        return {"ok": True, "repaired": True}

    def ilma_doctor(self) -> dict[str, Any]:
        return {"ok": True, "healthy": True, "checks": {}}

    def ilma_migrate(self) -> dict[str, Any]:
        return {"ok": True, "migrated": True, "surfaces": 8, "audit_log": True}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.delenv("ILMA_API_KEY", raising=False)
    monkeypatch.delenv("ILMA_RATE_LIMIT_RPS", raising=False)
    app = create_app(cast(Any, FakeHttpService()))
    with TestClient(app) as test_client:
        yield test_client
    set_service(None)


def assert_ok(response) -> dict:
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ok"] is True
    return payload


def test_health_status_and_openapi(client: TestClient) -> None:
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["ok"] is True
    assert health.json()["backend"]["database"] == "fake"

    status = assert_ok(client.get("/status"))
    assert status["surfaces"] == SURFACES
    assert status["tool_count"] == 29

    spec = client.get("/openapi.json")
    assert spec.status_code == 200
    paths = spec.json()["paths"]
    for path in [
        "/health",
        "/status",
        "/recall",
        "/remember",
        "/forget",
        "/memories/{memory_id}",
        "/memories",
        "/wiki/search",
        "/wiki/{slug}",
        "/wiki",
        "/journal/search",
        "/journal/recent",
        "/skills/search",
        "/skills/{name}",
        "/kanban",
        "/kanban/{task_id}",
        "/kanban/{task_id}/complete",
        "/metrics",
        "/metrics/query",
        "/observations",
        "/observations/query",
        "/sessions/search",
        "/sessions/{session_id}",
        "/repair",
        "/doctor",
        "/migrate",
    ]:
        assert path in paths


def test_memory_routes(client: TestClient) -> None:
    remembered = assert_ok(
        client.post(
            "/remember",
            json={
                "content": "HTTP memory",
                "tags": ["api"],
                "category": "tests",
            },
        )
    )
    assert remembered["memory_id"] == 2

    recall = assert_ok(client.post("/recall", json={"query": "dark", "limit": 3}))
    assert recall["results"][0]["content"] == "User prefers dark mode"

    memory = assert_ok(client.get("/memories/1"))
    assert memory["memory"]["content"] == "User prefers dark mode"

    memories = assert_ok(client.get("/memories", params={"limit": 10, "offset": 0}))
    assert len(memories["results"]) >= 1

    forgotten = assert_ok(client.post("/forget", json={"memory_id": 1}))
    assert forgotten["deleted"] is True


def test_wiki_journal_and_skills_routes(client: TestClient) -> None:
    wiki_search = assert_ok(client.post("/wiki/search", json={"query": "Intro"}))
    assert wiki_search["results"][0]["slug"] == "intro"

    wiki_doc = assert_ok(client.get("/wiki/intro"))
    assert wiki_doc["document"]["title"] == "Intro"

    wiki_list = assert_ok(client.get("/wiki"))
    assert wiki_list["results"][0]["slug"] == "intro"

    created_wiki = assert_ok(
        client.post(
            "/wiki",
            json={"slug": "api", "title": "API", "body_md": "HTTP docs"},
        )
    )
    assert created_wiki["document_id"] == 2

    updated_wiki = assert_ok(
        client.patch(
            "/wiki/api",
            json={"slug": "ignored", "title": "API v2", "body_md": "Updated"},
        )
    )
    assert updated_wiki["version_id"] == 1

    journal_search = assert_ok(client.post("/journal/search", json={"query": "day"}))
    assert journal_search["results"][0]["content"] == "journal day"

    journal_recent = assert_ok(client.get("/journal/recent", params={"limit": 5}))
    assert journal_recent["results"][0]["content"] == "journal recent"

    skills_search = assert_ok(client.post("/skills/search", json={"query": "py"}))
    assert skills_search["results"][0]["name"] == "python"

    skill = assert_ok(client.get("/skills/python"))
    assert skill["skill"]["name"] == "python"


def test_kanban_metrics_observations_sessions_and_maintenance_routes(
    client: TestClient,
) -> None:
    kanban_list = assert_ok(client.get("/kanban", params={"status": "todo"}))
    assert kanban_list["results"][0]["status"] == "todo"

    kanban_task = assert_ok(client.get("/kanban/7"))
    assert kanban_task["task"]["id"] == 7

    kanban_created = assert_ok(client.post("/kanban", json={"title": "ship http"}))
    assert kanban_created["task_id"] == 7

    kanban_updated = assert_ok(client.patch("/kanban/7", json={"status": "doing"}))
    assert kanban_updated["updated"] is True

    kanban_completed = assert_ok(client.post("/kanban/7/complete"))
    assert kanban_completed["completed"] is True

    metric = assert_ok(
        client.post("/metrics", json={"name": "latency", "value": 12.5, "labels": {"route": "/"}})
    )
    assert metric["metric_id"] == 9

    metric_query = assert_ok(client.post("/metrics/query", json={"name": "latency"}))
    assert metric_query["aggregate"] is False

    metric_aggregate = assert_ok(
        client.post(
            "/metrics/query",
            json={"name": "latency", "aggregate_window": "1 hour"},
        )
    )
    assert metric_aggregate["aggregate"] is True

    observation = assert_ok(
        client.post(
            "/observations",
            json={"level": "info", "message": "hello", "context": {"test": True}},
        )
    )
    assert observation["observation_id"] == 11

    observations = assert_ok(client.post("/observations/query", json={"level": "info"}))
    assert observations["results"][0]["level"] == "info"

    sessions = assert_ok(client.post("/sessions/search", json={"query": "hello"}))
    assert sessions["results"][0]["session_id"] == "s1"

    session = assert_ok(client.get("/sessions/s1"))
    assert session["messages"][0]["session_id"] == "s1"

    assert_ok(client.post("/repair"))
    doctor = assert_ok(client.post("/doctor"))
    assert doctor["healthy"] is True
    migrate = assert_ok(client.post("/migrate"))
    assert migrate["surfaces"] == 8


def test_http_returns_structured_service_errors(client: TestClient) -> None:
    response = client.post("/remember", json={"content": "boom"})
    assert response.status_code == 200
    assert response.json() == {
        "ok": False,
        "error": {"type": "ValueError", "message": "cannot remember boom"},
    }


def test_api_key_auth_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ILMA_API_KEY", "secret")
    monkeypatch.delenv("ILMA_RATE_LIMIT_RPS", raising=False)
    app = create_app(cast(Any, FakeHttpService()))
    with TestClient(app) as test_client:
        assert test_client.get("/health").status_code == 200
        assert test_client.get("/openapi.json").status_code == 200
        assert test_client.get("/status").status_code == 401
        assert test_client.get("/status", headers={"X-API-Key": "wrong"}).status_code == 401
        assert test_client.get("/status", headers={"X-API-Key": "secret"}).status_code == 200
    set_service(None)


def test_rate_limit_and_health_exemption(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ILMA_API_KEY", raising=False)
    monkeypatch.setenv("ILMA_RATE_LIMIT_RPS", "1")
    app = create_app(cast(Any, FakeHttpService()))
    with TestClient(app) as test_client:
        assert test_client.get("/status").status_code == 200
        assert test_client.get("/status").status_code == 429
        assert test_client.get("/health").status_code == 200
        assert test_client.get("/health").status_code == 200
    set_service(None)


def test_metrics_cors_and_access_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ILMA_API_KEY", raising=False)
    monkeypatch.delenv("ILMA_RATE_LIMIT_RPS", raising=False)
    monkeypatch.setenv("ILMA_CORS_ORIGINS", "https://app.example")
    service = FakeHttpService()
    app = create_app(cast(Any, service))
    with TestClient(app) as test_client:
        status = test_client.get("/status", headers={"Origin": "https://app.example"})
        assert status.status_code == 200
        assert status.headers["access-control-allow-origin"] == "https://app.example"

        metrics = test_client.get("/metrics")
        assert metrics.status_code == 200
        body = metrics.text
        assert "request_count" in body
        assert "request_duration" in body
        assert "tool_call_count" in body
        assert "tool_call_duration" in body
        assert "memory_search_latency" in body
        assert "db_connection_pool_size" in body

    assert any(record["source"] == "http.access" for record in service.access_logs)
    assert service.access_logs[-1]["context"]["client_ip"]
    set_service(None)
