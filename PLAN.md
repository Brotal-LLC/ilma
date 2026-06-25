# ilma Execution Plan

Framework-agnostic agent memory system — Postgres + pgvector + MCP.

---

## Naming

**ilma** — from Arabic علم (*'ilm*), meaning knowledge, science, learning.
Domain: `ilma.bd` (owned).
Repo: `github.com/Brotal-LLC/ilma`

---

## Parallel Workstreams

### P0: Fix Injection Layer (hermes-memory v2)

**Goal:** `build_memory_block` retrieves real memories instead of returning counts.

**Current (stub):**
```python
def build_memory_block(repo, *, char_limit=2200):
    s = repo.status()
    live = s.get("live_memories", 0)
    return header + f"\n(live: {live}, ...)\n"  # char_limit UNUSED
```

**Target:**
- Retrieve top-k via hybrid search
- Format as compact bullets
- Truncate to `char_limit`
- Prioritize `category="identity"` and `tags=["user","preference"]`

**Tasks:**
1. TDD: tests for empty repo, single memory, truncation, hybrid search
2. Implement `InjectionLayer` class in `ilma.core.retrieval`
3. `build_memory_block` calls `InjectionLayer.render(repo, char_limit)`
4. Integration test against real `PgMemoryRepo`
5. PR to `skb50bd/hermes-memory` (last PR before extraction)

**ETA:** 1-2 days
**Owner:** 1 dev
**Deliverable:** PR #11 on hermes-memory

---

### P1: Extract Framework-Agnostic Core

**Goal:** New package `ilma` with zero Hermes deps.

**Repo:** `github.com/Brotal-LLC/ilma`

**Structure:**
```
ilma/
├── pyproject.toml
├── src/ilma/
│   ├── core/
│   │   ├── memory.py          # MemoryRepo ABC
│   │   ├── wiki.py            # WikiRepo ABC
│   │   ├── journal.py         # JournalRepo ABC
│   │   ├── skills.py          # SkillsRepo ABC
│   │   ├── metrics.py         # MetricsRepo ABC
│   │   ├── kanban.py          # KanbanRepo ABC
│   │   ├── observability.py   # ObservabilityRepo ABC
│   │   ├── sessions.py        # SessionsRepo ABC
│   │   └── retrieval.py       # InjectionLayer, HybridSearch
│   ├── storage/
│   │   ├── base.py            # StorageBackend ABC
│   │   ├── postgres.py        # PgBackend
│   │   └── memory.py          # InMemoryBackend (tests)
│   ├── embeddings/
│   │   ├── base.py            # Embedder ABC
│   │   ├── registry.py        # EmbedderRegistry
│   │   └── providers/         # ollama.py, openai.py, kimi.py
│   ├── chunking/
│   │   └── semantic.py        # SemanticChunker
│   ├── api/
│   │   ├── mcp.py             # MCP server
│   │   ├── http.py            # FastAPI app
│   │   └── cli.py             # Typer CLI
│   └── adapters/
│       └── hermes/            # Thin Hermes wrapper
│           ├── __init__.py
│           ├── plugin.yaml
│           └── register.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── pg/
│   ├── Dockerfile             # Postgres 18 + pgvector + pg_cron + timescaledb + age
│   └── bin/                   # ilma-pg-entrypoint.sh, ilma-pg-init.sh, ilma-pg-cron.sh
├── ollama/
│   ├── Dockerfile             # ollama/ollama + bge-m3 (1024-dim) pulled on container start
│   └── bin/                   # ilma-ollama-entrypoint.sh
├── docs/
│   └── architecture.md
```

**Parallel substreams:**

| ID | Name | Tasks | Upstream |
|---|---|---|---|
| S1 | Core interfaces | `StorageBackend` ABC, all 8 `*Repo` ABCs, dataclasses | None |
| S2 | Postgres backend | Port `pg_repos.py` → `storage/postgres.py`, migrations, `chunk_text()` | S1 |
| S3 | Embeddings | Port `embeddings/`, add embedder identity enforcement | S1 |
| S4 | Retrieval + Injection | Port hybrid search, build `InjectionLayer`, query sanitizer | S2, S3 |
| S5 | All 8 surfaces | Port wiki, journal, skills, metrics, kanban, observability, sessions | S2 |
| S6 | MCP server | ~30 tools, write-ahead audit log, validation | S1-S5 |
| S7 | HTTP API | FastAPI, CRUD for all surfaces, OpenAPI | S1-S5 |
| S8 | CLI | `ilma init`, `status`, `search`, `remember`, `doctor`, `migrate`, `repair` | S1-S5 |
| S9 | Hermes adapter | `register(ctx)`, `build_memory_block` via `InjectionLayer` | S6 |

**ETA:** 1.5-2 weeks (parallel team)
**Deliverable:** `ilma` v0.1.0 on PyPI

---

### P2: MCP Server + HTTP API Hardening

**Goal:** Production-ready MCP and HTTP interfaces.

**MCP tools:**
```
# Read
ilma_status, ilma_search, ilma_recent, ilma_get_memory,
ilma_list_memories, ilma_get_wiki, ilma_search_wiki, ilma_list_wiki,
ilma_kanban_list, ilma_kanban_get, ilma_journal_search, ilma_session_search

# Write
ilma_remember, ilma_forget, ilma_wiki_create, ilma_wiki_update,
ilma_kanban_create, ilma_kanban_update, ilma_kanban_complete,
ilma_comment, ilma_record_metric, ilma_log_observation

# Maintenance
ilma_repair, ilma_migrate, ilma_doctor, ilma_export, ilma_import
```

**HTTP endpoints:**
```
GET  /health
GET  /status
POST /search
POST /remember
POST /forget
GET  /memories/{id}
GET  /wiki/{slug}
POST /wiki
GET  /wiki/search
... (CRUD for all 8 surfaces)
```

**Tasks:**
1. Wire MCP tools to `ilma.core.*`
2. Write-ahead audit log for all writes
3. Request/response validation
4. Rate limiting per scope
5. HTTP auth (API key, mTLS behind Caddy)

**ETA:** 3-4 days (after P1)
**Deliverable:** `ilma mcp` and `ilma serve` commands

---

### P3: Hermes Adapter

**Goal:** Hermes Agent talks to ilma.

**Modes:**
- **Local:** `memory.provider: ilma` — in-process, fast
- **Remote:** `memory.provider: ilma-mcp` — MCP client, network hop

**Tasks:**
1. `ilma.adapters.hermes.register(ctx)` — registers all tools
2. `build_memory_block` calls `ilma.core.retrieval.InjectionLayer`
3. Config resolution: env → `config.yaml` → default
4. Backward compat: `memory.provider: postgres` still works

**ETA:** 2-3 days
**Deliverable:** Hermes plugin in `ilma/src/adapters/hermes/`

---

### P4: Caddy Deployment + Dev-Containers

**Goal:** One-command deployment.

**Status: SHIPPED.** `ilma-pg` and `ilma-ollama` images are published to
GHCR; the README has the canonical `docker run` recipes with named
volumes, restart policies, and resource limits. The local-dev compose
file lives in `~/infra/ilma/compose.yaml` on the deploy host (NOT in
this repo — per project layout decision).

**`~/infra/ilma/compose.yaml` (canonical recipe):**
```yaml
services:
  ilma-db:
    image: ghcr.io/brotal-llc/ilma-pg:latest
    container_name: ilma-db
    restart: always
    cpus: 2
    memory: 4G
    environment:
      POSTGRES_DB: ilma
      POSTGRES_USER: ilma
      POSTGRES_PASSWORD: ${ILMA_PG_PASSWORD}
    volumes:
      - ilma-pg-data:/var/lib/postgresql/data
      - ilma-pg-init:/docker-entrypoint-initdb.d
    ports:
      - "127.0.0.1:5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U $$POSTGRES_USER -d $$POSTGRES_DB"]
      interval: 10s
      timeout: 5s
      retries: 5

  ilma-ollama:
    image: ghcr.io/brotal-llc/ilma-ollama:latest
    container_name: ilma-ollama
    restart: always
    cpus: 4
    memory: 8G
    volumes:
      - ilma-ollama-data:/root/.ollama
    ports:
      - "127.0.0.1:11434:11434"
    healthcheck:
      test: ["CMD", "bash", "-c", "(echo > /dev/tcp/localhost/11434) 2>/dev/null"]
      interval: 15s
      timeout: 5s
      retries: 10
      start_period: 180s

volumes:
  ilma-pg-data:
  ilma-pg-init:
  ilma-ollama-data:
```

**Container names are stable contracts:**
- `ilma-db` — Postgres 18 + all extensions (replaces the deprecated
  `ghcr.io/skb50bd/hermes-memory/hermes-postgres`).
- `ilma-ollama` — Ollama + bge-m3 (1024-dim embedder). Pulls bge-m3 on
  every container start; named volume caches it across restarts.
  First start takes ~80s for the model download.
- `ilma-agent` — the API/MCP service. Run however you like (systemd,
  docker, supervisor); just point `ILMA_DSN` at `ilma-db:5432`.

**Caddy (terminates TLS, reverse-proxies to `ilma-agent:8000`):**
Caddyfile lives in `~/infra/Caddyfile` (host-resident, not in this
repo). It handles cert issuance via the Cloudflare DNS plugin.
```
memory.ilma.bd {
    reverse_proxy ilma-api:8000
    tls { dns cloudflare {env.CF_API_TOKEN} }
}
```

**Dev-container:**
- Postgres + pgvector service
- Python 3.11+ with uv
- Pre-configured DSN
- `ilma` CLI in PATH
- `pytest` with Testcontainers

**Tasks:**
1. `Dockerfile` at repo root (multi-stage, ~140MB) — **DONE**
2. `pg/Dockerfile` (Postgres 18 + pgvector + pg_cron + timescaledb + age) — **DONE**, published as `ghcr.io/brotal-llc/ilma-pg`
3. ~~`infra/` compose files~~ — **REMOVED**: compose-managed local dev lives in `~/infra/` on the host, not in this repo. CI publishes images directly.
4. `.devcontainer/` config (deferred)
5. Verify non-root containers — **DONE** (`USER 1000:1000` in `Dockerfile`)
6. Domain cert via Caddy (host concern, `~/infra/Caddyfile`)

**ETA:** mostly done; remaining items deferred.

**Deliverable:** `docker compose up` in `~/infra/ilma/` (compose lives on host, not in this repo)

---

### P5: Borrow MemPalace Features

**Goal:** Adopt proven patterns from MemPalace.

| Feature | Priority | Implementation |
|---|---|---|
| Query sanitizer | **High** | Strip prompt contamination before embedding |
| Write-ahead audit log | **High** | Log all MCP writes, redact sensitive fields |
| Embedder identity enforcement | **High** | Store `embedder_model` + `dim` per memory |
| Repair tooling | Medium | `ilma repair` — index health, re-embed, vacuum |
| Knowledge graph v2 | Medium | Temporal triples on top of wiki links |
| Memory layers (L0-L3) | Low | Working vs. long-term memory abstraction |

**ETA:** Ongoing, 1-2 days per feature
**Deliverable:** Incremental PRs

---

### P6: Migration from hermes-memory v2

**Goal:** Zero-data-loss migration.

**Steps:**
1. `ilma init` — creates `ilma_` DBs
2. `ilma migrate --from hermes-memory` — copies from `hermes_default`
3. Schema mapping: `agent_memory.*` → `ilma.*`, `hermes_wiki.*` → `ilma.wiki_*`
4. Hermes config: `memory.provider: ilma`
5. Uninstall hermes-memory after verification

**Tasks:**
1. `migrate.py` with schema mapping
2. Data integrity tests (row counts, vector coverage, FTS)
3. Rollback script
4. `MIGRATION.md` documentation

**ETA:** 2-3 days
**Deliverable:** `ilma migrate --from hermes-memory --dry-run`

---

### P7: CI/CD Pipeline

**Goal:** Green CI on first PR.

**`.github/workflows/ci.yml`:**
1. `changes` — path filter
2. `lint` — ruff check + format
3. `unit-test` — pytest with InMemoryBackend
4. `integration-test` — pytest with Testcontainers (Postgres)
5. `build-postgres` — multi-arch Docker → GHCR
6. `build-api` — multi-arch Docker → GHCR
7. `smoke-postgres` — boot container, verify schemas
8. `smoke-api` — boot API, hit `/health`, verify MCP tools

**Lessons from hermes-memory:**
- No `runs-on: ubuntu-*-arm64` — buildx multi-arch
- `:latest` for service containers
- `TEMPLATE = template0` in test fixtures
- PyYAML in lint deps
- 5s stabilization sleep after init

**ETA:** 2 days
**Deliverable:** Green CI

---

## Execution Timeline

```
Week 1
├── P0: Injection fix (2d) ──────────────────────────────────────┐
├── P1-S1: Core interfaces (2d) ─────────────────────────────────┤
├── P1-S2: Postgres backend (3d) ────────────────────────────────┤
├── P1-S3: Embeddings (2d) ──────────────────────────────────────┤
└── P1-S5: Other surfaces (3d, after S2) ────────────────────────┘

Week 2
├── P1-S4: Retrieval + Injection (3d, after S2+S3) ──────────────┐
├── P1-S6: MCP server (3d, after S1-S5) ─────────────────────────┤
├── P1-S7: HTTP API (3d, after S1-S5) ───────────────────────────┤
├── P1-S8: CLI (2d, after S1-S5) ────────────────────────────────┤
├── P1-S9: Hermes adapter (2d, after S6) ────────────────────────┤
├── P4: Caddy + dev-containers (3d, after S7) ───────────────────┤
└── P7: CI pipeline (2d) ────────────────────────────────────────┘

Week 3
├── P2: MCP + HTTP hardening (2d) ───────────────────────────────┐
├── P3: Hermes adapter integration (2d) ─────────────────────────┤
├── P5: MemPalace features (ongoing) ────────────────────────────┤
├── P6: Migration path (3d) ─────────────────────────────────────┤
└── Polish, docs, release v0.1.0 ────────────────────────────────┘
```

**Critical path:** P1-S1 → S2 → S4 → S6 → P3 → release.

---

## Immediate Next Steps

1. ✅ Repo created: `github.com/Brotal-LLC/ilma`
2. ⬜ Scaffold initial structure (pyproject.toml, src/, tests/)
3. ⬜ Start P0 (injection fix on hermes-memory) OR jump to P1-S1
4. ⬜ Set up branch protection, CI skeleton

**Decision needed:** Start with P0 (hermes-memory injection fix) or go straight to P1 (ilma core scaffolding)?
