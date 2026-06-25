# ilma

**Framework-agnostic agent memory system.**

Postgres + pgvector backend. MCP-native. Hermes Agent, Claude, Cursor, Codex — any MCP client.

```bash
pip install ilma-agent
ilma init
ilma status
```

---

## What ilma is

ilma stores what your agents know — and makes it retrievable across sessions, frameworks, and machines.

- **8 memory surfaces**: memories, wiki, journal, skills, metrics, kanban, observability, sessions
- **Hybrid retrieval**: vector + FTS + chunk-level reranking
- **Graph layer (Apache AGE)**: cross-entity traversal over Memory / Wiki / Skill vertices with SHARES_TAG, CO_OCCURS, REFERENCES_WIKI, and USES_SKILL edges. Rebuild on demand; expand recall hits via `expand_graph=True` on `ilma_recall`.
- **MCP server**: `ilma-mcp` — works with any MCP client
- **HTTP API**: REST endpoints behind your own reverse proxy
- **CLI**: `ilma init`, `ilma search`, `ilma remember`, `ilma graph rebuild`, `ilma doctor`
- **Postgres + pgvector**: proven, backup-friendly, multi-client

## Quick start

```bash
# Install
pip install ilma-agent

# Start the Postgres image (ships with all extensions)
docker run -d --name ilma-pg \
    -e POSTGRES_DB=ilma \
    -e POSTGRES_USER=ilma \
    -e POSTGRES_PASSWORD=change-me \
    -p 5432:5432 \
    ghcr.io/brotal-llc/ilma-pg:latest

# Initialize (9-step wizard: Postgres, extensions, schemas, embedder)
ilma init

# Store a memory
ilma remember "User prefers dark mode" --tags user,preference

# Search
ilma search "dark mode preference"

# Check health
ilma doctor
```

## Architecture

```
MCP client (Claude, Cursor, Hermes, etc.)
    ↓  stdio / HTTP
ilma MCP server / HTTP API
    ↓  psycopg3
Postgres + pgvector (+ pg_cron + timescaledb + apache-age)
    ├─ ilma.memories
    ├─ ilma.wiki
    ├─ ilma.journal
    ├─ ilma.skills
    ├─ ilma.metrics
    ├─ ilma.kanban
    ├─ ilma.observability
    ├─ ilma.sessions
    └─ ag_catalog.ilma_graph (derived view — Memory / Wiki / Skill vertices + edges)
```

The `ilma_graph` lives in Apache AGE. It's a derived view over the relational
state in `ilma.*` and is rebuilt on demand via `ilma graph rebuild`. See
`src/ilma/core/graph.py` and the `ilma-age-graph` skill for design notes.

## Docker images

| Image | Purpose | Tag |
|-------|---------|-----|
| `ghcr.io/brotal-llc/ilma` | ilma CLI + HTTP API | `latest`, `main`, `:{sha}` |
| `ghcr.io/brotal-llc/ilma-pg` | Postgres 18 + all required extensions | `latest`, `main`, `:{sha}` |

The ilma-pg image ships with: pgvector, pg_trgm, ltree, timescaledb,
pg_cron, and apache-age. The ilma Python package's `initialize_schema()`
creates the `ilma` schema and tables on first API connection — no SQL
migration step needed.

The legacy `ghcr.io/skb50bd/hermes-memory/hermes-postgres` image is
deprecated. New deployments should use `ghcr.io/brotal-llc/ilma-pg`.

## Project layout

```
.
├── src/ilma/                  # Python package
│   ├── core/                  # pure logic (memory, wiki, retrieval, graph)
│   ├── storage/               # PostgreSQL backend (pgvector, AGE)
│   ├── api/                   # CLI + MCP + HTTP API
│   ├── adapters/              # hermes-memory provider shim
│   └── plugins/               # hermes-agent plugin entry points
├── pg/                        # Postgres image (Dockerfile + entrypoint scripts)
│   ├── Dockerfile             # pgvector + pg_cron + timescaledb + age
│   └── bin/                   # ilma-pg-entrypoint.sh, ilma-pg-init.sh, ilma-pg-cron.sh
├── tests/                     # unit + integration
│   ├── unit/                  # 226 tests, no DB
│   └── integration/           # 12 tests via Testcontainers (ilma-pg:latest)
├── Dockerfile                 # ilma CLI image (the API service)
├── pyproject.toml
└── .github/workflows/ilma.yml # single gated CI: lint, test, security, tag,
                              # build (ilma CLI), build-pg (Postgres), release
```

> **No `docker/` or `infra/` directories.** Local-dev Docker Compose and
> Caddy configs live in `~/infra/` on the deployment host, not in this
> repo. The CI workflow publishes ready-to-pull images.

## MCP tools (31)

The ilma MCP server exposes 31 tools, including the graph surface:

- `ilma_recall(query, ..., expand_graph=False, graph_hops=1)` — graph-aware recall.
- `ilma_wiki_search(query, ..., expand_graph=False, graph_hops=1)` — graph-aware wiki search.
- `ilma_graph_rebuild(min_shared_tags=2)` — drop and rebuild the AGE graph.
- `ilma_traverse(kind, src_id, max_hops=2, edge_types=None, limit=50)` — bounded BFS.
- ...plus 27 others (memory, wiki, journal, skills, metrics, observability, sessions, kanban, audit, doctor, migrate, repair).

## Development

Install dev dependencies and set up pre-commit:

```bash
make install
pip install pre-commit
pre-commit install
```

Run all checks locally:

```bash
make all
```

Run only the integration tests against the ilma-pg image:

```bash
uv run pytest tests/integration/ -v --tb=short
```

(Requires Docker — Testcontainers spins up `ghcr.io/brotal-llc/ilma-pg:latest`
automatically.)

## Project status

In active development. See [PLAN.md](PLAN.md) for the execution roadmap.

## License

MIT
