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

# Start the ilma-db container (Postgres 18 + pgvector + pg_cron + timescaledb + age)
# See the "Production deployment" section below for the full recipe with
# named volumes, restart policy, and resource limits.
docker run -d --name ilma-db --restart always \
    --cpus=2 --memory=4g \
    -e POSTGRES_DB=ilma \
    -e POSTGRES_USER=ilma \
    -e POSTGRES_PASSWORD=change-me \
    -v ilma-pg-data:/var/lib/postgresql/data \
    -v ilma-pg-init:/docker-entrypoint-initdb.d \
    -p 127.0.0.1:5432:5432 \
    ghcr.io/brotal-llc/ilma-pg:latest

# (Optional) Start an embedder — Ollama with bge-m3 (1024-dim) pre-pulled
docker run -d --name ilma-ollama --restart always \
    --cpus=4 --memory=8g \
    -v ilma-ollama-data:/root/.ollama \
    -p 127.0.0.1:11434:11434 \
    ghcr.io/brotal-llc/ilma-ollama:latest
# (the image's entrypoint pulls `bge-m3` on first start if missing)

# Initialize (9-step wizard: Postgres, extensions, schemas, embedder)
ilma init

# Store a memory
ilma remember "User prefers dark mode" --tags user,preference

# Search
ilma search "dark mode preference"

# Check health
ilma doctor
```

## Production deployment

The canonical recipe: two containers (`ilma-db` for Postgres, `ilma-ollama`
for embeddings) plus the `ilma-agent` Python package running as the API/MCP
service. All three names are stable so systemd / docker-compose / k8s can
reference them by name.

### ilma-db (Postgres 18 + extensions)

```bash
docker run -d --name ilma-db --restart always \
    --cpus=2 --memory=4g \
    --health-cmd="pg_isready -U $$POSTGRES_USER -d $$POSTGRES_DB" \
    --health-interval=10s --health-timeout=5s --health-retries=5 \
    -e POSTGRES_DB=ilma \
    -e POSTGRES_USER=ilma \
    -e POSTGRES_PASSWORD=change-me \
    -v ilma-pg-data:/var/lib/postgresql/data \
    -v ilma-pg-init:/docker-entrypoint-initdb.d \
    -p 127.0.0.1:5432:5432 \
    ghcr.io/brotal-llc/ilma-pg:latest
```

- **`--name ilma-db`** — stable name. The ilma API and any tooling reference it by name.
- **`--restart always`** — survives host reboots, Docker daemon restarts.
- **`--cpus=2 --memory=4g`** — Postgres 18 + 5 extensions needs at least 1 CPU and 2GB
  to be comfortable; bump these up for large wiki/journal tables.
- **Named volumes** — `ilma-pg-data` holds the database cluster (survives
  container removal); `ilma-pg-init` exposes `/docker-entrypoint-initdb.d`
  so you can drop custom `.sh` / `.sql` files into it without rebuilding the
  image.
- **`-p 127.0.0.1:5432:5432`** — bind to localhost only. The ilma API runs on the
  same host and connects via `host.docker.internal` or the unix socket. If
  you need cross-host access, terminate Postgres at a firewall or use a
  WireGuard tunnel rather than exposing 5432 publicly.

The image ships with pgvector, pg_trgm, ltree, timescaledb, pg_cron, and
apache-age preinstalled. The ilma Python package's `initialize_schema()`
creates the `ilma` schema and tables on first API connection — no SQL
migration step needed.

### ilma-ollama (embedder, optional but recommended)

ilma defaults to `bge-m3` (1024-dim) as the embedding model. The image's
entrypoint pulls `bge-m3` on every container start — idempotent when the
named volume `/root/.ollama` already has the model on disk.

```bash
docker run -d --name ilma-ollama --restart always \
    --cpus=4 --memory=8g \
    --health-cmd="curl -fsS http://localhost:11434/api/tags || exit 1" \
    --health-interval=15s --health-timeout=5s --health-retries=10 \
    --health-start-period=180s \
    -v ilma-ollama-data:/root/.ollama \
    -p 127.0.0.1:11434:11434 \
    ghcr.io/brotal-llc/ilma-ollama:latest
```

The first boot takes 1-3 minutes while the model downloads (~2.2GB for
bge-m3). The healthcheck waits 180s before starting to account for that.
After the first run, subsequent restarts are instant because the model is
in the named volume.

**Why not bake the model into the image?** BuildKit's sandbox can't
reliably reach a long-running background `ollama serve` from inside a
single `RUN` step, so we can't pre-pull at build time. Pulling on every
start (with named-volume caching) is faster on subsequent runs and
keeps the image small (~1GB instead of ~3.2GB).

### ilma-agent (the API/MCP service)

Run the API/MCP service however you run Python services today (systemd,
docker, supervisor). All it needs is `ILMA_DSN` and optionally an embedder
URL:

```bash
# .env or systemd Environment= entries
ILMA_DSN=postgresql://ilma:change-me@ilma-db:5432/ilma
ILMA_EMBED_PROVIDER=ollama
ILMA_EMBED_BASE_URL=http://ilma-ollama:11434
ILMA_EMBED_MODEL=bge-m3
ILMA_EMBED_DIM=1024
```

See `ilma init` for the interactive wizard that writes a complete `.env`.

### docker-compose equivalent

If you prefer compose, here's the same recipe as a single file (typically
lives in `~/infra/ilma/compose.yaml` on the deploy host, NOT in this repo):

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
      POSTGRES_PASSWORD: change-me
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
      test: ["CMD", "curl", "-fsS", "http://localhost:11434/api/tags"]
      interval: 15s
      timeout: 5s
      retries: 10
      start_period: 180s

volumes:
  ilma-pg-data:
  ilma-pg-init:
  ilma-ollama-data:
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
| `ghcr.io/brotal-llc/ilma-ollama` | Ollama + bge-m3 (1024-dim embedder, auto-pulled on first start) | `latest`, `main`, `:{sha}` |

The ilma-pg image ships with: pgvector, pg_trgm, ltree, timescaledb,
pg_cron, and apache-age. The ilma Python package's `initialize_schema()`
creates the `ilma` schema and tables on first API connection — no SQL
migration step needed.

The ilma-ollama image is a thin wrapper over `ollama/ollama` that runs
`ollama pull bge-m3` on first start. Use it as the `ILMA_EMBED_BASE_URL`
target for any ilma installation that wants self-hosted embeddings.

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
├── ollama/                    # Ollama embedder image (Dockerfile + entrypoint)
│   ├── Dockerfile             # ollama/ollama + bge-m3 (1024-dim) auto-pulled
│   └── bin/                   # ilma-ollama-entrypoint.sh
├── tests/                     # unit + integration
│   ├── unit/                  # 226 tests, no DB
│   └── integration/           # 12 tests via Testcontainers (ilma-pg:latest)
├── Dockerfile                 # ilma CLI image (the API service)
├── pyproject.toml
└── .github/workflows/ilma.yml # single gated CI: lint, test, security, tag,
                              # build (ilma CLI), build-pg (Postgres),
                              # build-ollama (Ollama+bge-m3), release
```

> **No `docker/` or `infra/` directories.** Local-dev Docker Compose and
> Caddy configs live in `~/infra/` on the deployment host, not in this
> repo. The CI workflow publishes ready-to-pull images. See the
> "Production deployment" section above for the canonical recipe
> (container names `ilma-db` / `ilma-ollama`, named volumes, restart
> policy, resource limits).

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
