# ilma

**Framework-agnostic agent memory system.**

Postgres + pgvector backend. MCP-native. Hermes Agent, Claude, Cursor, Codex — any MCP client.

```bash
pip install ilma
ilma init
ilma status
```

---

## What ilma is

ilma stores what your agents know — and makes it retrievable across sessions, frameworks, and machines.

- **8 memory surfaces**: memories, wiki, journal, skills, metrics, kanban, observability, sessions
- **Hybrid retrieval**: vector + FTS + chunk-level reranking
- **MCP server**: `ilma-mcp` — works with any MCP client
- **HTTP API**: REST endpoints behind your own reverse proxy
- **CLI**: `ilma init`, `ilma search`, `ilma remember`, `ilma doctor`
- **Postgres + pgvector**: proven, backup-friendly, multi-client

## Quick start

```bash
# Install
pip install ilma

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
Postgres + pgvector
    ├─ ilma.memories
    ├─ ilma.wiki
    ├─ ilma.journal
    ├─ ilma.skills
    ├─ ilma.metrics
    ├─ ilma.kanban
    ├─ ilma.observability
    └─ ilma.sessions
```

## Project status

In active development. See [PLAN.md](PLAN.md) for the execution roadmap.

## License

MIT
