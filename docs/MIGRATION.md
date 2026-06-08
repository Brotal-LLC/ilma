# Migrating from hermes-memory v2 to ilma

This guide moves a Hermes Agent install that uses `hermes-memory` v2 Postgres storage to `ilma` without deleting the old schemas. The migration creates the `ilma` schema alongside the existing `agent_memory` and `hermes_*` schemas, copies data, and leaves the old data available for rollback.

## What is migrated

`ilma migrate` detects `agent_memory.memories` and migrates these v2 surfaces when their source tables exist:

- Memories: `agent_memory.memories`, `agent_memory.memory_chunks`
- Wiki: `hermes_wiki.documents`, `hermes_wiki.document_chunks`
- Journal: `hermes_journal.sessions`, `hermes_journal.messages`
- Skills: `hermes_skills.skills`
- Metrics: `hermes_metrics.events`
- Kanban: `hermes_kanban.tenants`, `hermes_kanban.tasks`, `hermes_kanban.task_links`
- Observability: `hermes_observability.logs`, `traces`, `spans`, `llm_calls`, `tool_calls`
- Sessions: `hermes_sessions.sessions`, `hermes_sessions.messages`

The migration is additive and idempotent. It stores Hermes source identifiers in `metadata` where the target table supports metadata and skips duplicates it can identify on re-runs. Duplicate memory content hashes are counted as conflicts and only one live row per `(content hash, source)` is imported.

## Pre-migration checklist

1. Confirm the Postgres DSN used by hermes-memory v2:

   ```bash
   echo "$HERMES_PG_CONN_STR"
   echo "$PG_MEM_DB_CONN_STR"
   ```

2. Create a database backup:

   ```bash
   pg_dump "$PG_MEM_DB_CONN_STR" > hermes-memory-v2-before-ilma.sql
   ```

3. Install ilma in the same environment where you will run the migration.
4. Make sure the target Postgres server has pgvector available. The migration creates `CREATE EXTENSION IF NOT EXISTS vector` when initializing ilma.
5. Stop writers or put Hermes in maintenance mode while the final migration runs. The migration is safe to re-run, but quiescing writers avoids missing rows inserted during the copy.

## 1. Dry-run the schema/data migration

Run a dry-run first. It detects source tables and reports row counts without writing any `ilma` rows:

```bash
ilma migrate --dsn "$PG_MEM_DB_CONN_STR" --dry-run
```

For machine-readable output:

```bash
ilma migrate --dsn "$PG_MEM_DB_CONN_STR" --dry-run --json
```

If `ILMA_DSN` or `PG_MEM_DB_CONN_STR` is already set, `--dsn` is optional.

## 2. Run the migration

```bash
ilma migrate --dsn "$PG_MEM_DB_CONN_STR"
```

Expected output shows progress and final counters:

```text
- detected hermes-memory v2 schema
- creating ilma schema alongside hermes-memory schemas
- migrating memories
- migrating wiki
...
Migration complete: inserted=... updated=... skipped=... conflicts=...
```

The old schemas are not renamed or dropped. The new tables are under the `ilma` schema.

## 3. Migrate Hermes config

`ilma migrate-config` updates `~/.hermes/config.yaml`:

- Changes `memory.provider` from `postgres` to `ilma`
- Adds `env.ILMA_DSN` if no `ILMA_DSN` is already present
- Derives the DSN from `ILMA_DSN`, `HERMES_PG_CONN_STR`, or `PG_MEM_DB_CONN_STR`
- Writes a timestamped backup next to the original config

Preview first:

```bash
ilma migrate-config --dry-run --json
```

Apply:

```bash
ilma migrate-config
```

Use a non-default config path if needed:

```bash
ilma migrate-config --config /path/to/config.yaml
```

## 4. Verify

Run:

```bash
export ILMA_DSN="$PG_MEM_DB_CONN_STR"
ilma status
ilma doctor
ilma search "a known memory phrase"
```

You can also inspect row counts directly:

```sql
SELECT count(*) FROM ilma.memories;
SELECT count(*) FROM ilma.wiki_docs;
SELECT count(*) FROM ilma.session_messages;
```

## Rollback

Because the migration is additive, rollback is straightforward:

1. Restore the config backup printed by `ilma migrate-config`:

   ```bash
   cp ~/.hermes/config.yaml.bak.<timestamp> ~/.hermes/config.yaml
   ```

2. Point Hermes back to the old provider/DSN if you changed environment variables.
3. If you want to remove migrated data, drop only the `ilma` schema:

   ```sql
   DROP SCHEMA IF EXISTS ilma CASCADE;
   ```

4. If a database-level issue occurred, restore the `pg_dump` backup you made before migration.

## Troubleshooting

### `no hermes-memory v2 agent_memory.memories table found`

The DSN points at a database that does not contain hermes-memory v2 data. Re-check `HERMES_PG_CONN_STR` / `PG_MEM_DB_CONN_STR` and rerun with `--dsn`.

### `CREATE EXTENSION vector` fails

Install pgvector on the Postgres server or migrate to a pgvector-enabled database image such as `pgvector/pgvector:pg16`.

### Duplicate/conflict counters are non-zero

This usually means hermes-memory contained duplicate live memories with the same content hash and source, or an earlier ilma row already matches the source. The migration skips duplicates rather than creating multiple identical live memories.

### Migration was interrupted

Re-run the same command. The migration records Hermes identifiers in target metadata and uses natural keys where available, so it is safe to retry.

### Config migration cannot derive `ILMA_DSN`

Set one of these and rerun:

```bash
export ILMA_DSN="postgresql://..."
# or
export PG_MEM_DB_CONN_STR="postgresql://..."
# or
export HERMES_PG_CONN_STR="postgresql://..."
```
