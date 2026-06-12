"""Migration helpers for moving hermes-memory v2 data/config into ilma."""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from psycopg import Connection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from ilma.api.mcp import IlmaMcpService
from ilma.storage.postgres import PgBackend, close_all_pools

Progress = Callable[[str], None]

HERMES_SURFACE_TABLES: Mapping[str, tuple[tuple[str, str], ...]] = {
    "memory": (("agent_memory", "memories"), ("agent_memory", "memory_chunks")),
    "wiki": (("hermes_wiki", "documents"), ("hermes_wiki", "document_chunks")),
    "journal": (("hermes_journal", "sessions"), ("hermes_journal", "messages")),
    "skills": (("hermes_skills", "skills"),),
    "metrics": (("hermes_metrics", "events"),),
    "kanban": (("hermes_kanban", "tenants"), ("hermes_kanban", "tasks")),
    "observability": (
        ("hermes_observability", "logs"),
        ("hermes_observability", "traces"),
        ("hermes_observability", "spans"),
        ("hermes_observability", "llm_calls"),
        ("hermes_observability", "tool_calls"),
    ),
    "sessions": (("hermes_sessions", "sessions"), ("hermes_sessions", "messages")),
}


@dataclass
class SurfaceStats:
    """Migration counters for one surface."""

    source_rows: int = 0
    inserted: int = 0
    updated: int = 0
    skipped: int = 0
    conflicts: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_rows": self.source_rows,
            "inserted": self.inserted,
            "updated": self.updated,
            "skipped": self.skipped,
            "conflicts": self.conflicts,
            "warnings": self.warnings,
        }


def _now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _table_exists(conn: Connection[Any], schema: str, table: str) -> bool:
    row = conn.execute(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s
        )
        """,
        (schema, table),
    ).fetchone()
    return bool(row and row[0])


def _column_exists(conn: Connection[Any], schema: str, table: str, column: str) -> bool:
    row = conn.execute(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s AND column_name = %s
        )
        """,
        (schema, table, column),
    ).fetchone()
    return bool(row and row[0])


def _count(conn: Connection[Any], schema: str, table: str) -> int:
    if not _table_exists(conn, schema, table):
        return 0
    row = conn.execute(f"SELECT count(*) FROM {schema}.{table}").fetchone()
    return int(row[0]) if row else 0


def _jsonb(value: Any) -> Jsonb:
    return Jsonb(value if value is not None else {})


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _as_str_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()


def detect_hermes_v2_schema(dsn: str) -> dict[str, Any]:
    """Detect hermes-memory v2 source schemas/tables in a Postgres database."""

    with Connection.connect(dsn) as conn:
        surfaces: dict[str, Any] = {}
        for surface, tables in HERMES_SURFACE_TABLES.items():
            present = [
                f"{schema}.{table}"
                for schema, table in tables
                if _table_exists(conn, schema, table)
            ]
            row_count = sum(_count(conn, schema, table) for schema, table in tables)
            surfaces[surface] = {
                "present": bool(present),
                "tables": present,
                "source_rows": row_count,
            }
        memory_schema = _table_exists(conn, "agent_memory", "memories")
        return {
            "detected": memory_schema,
            "memory_schema": memory_schema,
            "surfaces": surfaces,
        }


def _initialize_ilma_schema(dsn: str) -> None:
    close_all_pools()
    backend = PgBackend(dsn, min_pool_size=1, max_pool_size=4)
    service = IlmaMcpService(backend)
    result = service.ilma_migrate()
    if not result.get("ok"):
        error = result.get("error")
        raise RuntimeError(f"failed to initialize ilma schema: {error}")


def migrate_hermes_v2_schema(
    dsn: str,
    *,
    dry_run: bool = False,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Migrate hermes-memory v2 Postgres tables into the ilma schema.

    The migration is additive and leaves all hermes_* / agent_memory schemas in
    place. Re-runs are safe: natural keys and hermes_v2 metadata are used to avoid
    duplicate rows where possible.
    """

    emit = progress or (lambda _message: None)
    detected = detect_hermes_v2_schema(dsn)
    if not detected["detected"]:
        return {
            "ok": True,
            "detected": False,
            "dry_run": dry_run,
            "message": "no hermes-memory v2 agent_memory.memories table found",
            "surfaces": detected["surfaces"],
        }

    emit("detected hermes-memory v2 schema")
    surfaces: dict[str, SurfaceStats]
    if dry_run:
        with Connection.connect(dsn) as conn:
            surfaces = _dry_run_counts(conn)
        return {
            "ok": True,
            "detected": True,
            "dry_run": True,
            "message": "dry-run complete; no data was written",
            "surfaces": {name: stats.as_dict() for name, stats in surfaces.items()},
        }

    emit("creating ilma schema alongside hermes-memory schemas")
    _initialize_ilma_schema(dsn)

    close_all_pools()
    with Connection.connect(dsn) as conn:
        conn.autocommit = False
        try:
            surfaces = {}
            memory_id_map: dict[int, int] = {}
            wiki_id_map: dict[int, int] = {}
            kanban_id_map: dict[str, int] = {}

            emit("migrating memories")
            surfaces["memory"] = _migrate_memory(conn, memory_id_map)
            emit("migrating wiki")
            surfaces["wiki"] = _migrate_wiki(conn, wiki_id_map)
            emit("migrating journal")
            surfaces["journal"] = _migrate_journal(conn)
            emit("migrating skills")
            surfaces["skills"] = _migrate_skills(conn)
            emit("migrating metrics")
            surfaces["metrics"] = _migrate_metrics(conn)
            emit("migrating kanban")
            surfaces["kanban"] = _migrate_kanban(conn, kanban_id_map)
            emit("migrating observability")
            surfaces["observability"] = _migrate_observability(conn)
            emit("migrating sessions")
            surfaces["sessions"] = _migrate_sessions(conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    total_inserted = sum(stats.inserted for stats in surfaces.values())
    total_updated = sum(stats.updated for stats in surfaces.values())
    total_skipped = sum(stats.skipped for stats in surfaces.values())
    total_conflicts = sum(stats.conflicts for stats in surfaces.values())
    emit("migration complete")
    return {
        "ok": True,
        "detected": True,
        "dry_run": False,
        "message": "hermes-memory v2 data migrated to ilma",
        "inserted": total_inserted,
        "updated": total_updated,
        "skipped": total_skipped,
        "conflicts": total_conflicts,
        "surfaces": {name: stats.as_dict() for name, stats in surfaces.items()},
    }


def _dry_run_counts(conn: Connection[Any]) -> dict[str, SurfaceStats]:
    stats: dict[str, SurfaceStats] = {}
    for surface, tables in HERMES_SURFACE_TABLES.items():
        item = SurfaceStats()
        item.source_rows = sum(_count(conn, schema, table) for schema, table in tables)
        stats[surface] = item
    return stats


def _migrate_memory(conn: Connection[Any], memory_id_map: dict[int, int]) -> SurfaceStats:
    stats = SurfaceStats()
    if not _table_exists(conn, "agent_memory", "memories"):
        stats.warnings.append("agent_memory.memories missing")
        return stats
    category_expr = (
        "category::text" if _column_exists(conn, "agent_memory", "memories", "category") else "NULL"
    )
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT id, content, tags, {category_expr} AS category, metadata, source,
                   vector_768::text AS vector_768,
                   vector_1024::text AS vector_1024,
                   vector_1536::text AS vector_1536,
                   created_at, deleted_at
            FROM agent_memory.memories
            ORDER BY id
            """
        )
        rows = cur.fetchall()
        stats.source_rows += len(rows)
        seen_hashes: set[tuple[str, str | None]] = set()
        for row in rows:
            content = str(row["content"])
            source = row["source"]
            content_key = (_content_hash(content), source)
            existing = conn.execute(
                """
                SELECT id FROM ilma.memories
                WHERE (metadata ->> 'hermes_v2_id') = %s
                   OR (content = %s AND source IS NOT DISTINCT FROM %s AND deleted_at IS NULL)
                ORDER BY id LIMIT 1
                """,
                (str(row["id"]), content, source),
            ).fetchone()
            if existing is not None:
                memory_id_map[int(row["id"])] = int(existing[0])
                stats.skipped += 1
                if content_key in seen_hashes:
                    stats.conflicts += 1
                seen_hashes.add(content_key)
                continue
            if content_key in seen_hashes:
                stats.conflicts += 1
                stats.skipped += 1
                continue
            metadata = _as_str_dict(row["metadata"])
            metadata.update(
                {
                    "hermes_v2_id": str(row["id"]),
                    "hermes_v2_content_hash": content_key[0],
                }
            )
            inserted = conn.execute(
                """
                INSERT INTO ilma.memories
                    (content, tags, category, source, metadata, vector_768, vector_1024,
                     vector_1536, created_at, deleted_at)
                VALUES (%s, %s, %s, %s, %s, %s::vector, %s::vector, %s::vector, %s, %s)
                RETURNING id
                """,
                (
                    content,
                    _as_list(row["tags"]),
                    row["category"],
                    source,
                    _jsonb(metadata),
                    row["vector_768"],
                    row["vector_1024"],
                    row["vector_1536"],
                    row["created_at"],
                    row["deleted_at"],
                ),
            ).fetchone()
            assert inserted is not None
            memory_id_map[int(row["id"])] = int(inserted[0])
            seen_hashes.add(content_key)
            stats.inserted += 1

    if _table_exists(conn, "agent_memory", "memory_chunks"):
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, memory_id, chunk_index, content, token_count,
                       vector_768::text AS vector_768,
                       vector_1024::text AS vector_1024,
                       vector_1536::text AS vector_1536,
                       created_at
                FROM agent_memory.memory_chunks
                ORDER BY memory_id, chunk_index
                """
            )
            chunk_rows = cur.fetchall()
            stats.source_rows += len(chunk_rows)
            for row in chunk_rows:
                target_memory_id = memory_id_map.get(int(row["memory_id"]))
                if target_memory_id is None:
                    stats.skipped += 1
                    continue
                inserted = conn.execute(
                    """
                    INSERT INTO ilma.memory_chunks
                        (memory_id, chunk_index, content, token_count, metadata,
                         vector_768, vector_1024, vector_1536, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s::vector, %s::vector, %s::vector, %s)
                    ON CONFLICT (memory_id, chunk_index) DO NOTHING
                    RETURNING id
                    """,
                    (
                        target_memory_id,
                        row["chunk_index"],
                        row["content"],
                        row["token_count"],
                        _jsonb({"hermes_v2_id": str(row["id"])}),
                        row["vector_768"],
                        row["vector_1024"],
                        row["vector_1536"],
                        row["created_at"],
                    ),
                ).fetchone()
                if inserted is None:
                    stats.skipped += 1
                else:
                    stats.inserted += 1
    return stats


def _migrate_wiki(conn: Connection[Any], wiki_id_map: dict[int, int]) -> SurfaceStats:
    stats = SurfaceStats()
    if not _table_exists(conn, "hermes_wiki", "documents"):
        stats.warnings.append("hermes_wiki.documents missing")
        return stats
    category_expr = (
        "d.category::text"
        if _column_exists(conn, "hermes_wiki", "documents", "category")
        else "NULL"
    )
    source_expr = (
        "d.source_uri" if _column_exists(conn, "hermes_wiki", "documents", "source_uri") else "NULL"
    )
    tags_expr = "array_remove(array_agg(t.name ORDER BY t.name), NULL)"
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT d.id, d.slug, d.title, d.body_md, {category_expr} AS category,
                   d.metadata, {source_expr} AS source_uri, d.created_at, d.updated_at,
                   {tags_expr} AS tags
            FROM hermes_wiki.documents d
            LEFT JOIN hermes_wiki.document_tags dt ON dt.document_id = d.id
            LEFT JOIN hermes_wiki.tags t ON t.id = dt.tag_id
            GROUP BY d.id
            ORDER BY d.id
            """
        )
        rows = cur.fetchall()
        stats.source_rows += len(rows)
        for row in rows:
            metadata = _as_str_dict(row["metadata"])
            metadata["hermes_v2_id"] = str(row["id"])
            inserted = conn.execute(
                """
                INSERT INTO ilma.wiki_docs
                    (slug, title, body_md, category, tags, source_uri, metadata,
                     created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (slug) DO UPDATE
                SET title = EXCLUDED.title,
                    body_md = EXCLUDED.body_md,
                    category = EXCLUDED.category,
                    tags = EXCLUDED.tags,
                    source_uri = EXCLUDED.source_uri,
                    metadata = ilma.wiki_docs.metadata || EXCLUDED.metadata,
                    updated_at = EXCLUDED.updated_at
                RETURNING id, (xmax = 0) AS inserted
                """,
                (
                    row["slug"],
                    row["title"],
                    row["body_md"],
                    row["category"],
                    _as_list(row["tags"]),
                    row["source_uri"],
                    _jsonb(metadata),
                    row["created_at"],
                    row["updated_at"],
                ),
            ).fetchone()
            assert inserted is not None
            wiki_id_map[int(row["id"])] = int(inserted[0])
            if bool(inserted[1]):
                stats.inserted += 1
            else:
                stats.updated += 1

    if _table_exists(conn, "hermes_wiki", "document_chunks"):
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, document_id, ordinal, content, token_count,
                       vector_768::text AS vector_768,
                       vector_1024::text AS vector_1024,
                       vector_1536::text AS vector_1536,
                       created_at
                FROM hermes_wiki.document_chunks
                ORDER BY document_id, ordinal
                """
            )
            chunk_rows = cur.fetchall()
            stats.source_rows += len(chunk_rows)
            for row in chunk_rows:
                target_doc_id = wiki_id_map.get(int(row["document_id"]))
                if target_doc_id is None:
                    stats.skipped += 1
                    continue
                inserted = conn.execute(
                    """
                    INSERT INTO ilma.wiki_chunks
                        (doc_id, chunk_index, content, token_count, vector_768,
                         vector_1024, vector_1536, created_at)
                    VALUES (%s, %s, %s, %s, %s::vector, %s::vector, %s::vector, %s)
                    ON CONFLICT (doc_id, chunk_index) DO NOTHING
                    RETURNING id
                    """,
                    (
                        target_doc_id,
                        row["ordinal"],
                        row["content"],
                        row["token_count"] or 1,
                        row["vector_768"],
                        row["vector_1024"],
                        row["vector_1536"],
                        row["created_at"],
                    ),
                ).fetchone()
                if inserted is None:
                    stats.skipped += 1
                else:
                    stats.inserted += 1
    return stats


def _migrate_journal(conn: Connection[Any]) -> SurfaceStats:
    stats = SurfaceStats()
    if not _table_exists(conn, "hermes_journal", "messages"):
        stats.warnings.append("hermes_journal.messages missing")
        return stats
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT m.id, m.session_id, m.role, m.content, m.ts, s.profile
            FROM hermes_journal.messages m
            LEFT JOIN hermes_journal.sessions s ON s.id = m.session_id
            ORDER BY m.ts, m.id
            """
        )
        rows = cur.fetchall()
        stats.source_rows += len(rows)
        for row in rows:
            existing = conn.execute(
                "SELECT id FROM ilma.journal_entries WHERE tags @> %s LIMIT 1",
                ([f"hermes_journal_message:{row['id']}"],),
            ).fetchone()
            if existing is not None:
                stats.skipped += 1
                continue
            tags = [
                f"hermes_journal_message:{row['id']}",
                f"session:{row['session_id']}",
                f"role:{row['role']}",
            ]
            if row["profile"]:
                tags.append(f"profile:{row['profile']}")
            conn.execute(
                """
                INSERT INTO ilma.journal_entries (content, tags, created_at)
                VALUES (%s, %s, %s)
                """,
                (row["content"], tags, row["ts"]),
            )
            stats.inserted += 1
    return stats


def _migrate_skills(conn: Connection[Any]) -> SurfaceStats:
    stats = SurfaceStats()
    if not _table_exists(conn, "hermes_skills", "skills"):
        stats.warnings.append("hermes_skills.skills missing")
        return stats
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, name, version, owner, description, tags, metadata, created_at, updated_at
            FROM hermes_skills.skills
            ORDER BY id
            """
        )
        rows = cur.fetchall()
        stats.source_rows += len(rows)
        for row in rows:
            content = row["description"] or f"Hermes skill {row['name']}"
            metadata = _as_str_dict(row["metadata"])
            metadata.update(
                {"hermes_v2_id": str(row["id"]), "version": row["version"], "owner": row["owner"]}
            )
            inserted = conn.execute(
                """
                INSERT INTO ilma.skills (name, content, category, tags, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (name) DO UPDATE
                SET content = EXCLUDED.content,
                    category = EXCLUDED.category,
                    tags = EXCLUDED.tags,
                    updated_at = EXCLUDED.updated_at
                RETURNING (xmax = 0) AS inserted
                """,
                (
                    row["name"],
                    content,
                    row["owner"],
                    _as_list(row["tags"]),
                    row["created_at"],
                    row["updated_at"],
                ),
            ).fetchone()
            if inserted and bool(inserted[0]):
                stats.inserted += 1
            else:
                stats.updated += 1
    return stats


def _migrate_metrics(conn: Connection[Any]) -> SurfaceStats:
    stats = SurfaceStats()
    if not _table_exists(conn, "hermes_metrics", "events"):
        stats.warnings.append("hermes_metrics.events missing")
        return stats
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT ts, profile, metric_name, value, tags FROM hermes_metrics.events ORDER BY ts"
        )
        rows = cur.fetchall()
        stats.source_rows += len(rows)
        for row in rows:
            labels = _as_str_dict(row["tags"])
            labels.setdefault("profile", row["profile"])
            labels.setdefault("hermes_v2", "true")
            conn.execute(
                """
                INSERT INTO ilma.metrics (name, value, labels, recorded_at)
                VALUES (%s, %s, %s, %s)
                """,
                (row["metric_name"], row["value"], _jsonb(labels), row["ts"]),
            )
            stats.inserted += 1
    return stats


def _status_to_ilma(status: str) -> str:
    return {"ready": "todo", "running": "in_progress", "blocked": "blocked", "done": "done"}.get(
        status, status
    )


def _migrate_kanban(conn: Connection[Any], kanban_id_map: dict[str, int]) -> SurfaceStats:
    stats = SurfaceStats()
    if not _table_exists(conn, "hermes_kanban", "tasks"):
        stats.warnings.append("hermes_kanban.tasks missing")
        return stats
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT t.*, ten.slug AS tenant_slug, ten.name AS tenant_name
            FROM hermes_kanban.tasks t
            JOIN hermes_kanban.tenants ten ON ten.id = t.tenant_id
            ORDER BY t.created_at, t.id
            """
        )
        rows = cur.fetchall()
        stats.source_rows += len(rows) + _count(conn, "hermes_kanban", "tenants")
        for row in rows:
            existing = conn.execute(
                "SELECT id FROM ilma.kanban_tasks WHERE metadata ->> 'hermes_v2_id' = %s LIMIT 1",
                (row["id"],),
            ).fetchone()
            if existing is not None:
                kanban_id_map[str(row["id"])] = int(existing[0])
                stats.skipped += 1
                continue
            metadata = {
                "hermes_v2_id": row["id"],
                "tenant_slug": row["tenant_slug"],
                "tenant_name": row["tenant_name"],
                "assignee": row["assignee"],
                "created_by": row["created_by"],
                "skills": row["skills"],
                "result": row["result"],
                "session_id": row["session_id"],
            }
            inserted = conn.execute(
                """
                INSERT INTO ilma.kanban_tasks
                    (title, description, status, priority, tags, metadata, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, now())
                RETURNING id
                """,
                (
                    row["title"],
                    row["body"] or "",
                    _status_to_ilma(str(row["status"])),
                    row["priority"],
                    [],
                    _jsonb(metadata),
                    row["created_at"],
                ),
            ).fetchone()
            assert inserted is not None
            kanban_id_map[str(row["id"])] = int(inserted[0])
            stats.inserted += 1
    if _table_exists(conn, "hermes_kanban", "task_links"):
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT parent_id, child_id FROM hermes_kanban.task_links ORDER BY parent_id, child_id"
            )
            links = cur.fetchall()
            stats.source_rows += len(links)
            for row in links:
                parent = kanban_id_map.get(str(row["parent_id"]))
                child = kanban_id_map.get(str(row["child_id"]))
                if parent is None or child is None:
                    stats.skipped += 1
                    continue
                conn.execute(
                    "UPDATE ilma.kanban_tasks SET parent_id = %s WHERE id = %s", (parent, child)
                )
                stats.updated += 1
    return stats


def _migrate_observability(conn: Connection[Any]) -> SurfaceStats:
    stats = SurfaceStats()
    if _table_exists(conn, "hermes_observability", "logs"):
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM hermes_observability.logs ORDER BY ts")
            rows = cur.fetchall()
            stats.source_rows += len(rows)
            for row in rows:
                context = _as_str_dict(row["metadata"])
                context.update(
                    {
                        "logger": row["logger"],
                        "exception": row["exception"],
                        "profile": row["profile"],
                        "session_id": row["session_id"],
                        "task_id": row["task_id"],
                        "platform": row["platform"],
                    }
                )
                conn.execute(
                    """
                    INSERT INTO ilma.observations (level, message, source, context, recorded_at)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (row["level"], row["message"], row["logger"], _jsonb(context), row["ts"]),
                )
                stats.inserted += 1
    for table, message_field in (
        ("traces", "name"),
        ("spans", "name"),
        ("llm_calls", "model"),
        ("tool_calls", "tool_name"),
    ):
        if not _table_exists(conn, "hermes_observability", table):
            continue
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(f"SELECT * FROM hermes_observability.{table} ORDER BY ts")
            rows = cur.fetchall()
            stats.source_rows += len(rows)
            for row in rows:
                context = dict(row)
                ts = context.pop("ts", None)
                message = f"hermes {table}: {context.get(message_field) or 'event'}"
                conn.execute(
                    """
                    INSERT INTO ilma.observations (level, message, source, context, recorded_at)
                    VALUES ('info', %s, %s, %s, %s)
                    """,
                    (message, f"hermes_observability.{table}", _jsonb(context), ts),
                )
                stats.inserted += 1
    if stats.source_rows == 0:
        stats.warnings.append("hermes_observability tables missing or empty")
    return stats


def _migrate_sessions(conn: Connection[Any]) -> SurfaceStats:
    stats = SurfaceStats()
    if not _table_exists(conn, "hermes_sessions", "sessions"):
        stats.warnings.append("hermes_sessions.sessions missing")
        return stats
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM hermes_sessions.sessions ORDER BY started_at, id")
        rows = cur.fetchall()
        stats.source_rows += len(rows)
        for row in rows:
            conn.execute(
                """
                INSERT INTO ilma.sessions (session_id, created_at, updated_at)
                VALUES (%s, %s, COALESCE(%s, %s))
                ON CONFLICT (session_id) DO UPDATE
                SET updated_at = GREATEST(ilma.sessions.updated_at, EXCLUDED.updated_at)
                """,
                (row["id"], row["started_at"], row["ended_at"], row["started_at"]),
            )
            stats.inserted += 1
    if _table_exists(conn, "hermes_sessions", "messages"):
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM hermes_sessions.messages ORDER BY timestamp, id")
            message_rows = cur.fetchall()
            stats.source_rows += len(message_rows)
            for row in message_rows:
                existing = conn.execute(
                    """
                    SELECT id FROM ilma.session_messages
                    WHERE session_id = %s AND created_at = %s AND role = %s AND content = %s
                    LIMIT 1
                    """,
                    (row["session_id"], row["timestamp"], row["role"], row["content"]),
                ).fetchone()
                if existing is not None:
                    stats.skipped += 1
                    continue
                conn.execute(
                    """
                    INSERT INTO ilma.session_messages (session_id, role, content, created_at)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (row["session_id"], row["role"], row["content"], row["timestamp"]),
                )
                stats.inserted += 1
    return stats


def _find_existing_ilma_dsn(config: Mapping[str, Any]) -> str | None:
    for key in ("ILMA_DSN", "ilma_dsn"):
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for section in ("env", "environment"):
        env = config.get(section)
        if isinstance(env, Mapping):
            value = env.get("ILMA_DSN")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _derive_dsn(config: Mapping[str, Any], env: Mapping[str, str]) -> str | None:
    existing = _find_existing_ilma_dsn(config)
    if existing:
        return existing
    for key in ("ILMA_DSN", "HERMES_PG_CONN_STR", "PG_MEM_DB_CONN_STR"):
        value = env.get(key)
        if value and value.strip():
            return value.strip()
    for section in ("env", "environment"):
        cfg_env = config.get(section)
        if isinstance(cfg_env, Mapping):
            for key in ("HERMES_PG_CONN_STR", "PG_MEM_DB_CONN_STR"):
                value = cfg_env.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    for key in ("HERMES_PG_CONN_STR", "PG_MEM_DB_CONN_STR"):
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def migrate_hermes_config(
    config_path: str | Path | None = None,
    *,
    dry_run: bool = False,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Update a Hermes config.yaml to use ilma and preserve a backup."""

    path = (
        Path(config_path).expanduser() if config_path else Path.home() / ".hermes" / "config.yaml"
    )
    if not path.exists():
        raise FileNotFoundError(f"Hermes config not found: {path}")
    original_text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(original_text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Hermes config must be a YAML mapping: {path}")
    source_env = env if env is not None else os.environ
    dsn = _derive_dsn(data, source_env)
    if not dsn:
        raise ValueError(
            "could not derive ILMA_DSN from ILMA_DSN, HERMES_PG_CONN_STR, or PG_MEM_DB_CONN_STR"
        )

    changed = False
    memory = data.setdefault("memory", {})
    if not isinstance(memory, dict):
        raise ValueError("config key 'memory' exists but is not a mapping")
    old_provider = memory.get("provider")
    if old_provider != "ilma":
        memory["provider"] = "ilma"
        changed = True

    if not _find_existing_ilma_dsn(data):
        cfg_env = data.setdefault("env", {})
        if not isinstance(cfg_env, dict):
            raise ValueError("config key 'env' exists but is not a mapping")
        cfg_env["ILMA_DSN"] = dsn
        changed = True

    backup_path: Path | None = None
    if changed and not dry_run:
        backup_path = path.with_name(f"{path.name}.bak.{_now_stamp()}")
        shutil.copy2(path, backup_path)
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    return {
        "ok": True,
        "dry_run": dry_run,
        "changed": changed,
        "config_path": str(path),
        "backup_path": str(backup_path) if backup_path else None,
        "old_provider": old_provider,
        "new_provider": memory.get("provider"),
        "ilma_dsn": "set" if dsn else "missing",
        "config": data if dry_run else None,
    }


__all__ = [
    "detect_hermes_v2_schema",
    "migrate_hermes_config",
    "migrate_hermes_v2_schema",
]
