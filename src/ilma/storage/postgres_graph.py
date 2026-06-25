"""Postgres + Apache AGE graph repository for ilma.

The graph is a **derived view** of the relational state in ``ilma.*``. This
module handles Cypher execution; the planning logic (which edges to create
from which source rows) lives in :mod:`ilma.core.graph` so it can be tested
without a database.

Wire format reminders for AGE 1.7.0 on PG18:

- The ``age`` extension must be in ``shared_preload_libraries``. Once it is,
  ``LOAD 'age'`` is also acceptable but redundant at session level — we do
  it defensively in each connection.
- Cypher is invoked via ``SELECT * FROM cypher('graph_name', $$ ... $$) AS
  (col type, ...)``. Use ``$$ ... $$`` so we don't have to escape ``$`` in
  dollar-quoted parameter strings.
- Return types: vertex/edge come back as ``agtype`` strings with a
  ``::vertex`` / ``::edge`` suffix on each element. Use
  :func:`ilma.core.graph.parse_agtype` to normalize.
- Per-graph schema: ``create_graph('name')`` creates both a row in
  ``ag_graph`` and a schema named ``name`` that contains the per-graph
  labels and edges.
- CREATE EXTENSION / create_graph / drop_graph are session-level; do NOT
  call them inside a transaction. We always run them via ``autocommit=True``.

Caveat documented in :class:`v2-ilma-schema-cleanup` skill section 4:
the AGE C library registers DDL hooks. Dropping the per-graph schema
directly can fail with ``table ag_label does not exist``. Always use
``SELECT drop_graph(name, true)`` to remove a graph — it handles the
internal cleanup via the hook.

Schema bootstrap contract:

- ``ensure_age_extension(dsn)`` — idempotent CREATE EXTENSION age.
- ``ensure_graph(dsn)`` — idempotent create_graph, also called by
  ``PgBackend.initialize_schema``.
- The graph is named ``ilma_graph`` (constant). One graph per DB.

Known AGE quirks we work around:

1. ``SET n += row.props`` raises ``SET clause expects a map`` when ``props``
   comes from a UNWIND variable. We expand each property into an explicit
   ``SET n.k = row.props.k`` clause.
2. Parameter passing through ``cypher()`` is awkward. We inline values as
   Cypher literals instead of using ``$param`` placeholders.
3. Graph names must be ≥3 characters. ``ilma_graph`` is fine.
4. ``ag_graph.name`` (not ``graph_name``) is the column.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Literal

import psycopg

from ilma.core.graph import (
    AgtypeEdge,
    AgtypeVertex,
    GraphEdge,
    GraphRebuildPlan,
    GraphVertex,
    cypher_quote,
    parse_agtype,
)

log = logging.getLogger(__name__)

GRAPH_NAME = "ilma_graph"
VertexKind = Literal["Memory", "Wiki", "Skill"]


# ---------------------------------------------------------------------------
# Bootstrap helpers — module-level so service.py / initialize_schema can use them
# ---------------------------------------------------------------------------


def age_available(dsn: str) -> bool:
    """True if the ``age`` extension can be loaded (binary is present).

    Note: this checks ``pg_available_extensions``, NOT ``pg_extension``. A
    freshly-created database will have the AGE binary on disk but the
    extension not yet installed. Use :func:`ensure_age_extension` to install
    it before any Cypher execution.
    """
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_available_extensions WHERE name = 'age'")
        return cur.fetchone() is not None


def ensure_age_extension(dsn: str) -> None:
    """Install the ``age`` extension. Idempotent. Requires autocommit because
    CREATE EXTENSION cannot run inside a transaction block."""
    with psycopg.connect(dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS age")


def ensure_graph(dsn: str) -> None:
    """Create the ilma graph if it doesn't already exist. Idempotent."""
    with psycopg.connect(dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("LOAD 'age'")
        cur.execute("SET search_path = ag_catalog, public")
        cur.execute("SELECT count(*) FROM ag_graph WHERE name = %s", (GRAPH_NAME,))
        row = cur.fetchone()
        exists = row is not None and row[0] > 0
        if not exists:
            cur.execute(f"SELECT create_graph({cypher_quote(GRAPH_NAME)})")


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class PgGraphRepo:
    """Postgres + Apache AGE implementation of the ilma graph layer.

    Usage::

        repo = PgGraphRepo(dsn)
        repo.ensure_schema()      # one-time
        stats = repo.rebuild(plan) # whenever data changes
        sub = repo.traverse("Memory", src_id=42, max_hops=2)
    """

    def __init__(self, dsn: str, *, min_pool_size: int = 1, max_pool_size: int = 4) -> None:
        self._dsn = dsn
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size

    def ensure_schema(self) -> None:
        """Idempotent: ensure age extension + the ilma graph exist."""
        ensure_age_extension(self._dsn)
        ensure_graph(self._dsn)

    def connection(self) -> Any:
        """Yield a raw psycopg connection for callers that need to query
        the SQL state directly (e.g. the rebuild service method). The caller
        is responsible for transaction handling.
        """
        return psycopg.connect(self._dsn)

    # ---- Internal Cypher helpers ----------------------------------------

    def _cypher_literal(self, value: Any) -> str:
        """Serialize a Python value to a Cypher literal (recursive).

        None becomes Cypher NULL. Dicts become {key: value, ...} maps. Lists
        become [a, b, c] arrays. Strings are single-quoted with doubled
        single-quotes for escape. Decimal (psycopg returns Decimal for
        NUMERIC columns) becomes a float Cypher literal.
        """
        if value is None:
            return "NULL"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        # Postgres NUMERIC/DECIMAL columns come back as Decimal; coerce.
        if isinstance(value, Decimal):
            return str(float(value))
        if isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace("'", "''")
            return f"'{escaped}'"
        if isinstance(value, list):
            return "[" + ", ".join(self._cypher_literal(v) for v in value) + "]"
        if isinstance(value, dict):
            parts: list[str] = []
            for k, v in value.items():
                if not isinstance(k, str) or not k.replace("_", "").isalnum():
                    raise ValueError(f"_cypher_literal: bad map key {k!r}")
                parts.append(f"{k}: {self._cypher_literal(v)}")
            return "{" + ", ".join(parts) + "}"
        raise TypeError(f"_cypher_literal: unsupported value type {type(value).__name__}")

    @staticmethod
    def _clean_props(props: dict[str, Any]) -> dict[str, Any]:
        """Strip None and empty values — they make AGE's SET clause complain
        or assign NULL where the caller wanted absence."""
        return {k: v for k, v in props.items() if v is not None and v != {} and v != [] and v != ""}

    def _run_cypher(self, cypher: str) -> list[Any]:
        """Execute one Cypher statement and return rows from a single agtype col."""
        with psycopg.connect(self._dsn) as conn, conn.cursor() as cur:
            cur.execute("LOAD 'age'")
            cur.execute("SET search_path = ag_catalog, public")
            cur.execute(
                f"SELECT * FROM cypher({cypher_quote(GRAPH_NAME)}, $${cypher}$$) AS (v agtype)"
            )
            return list(cur.fetchall())

    # ---- Rebuild ---------------------------------------------------------

    def rebuild(self, plan: GraphRebuildPlan) -> dict[str, int]:
        """Drop the graph contents and re-create from the plan. Idempotent.

        Returns the plan.stats dict for convenience (also accessible via
        ``plan.stats`` directly).
        """
        # 1. Drop + recreate the graph. drop_graph() handles the AGE C-hook
        #    cleanup correctly; do NOT drop the per-graph schema manually.
        with psycopg.connect(self._dsn, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute("LOAD 'age'")
            cur.execute("SET search_path = ag_catalog, public")
            cur.execute("SELECT count(*) FROM ag_graph WHERE name = %s", (GRAPH_NAME,))
            row = cur.fetchone()
            if row is not None and row[0] > 0:
                cur.execute(f"SELECT drop_graph({cypher_quote(GRAPH_NAME)}, true)")
            cur.execute(f"SELECT create_graph({cypher_quote(GRAPH_NAME)})")

        # 2. Insert vertices in batches.
        if plan.vertices:
            self._bulk_load_vertices(plan.vertices)

        # 3. Insert edges.
        if plan.edges:
            self._bulk_load_edges(plan.edges)

        return dict(plan.stats)

    def _bulk_load_vertices(self, vertices: list[GraphVertex]) -> None:
        """Load all vertices in one Cypher UNWIND batch per kind.

        AGE quirk: ``SET n += row.props`` raises ``SET clause expects a map``
        when ``props`` comes from a UNWIND variable. We work around it by
        expanding each property into an explicit
        ``SET n.<key> = row.props.<key>`` clause. This requires all keys to be
        known at SQL-build time, which is fine because every vertex of the
        same kind shares the property schema.
        """
        from collections import defaultdict

        by_kind: dict[str, list[GraphVertex]] = defaultdict(list)
        for v in vertices:
            by_kind[v.kind].append(v)

        for kind, batch in by_kind.items():
            all_keys: list[str] = sorted(
                {k for v in batch for k in self._clean_props(v.properties)}
            )
            if not all_keys:
                rows_literal = "[" + ", ".join(f"{{id: {v.src_id}}}" for v in batch) + "]"
                cypher = f"UNWIND {rows_literal} AS row MERGE (n:{kind} {{id: row.id}})"
            else:
                rows_literal = (
                    "["
                    + ", ".join(
                        f"{{id: {v.src_id}, props: {self._cypher_literal(self._clean_props(v.properties))}}}"
                        for v in batch
                    )
                    + "]"
                )
                set_clauses = ", ".join(f"n.{k} = row.props.{k}" for k in all_keys)
                cypher = (
                    f"UNWIND {rows_literal} AS row "
                    f"MERGE (n:{kind} {{id: row.id}}) "
                    f"SET {set_clauses}"
                )
            self._run_cypher(cypher)

    def _bulk_load_edges(self, edges: list[GraphEdge]) -> None:
        """Load all edges in one Cypher UNWIND batch per label."""
        from collections import defaultdict

        by_label: dict[str, list[GraphEdge]] = defaultdict(list)
        for e in edges:
            by_label[e.label].append(e)

        for label, batch in by_label.items():
            all_keys: list[str] = sorted(
                {k for e in batch for k in self._clean_props(e.properties)}
            )
            rows_literal = (
                "["
                + ", ".join(
                    "{"
                    f"src_id: {e.src_id}, "
                    f"dst_id: {e.dst_id}, "
                    f"props: {self._cypher_literal(self._clean_props(e.properties))}"
                    "}"
                    for e in batch
                )
                + "]"
            )
            if all_keys:
                set_clauses = ", ".join(f"r.{k} = row.props.{k}" for k in all_keys)
                cypher = (
                    f"UNWIND {rows_literal} AS row "
                    f"MATCH (src {{id: row.src_id}}), (dst {{id: row.dst_id}}) "
                    f"MERGE (src)-[r:{label}]->(dst) "
                    f"SET {set_clauses}"
                )
            else:
                cypher = (
                    f"UNWIND {rows_literal} AS row "
                    f"MATCH (src {{id: row.src_id}}), (dst {{id: row.dst_id}}) "
                    f"MERGE (src)-[r:{label}]->(dst)"
                )
            self._run_cypher(cypher)

    # ---- Traverse --------------------------------------------------------

    def traverse(
        self,
        *,
        kind: str,
        src_id: int,
        max_hops: int = 2,
        edge_types: list[str] | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Bounded BFS from a starting vertex.

        Returns ``{"nodes": [...], "edges": [...]}`` where each node has
        ``kind``, ``src_id``, ``vertex_id`` (AGE internal), and
        ``properties``. Each edge has ``label``, ``start_id``, ``end_id``,
        ``edge_id``, and ``properties``.

        Parameters
        ----------
        kind
            One of ``"Memory"``, ``"Wiki"``, ``"Skill"``. Other values raise
            ``ValueError``.
        src_id
            The SQL-table id (the ``id`` property we MERGE'd), NOT the
            AGE-internal vertex id.
        max_hops
            0..3. 0 returns only the start vertex (no edges).
        edge_types
            Optional whitelist of edge labels to traverse. None = all labels.
        limit
            Maximum number of (node, edge) pairs to return. Hard cap 500.
        """
        if kind not in ("Memory", "Wiki", "Skill"):
            raise ValueError(f"Unknown vertex kind: {kind!r}")
        if max_hops < 0 or max_hops > 3:
            raise ValueError(f"max_hops must be in 0..3, got {max_hops}")
        limit = max(1, min(limit, 500))

        if edge_types:
            labels = "|".join(edge_types)
            rel_pat = f"[r:{labels}*1..{max_hops}]"
        else:
            rel_pat = f"[r*1..{max_hops}]"

        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []

        # Always include the start vertex itself in the result set.
        start_cypher = f"MATCH (start:{kind} {{id: {int(src_id)}}}) RETURN start AS n"
        for (raw,) in self._run_cypher(start_cypher):
            v = parse_agtype(raw)
            if isinstance(v, AgtypeVertex):
                nodes.append(self._normalize_vertex(v, kind))

        if max_hops == 0:
            return {"nodes": nodes, "edges": edges}

        # Fetch neighbor nodes (deduped).
        nodes_cypher = (
            f"MATCH (start:{kind} {{id: {int(src_id)}}})-{rel_pat}-(other) "
            "WITH collect(DISTINCT other) AS ns "
            "UNWIND ns AS n RETURN n"
        )
        seen: set[tuple[str, int]] = {(n["kind"], n["src_id"]) for n in nodes}
        for (raw,) in self._run_cypher(nodes_cypher):
            if len(nodes) >= limit:
                break
            v = parse_agtype(raw)
            if not isinstance(v, AgtypeVertex):
                continue
            sid = int(v.properties.get("id", v.vertex_id))
            key = (v.label, sid)
            if key in seen:
                continue
            seen.add(key)
            nodes.append(self._normalize_vertex(v, v.label))

        # Fetch edges. Single-hop only for the edge enumeration because Cypher
        # can't extract relationships from a variable-length pattern. For
        # multi-hop traversals, the relationship count is bounded by the
        # neighbor set anyway; we just fetch every edge adjacent to any
        # neighbor.
        edge_type_filter = ":" + "|".join(edge_types) if edge_types else ""
        edges_cypher = (
            f"MATCH (start:{kind} {{id: {int(src_id)}}})-[r{edge_type_filter}]-() RETURN DISTINCT r"
        )
        seen_e: set[tuple[str, int]] = set()
        for (raw,) in self._run_cypher(edges_cypher):
            if len(edges) >= limit:
                break
            e = parse_agtype(raw)
            if not isinstance(e, AgtypeEdge):
                continue
            if (e.label, e.edge_id) in seen_e:
                continue
            seen_e.add((e.label, e.edge_id))
            edges.append(
                {
                    "edge_id": e.edge_id,
                    "label": e.label,
                    "start_id": e.start_id,
                    "end_id": e.end_id,
                    "properties": e.properties,
                }
            )

        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def _normalize_vertex(v: AgtypeVertex, kind: str) -> dict[str, Any]:
        sid = v.properties.get("id", v.vertex_id)
        return {
            "kind": kind or v.label,
            "src_id": int(sid),
            "vertex_id": v.vertex_id,
            "properties": dict(v.properties),
        }


__all__ = [
    "GRAPH_NAME",
    "PgGraphRepo",
    "age_available",
    "cypher_quote",
    "ensure_age_extension",
    "ensure_graph",
    "parse_agtype",
]
