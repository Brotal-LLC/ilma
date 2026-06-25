"""Unit tests for ilma.core.graph — pure pair-builder logic for the AGE graph layer.

These tests exercise the pure-Python logic (cypher escaping, edge-pair extraction
from SQL result rows, agtype JSON parsing) without any database. The integration
tests in tests/integration/test_graph.py exercise the full Cypher round-trip.

Design decisions encoded here:
- cypher_quote() must escape single quotes, backslashes, and the dollar-sign
  inside $$...$$ Cypher blocks (or any other delimiter).
- parse_agtype() must return dicts/lists/scalars matching the AGE 1.7.0 agtype
  wire format. Vertices come back as {"id": <int>, "label": <str>, "properties":
  {...}}. Edges as {"id": <int>, "label": <str>, "start_id": <int>, "end_id":
  <int>, "properties": {...}}.
- Pair builders return plain dataclasses — they don't execute SQL.
"""

from __future__ import annotations

from typing import Any

from ilma.core.graph import (
    AgtypeEdge,
    AgtypeVertex,
    cypher_quote,
    parse_agtype,
    plan_graph_rebuild,
)

# ---------------------------------------------------------------------------
# cypher_quote — Cypher string literal escape
# ---------------------------------------------------------------------------


class TestCypherQuote:
    def test_simple_string(self) -> None:
        assert cypher_quote("hello") == "'hello'"

    def test_empty_string(self) -> None:
        assert cypher_quote("") == "''"

    def test_single_quote_doubled(self) -> None:
        # Cypher escapes single quotes by doubling them (SQL-style).
        assert cypher_quote("it's") == "'it''s'"

    def test_double_quote_passes_through(self) -> None:
        # Double quotes are identifier delimiters in Cypher, not string escapes.
        # In string literals, double quotes are literal characters.
        assert cypher_quote('say "hi"') == "'say \"hi\"'"

    def test_backslash_escape(self) -> None:
        # Cypher uses C-style backslash escapes. \n inside string → \n literal.
        assert cypher_quote("a\\b") == "'a\\\\b'"

    def test_dollar_sign_is_safe_inside_quotes(self) -> None:
        # $ inside a single-quoted Cypher string is literal (Cypher only treats
        # $ specially for $param placeholders in unquoted contexts). We use a
        # $-quoted block $$ ... $$ so this matters: the inner single quotes
        # are escaped, but $ needs no escape inside a string literal.
        assert cypher_quote("cost: $5") == "'cost: $5'"

    def test_newline_is_literal(self) -> None:
        # We do not escape \n inside Cypher strings; AGE accepts literal newlines
        # in string literals (treated as whitespace per Cypher spec).
        assert cypher_quote("line1\nline2") == "'line1\nline2'"

    def test_unicode_passes_through(self) -> None:
        # Unicode must NOT be escaped — it's UTF-8 in the wire format.
        assert cypher_quote("café") == "'café'"

    def test_null_value_returns_null_keyword(self) -> None:
        # For Cypher property values, NULL must be the literal keyword, not 'NULL'.
        assert cypher_quote(None) == "NULL"

    def test_numeric_passes_through_as_number(self) -> None:
        # For property values, numbers should NOT be quoted — they're Cypher numerics.
        assert cypher_quote(42) == "42"
        assert cypher_quote(3.14) == "3.14"

    def test_bool_keyword(self) -> None:
        # Cypher bools are keywords, not strings.
        assert cypher_quote(True) == "true"
        assert cypher_quote(False) == "false"

    def test_list_serialized_as_array_literal(self) -> None:
        assert cypher_quote(["a", "b"]) == "['a', 'b']"

    def test_list_with_special_chars(self) -> None:
        # Each element gets escaped independently.
        assert cypher_quote(["it's", 'say "hi"']) == "['it''s', 'say \"hi\"']"


# ---------------------------------------------------------------------------
# parse_agtype — AGE wire-format decoder
# ---------------------------------------------------------------------------


class TestParseAgtype:
    def test_vertex(self) -> None:
        # AGE returns vertex as a JSON-like agtype string. We unwrap the {"id", "label",
        # "properties"} shape. The "id" is a hex-encoded int (Cypher spec) but AGE 1.7.0
        # returns it as a decimal integer in agtype JSON.
        raw = '{"id": 844424930131969, "label": "Memory", "properties": {"id": 1, "category": "fact"}}::vertex'
        v = parse_agtype(raw)
        assert isinstance(v, AgtypeVertex)
        assert v.label == "Memory"
        assert v.properties == {"id": 1, "category": "fact"}
        assert v.vertex_id == 844424930131969

    def test_edge(self) -> None:
        raw = '{"id": 1125899906842625, "label": "SHARES_TAG", "start_id": 844424930131969, "end_id": 844424930131970, "properties": {"tags": ["legacy", "target:memory"]}}::edge'
        e = parse_agtype(raw)
        assert isinstance(e, AgtypeEdge)
        assert e.label == "SHARES_TAG"
        assert e.properties == {"tags": ["legacy", "target:memory"]}
        assert e.start_id == 844424930131969
        assert e.end_id == 844424930131970

    def test_scalar_quoted_string(self) -> None:
        # Cypher returns string scalars as 'value' (single-quoted).
        assert parse_agtype("'hello'") == "hello"

    def test_scalar_quoted_string_with_escape(self) -> None:
        # Doubled '' inside is an escaped single quote.
        assert parse_agtype("'it''s'") == "it's"

    def test_scalar_int(self) -> None:
        assert parse_agtype("42") == 42

    def test_scalar_bool(self) -> None:
        assert parse_agtype("true") is True
        assert parse_agtype("false") is False

    def test_null(self) -> None:
        assert parse_agtype("NULL") is None

    def test_list(self) -> None:
        assert parse_agtype("[1, 2, 3]") == [1, 2, 3]

    def test_object(self) -> None:
        result = parse_agtype('{"a": 1, "b": "x"}')
        assert result == {"a": 1, "b": "x"}

    def test_object_with_trailing_agtype_suffix(self) -> None:
        # Sometimes the agtype ::TYPE suffix attaches to objects too.
        result = parse_agtype('{"a": 1}::vertex')
        assert result == {"a": 1}

    def test_nested_vertex_in_list(self) -> None:
        # Cypher MATCH...RETURN often returns a list of vertex agtype strings.
        # Each element has the ::vertex suffix.
        raw = '[{"id": 1, "label": "Memory", "properties": {"id": 1}}::vertex, {"id": 2, "label": "Wiki", "properties": {"id": 5}}::vertex]'
        result = parse_agtype(raw)
        assert result == [
            {"id": 1, "label": "Memory", "properties": {"id": 1}},
            {"id": 2, "label": "Wiki", "properties": {"id": 5}},
        ]


# ---------------------------------------------------------------------------
# plan_graph_rebuild — pure function from SQL rows → GraphRebuildPlan
# ---------------------------------------------------------------------------


class TestPlanGraphRebuild:
    """The plan function is a pure transformation. We feed it synthetic SQL
    result rows and verify it produces the right edges without ever touching a DB.
    """

    @staticmethod
    def _memory_row(id_: int, category: str, tags: list[str], content: str = "") -> dict[str, Any]:
        return {"id": id_, "category": category, "tags": tags, "content": content}

    def test_empty_input_yields_empty_plan(self) -> None:
        plan = plan_graph_rebuild(
            memories=[],
            wikis=[],
            skills=[],
            session_memories=[],
        )
        assert plan.vertices == []
        assert plan.edges == []
        assert plan.stats == {
            "memory_vertices": 0,
            "wiki_vertices": 0,
            "skill_vertices": 0,
            "shares_tag_edges": 0,
            "co_occurs_edges": 0,
            "references_wiki_edges": 0,
            "uses_skill_edges": 0,
        }

    def test_vertex_only_no_edges(self) -> None:
        # One memory, one wiki, one skill, all isolated → 3 vertices, 0 edges.
        plan = plan_graph_rebuild(
            memories=[self._memory_row(1, "fact", ["foo"])],
            wikis=[{"id": 1, "slug": "x", "title": "X"}],
            skills=[{"id": 1, "name": "ci"}],
            session_memories=[],
        )
        kinds = sorted(v.kind for v in plan.vertices)
        assert kinds == ["Memory", "Skill", "Wiki"]
        assert plan.edges == []

    def test_shares_tag_edge_requires_min_overlap(self) -> None:
        # Two memories sharing 1 tag → no edge (default min_shared_tags=2).
        plan = plan_graph_rebuild(
            memories=[
                self._memory_row(1, "fact", ["legacy"]),
                self._memory_row(2, "fact", ["legacy"]),
            ],
            wikis=[],
            skills=[],
            session_memories=[],
            min_shared_tags=2,
        )
        assert all(e.label != "SHARES_TAG" for e in plan.edges)

    def test_shares_tag_edge_with_overlap(self) -> None:
        # Two memories sharing 2 tags → 1 edge.
        plan = plan_graph_rebuild(
            memories=[
                self._memory_row(1, "fact", ["legacy", "target:memory"]),
                self._memory_row(2, "fact", ["legacy", "target:memory"]),
            ],
            wikis=[],
            skills=[],
            session_memories=[],
        )
        shares_tag_edges = [e for e in plan.edges if e.label == "SHARES_TAG"]
        assert len(shares_tag_edges) == 1
        edge = shares_tag_edges[0]
        assert edge.src_id == 1
        assert edge.dst_id == 2
        assert sorted(edge.properties["tags"]) == ["legacy", "target:memory"]

    def test_min_shared_tags_configurable(self) -> None:
        # Lowering threshold to 1 captures single-tag overlap.
        plan = plan_graph_rebuild(
            memories=[
                self._memory_row(1, "fact", ["x"]),
                self._memory_row(2, "fact", ["x"]),
            ],
            wikis=[],
            skills=[],
            session_memories=[],
            min_shared_tags=1,
        )
        assert any(e.label == "SHARES_TAG" for e in plan.edges)

    def test_shares_tag_dedupes_pair(self) -> None:
        # Even with three shared tags, memory pair (1,2) generates ONE edge, not three.
        plan = plan_graph_rebuild(
            memories=[
                self._memory_row(1, "fact", ["a", "b", "c"]),
                self._memory_row(2, "fact", ["a", "b", "c"]),
            ],
            wikis=[],
            skills=[],
            session_memories=[],
        )
        shares_tag_edges = [e for e in plan.edges if e.label == "SHARES_TAG"]
        assert len(shares_tag_edges) == 1
        # But the tags property has all three.
        assert sorted(shares_tag_edges[0].properties["tags"]) == ["a", "b", "c"]

    def test_co_occurs_edge(self) -> None:
        # Two memories in the same session → one CO_OCCURS edge.
        plan = plan_graph_rebuild(
            memories=[
                self._memory_row(1, "fact", []),
                self._memory_row(2, "fact", []),
            ],
            wikis=[],
            skills=[],
            session_memories=[
                {"memory_id": 1, "session_id": "sess-abc"},
                {"memory_id": 2, "session_id": "sess-abc"},
            ],
        )
        co_edges = [e for e in plan.edges if e.label == "CO_OCCURS"]
        assert len(co_edges) == 1
        assert co_edges[0].properties["session_id"] == "sess-abc"

    def test_co_occurs_different_sessions_no_edge(self) -> None:
        plan = plan_graph_rebuild(
            memories=[
                self._memory_row(1, "fact", []),
                self._memory_row(2, "fact", []),
            ],
            wikis=[],
            skills=[],
            session_memories=[
                {"memory_id": 1, "session_id": "s1"},
                {"memory_id": 2, "session_id": "s2"},
            ],
        )
        assert not any(e.label == "CO_OCCURS" for e in plan.edges)

    def test_references_wiki_literal_match(self) -> None:
        plan = plan_graph_rebuild(
            memories=[
                self._memory_row(1, "fact", [], content="See wiki page sv-ci-gotchas for context"),
            ],
            wikis=[{"id": 5, "slug": "sv-ci-gotchas", "title": "SV CI Gotchas"}],
            skills=[],
            session_memories=[],
        )
        ref_edges = [e for e in plan.edges if e.label == "REFERENCES_WIKI"]
        assert len(ref_edges) == 1
        assert ref_edges[0].src_id == 1
        assert ref_edges[0].dst_id == 5
        assert ref_edges[0].properties["via"] == "literal"

    def test_references_wiki_marker_match(self) -> None:
        plan = plan_graph_rebuild(
            memories=[
                self._memory_row(1, "fact", [], content="reference: wiki://sv-design-system"),
            ],
            wikis=[{"id": 18, "slug": "sv-design-system", "title": "SV DS"}],
            skills=[],
            session_memories=[],
        )
        ref_edges = [e for e in plan.edges if e.label == "REFERENCES_WIKI"]
        assert len(ref_edges) == 1
        assert ref_edges[0].properties["via"] == "marker"

    def test_references_wiki_bracket_marker(self) -> None:
        plan = plan_graph_rebuild(
            memories=[
                self._memory_row(1, "fact", [], content="look at [[infra-network-topology]]"),
            ],
            wikis=[{"id": 19, "slug": "infra-network-topology", "title": "Infra"}],
            skills=[],
            session_memories=[],
        )
        ref_edges = [e for e in plan.edges if e.label == "REFERENCES_WIKI"]
        assert len(ref_edges) == 1

    def test_references_wiki_dedupes_pair(self) -> None:
        # Memory references same wiki twice (literal + marker) → still one edge,
        # but via="both".
        plan = plan_graph_rebuild(
            memories=[
                self._memory_row(
                    1,
                    "fact",
                    [],
                    content="sv-ci-gotchas and wiki://sv-ci-gotchas",
                ),
            ],
            wikis=[{"id": 5, "slug": "sv-ci-gotchas", "title": "SV CI"}],
            skills=[],
            session_memories=[],
        )
        ref_edges = [e for e in plan.edges if e.label == "REFERENCES_WIKI"]
        assert len(ref_edges) == 1
        assert ref_edges[0].properties["via"] == "both"

    def test_no_references_wiki_when_no_overlap(self) -> None:
        plan = plan_graph_rebuild(
            memories=[
                self._memory_row(1, "fact", [], content="totally unrelated content"),
            ],
            wikis=[{"id": 5, "slug": "sv-ci-gotchas", "title": "SV CI"}],
            skills=[],
            session_memories=[],
        )
        assert not any(e.label == "REFERENCES_WIKI" for e in plan.edges)

    def test_uses_skill_from_tags(self) -> None:
        plan = plan_graph_rebuild(
            memories=[
                self._memory_row(1, "fact", ["ci-binary-smoke", "github-actions"], content="nope"),
            ],
            wikis=[],
            skills=[{"id": 1, "name": "ci-binary-smoke"}],
            session_memories=[],
        )
        skill_edges = [e for e in plan.edges if e.label == "USES_SKILL"]
        assert len(skill_edges) == 1
        assert skill_edges[0].dst_id == 1
        assert skill_edges[0].properties["via"] == "tag"

    def test_uses_skill_from_body(self) -> None:
        plan = plan_graph_rebuild(
            memories=[
                self._memory_row(1, "fact", [], content="uses ci-binary-smoke for smoke tests"),
            ],
            wikis=[],
            skills=[{"id": 1, "name": "ci-binary-smoke"}],
            session_memories=[],
        )
        skill_edges = [e for e in plan.edges if e.label == "USES_SKILL"]
        assert len(skill_edges) == 1
        assert skill_edges[0].properties["via"] == "body"

    def test_uses_skill_marker_only(self) -> None:
        # Body must mention the skill ONLY via '#name' marker, never as a bare
        # word — so only the marker regex fires. Use a phrase where '#name'
        # appears but 'name' doesn't appear bare elsewhere.
        plan = plan_graph_rebuild(
            memories=[
                self._memory_row(1, "fact", [], content="see #pixu-tool for reference"),
            ],
            wikis=[],
            skills=[{"id": 1, "name": "pixu-tool"}],
            session_memories=[],
        )
        skill_edges = [e for e in plan.edges if e.label == "USES_SKILL"]
        assert len(skill_edges) == 1
        assert skill_edges[0].properties["via"] == "marker"

    def test_uses_skill_body_only(self) -> None:
        # The skill name appears as a bare word but NOT as a # marker.
        plan = plan_graph_rebuild(
            memories=[
                self._memory_row(1, "fact", [], content="uses ci-binary-smoke for smoke tests"),
            ],
            wikis=[],
            skills=[{"id": 1, "name": "ci-binary-smoke"}],
            session_memories=[],
        )
        skill_edges = [e for e in plan.edges if e.label == "USES_SKILL"]
        assert len(skill_edges) == 1
        assert skill_edges[0].properties["via"] == "body"

    def test_uses_skill_dedupes(self) -> None:
        plan = plan_graph_rebuild(
            memories=[
                self._memory_row(
                    1,
                    "fact",
                    ["ci-binary-smoke"],
                    content="uses ci-binary-smoke #ci-binary-smoke",
                ),
            ],
            wikis=[],
            skills=[{"id": 1, "name": "ci-binary-smoke"}],
            session_memories=[],
        )
        skill_edges = [e for e in plan.edges if e.label == "USES_SKILL"]
        assert len(skill_edges) == 1
        assert skill_edges[0].properties["via"] == "both"

    def test_skill_name_false_positive_guard(self) -> None:
        # Substring match must be word-bounded so 'ci' doesn't match 'science'.
        plan = plan_graph_rebuild(
            memories=[
                self._memory_row(1, "fact", [], content="computer science is great"),
            ],
            wikis=[],
            skills=[{"id": 1, "name": "ci"}],
            session_memories=[],
        )
        assert not any(e.label == "USES_SKILL" for e in plan.edges)

    def test_stats_counts_match_edges(self) -> None:
        plan = plan_graph_rebuild(
            memories=[
                self._memory_row(1, "fact", ["x", "y"]),
                self._memory_row(2, "fact", ["x", "y"]),
                self._memory_row(3, "fact", ["z"], content="see [[wiki-x]]"),
            ],
            wikis=[{"id": 1, "slug": "wiki-x", "title": "Wiki X"}],
            skills=[],
            session_memories=[
                {"memory_id": 1, "session_id": "s1"},
                {"memory_id": 2, "session_id": "s1"},
            ],
        )
        assert plan.stats["memory_vertices"] == 3
        assert plan.stats["wiki_vertices"] == 1
        assert plan.stats["shares_tag_edges"] == 1  # (1,2) share 2 tags
        assert plan.stats["co_occurs_edges"] == 1  # (1,2) in s1
        assert plan.stats["references_wiki_edges"] == 1  # (3) refs wiki-x

    def test_plan_is_idempotent_under_repeated_input(self) -> None:
        # Running plan_graph_rebuild twice on the same input produces identical plans.
        kwargs: dict[str, Any] = {
            "memories": [
                self._memory_row(1, "fact", ["a", "b"]),
                self._memory_row(2, "fact", ["a", "b"]),
            ],
            "wikis": [],
            "skills": [],
            "session_memories": [
                {"memory_id": 1, "session_id": "s"},
                {"memory_id": 2, "session_id": "s"},
            ],
        }
        plan1 = plan_graph_rebuild(**kwargs)
        plan2 = plan_graph_rebuild(**kwargs)
        assert len(plan1.edges) == len(plan2.edges)
        assert plan1.stats == plan2.stats

    def test_skips_deleted_memories(self) -> None:
        # Deleted memories should not appear as vertices or in any edges.
        plan = plan_graph_rebuild(
            memories=[
                {"id": 1, "category": "fact", "tags": ["x"], "content": "", "deleted": True},
            ],
            wikis=[],
            skills=[],
            session_memories=[],
        )
        assert plan.vertices == []
        assert plan.edges == []
