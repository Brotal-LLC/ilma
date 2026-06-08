"""Retrieval and injection layer.

Builds the system-prompt MEMORY block from live memories.
Borrowed from hermes-memory v2, enhanced with priority ranking
and query sanitization.
"""

from __future__ import annotations

import re

from ilma.core.memory import Memory, MemoryRepo

#: Prompt-injection query for identity/preferences.
_DEFAULT_INJECTION_QUERY = "user identity preferences project context"

#: Max chars per bullet before truncation.
_BULLET_MAX_CHARS = 140

#: Priority boost for identity-related memories.
_IDENTITY_BOOST = 2.0

#: Priority boost for user/preference tags.
_PREFERENCE_BOOST = 1.5

#: Hard cap for search queries.  Postgres FTS and embedders both become wasteful
#: on prompt-sized strings; retrieval only needs a compact intent.
_MAX_QUERY_CHARS = 10_000

_PROMPT_INJECTION_PATTERNS = (
    r"\bignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|messages?|rules?|context)\b",
    r"\bdisregard\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|messages?|rules?|context)\b",
    r"\bforget\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|messages?|rules?|context)\b",
    r"\boverride\s+(?:the\s+)?(?:system|developer|safety)\s+(?:prompt|instructions?|rules?)\b",
    r"\breveal\s+(?:the\s+)?(?:system|developer)\s+(?:prompt|instructions?|message)\b",
    r"\byou\s+are\s+now\b",
    r"\bact\s+as\b",
    r"\bjailbreak\b",
)

_SQL_INJECTION_PATTERNS = (
    r"--.*$",
    r"/\*.*?\*/",
    r"\b(?:or|and)\s+\d+\s*=\s*\d+\b",
    r"\bunion\s+(?:all\s+)?select\b",
    r"\b(?:drop|truncate|alter)\s+(?:table|schema|database)\s+[\w.\"']+",
    r"\bdelete\s+from\s+[\w.\"']+",
    r"\binsert\s+into\s+[\w.\"']+",
    r"\bupdate\s+[\w.\"']+\s+set\b",
)


def _sanitize_query(query: str) -> str:
    """Strip prompt/SQL contamination from search queries.

    Borrowed from MemPalace: contaminated queries can drop R@10
    from ~89.8% to ~1%.  The sanitizer is intentionally conservative: it
    removes common instruction-hijacking phrases, SQL-injection operators, and
    prompt/XML framing while preserving the user's remaining retrieval intent.
    """
    query = str(query or "")[:_MAX_QUERY_CHARS]
    # Strip system/developer/tool prefixes anywhere they begin a line.
    query = re.sub(
        r"(?im)^\s*(system|developer|user|assistant|tool)\s*[:\-]\s*",
        "",
        query,
    )
    # Strip XML-like tags, including <system> / </system> prompt wrappers.
    query = re.sub(r"<[^>]+>", " ", query)
    for pattern in _PROMPT_INJECTION_PATTERNS:
        query = re.sub(pattern, " ", query, flags=re.I)
    for pattern in _SQL_INJECTION_PATTERNS:
        query = re.sub(pattern, " ", query, flags=re.I | re.S | re.M)
    # Semicolons are useful SQL statement separators and rarely useful search terms.
    query = query.replace(";", " ")
    # Collapse whitespace
    query = re.sub(r"\s+", " ", query).strip()
    return query[:_MAX_QUERY_CHARS]


def _score_memory(memory: Memory) -> float:
    """Priority score for injection ordering.

    Higher = more important for system prompt.
    """
    score = 0.0
    tags = {t.lower() for t in memory.tags}
    category = (memory.category or "").lower()

    if category == "identity" or "identity" in tags:
        score += _IDENTITY_BOOST
    if "user" in tags or "preference" in tags:
        score += _PREFERENCE_BOOST
    if "project" in tags:
        score += 1.0
    # Recency bonus could go here
    return score


def _format_bullet(memory: Memory) -> str:
    """Format a memory as a compact bullet."""
    content = memory.content.strip().replace("\n", " ")
    if len(content) > _BULLET_MAX_CHARS:
        content = content[: _BULLET_MAX_CHARS - 1] + "…"
    tag_str = ""
    if memory.tags:
        tag_str = f" [{' '.join(memory.tags)}]"
    return f"•{tag_str} {content}"


class InjectionLayer:
    """Builds the system-prompt MEMORY block from live memories."""

    def __init__(
        self,
        *,
        injection_query: str = _DEFAULT_INJECTION_QUERY,
        top_k: int = 15,
        char_limit: int = 2200,
    ) -> None:
        self.injection_query = injection_query
        self.top_k = top_k
        self.char_limit = char_limit

    def render(self, repo: MemoryRepo | None) -> str:
        """Build the MEMORY block.

        Mirrors hermes-agent format:
            ════════════════════════════════════════════════
            MEMORY (your personal notes) [95% — 2,098/2,200 chars]
            ════════════════════════════════════════════════
            <bullets>
        """
        if repo is None:
            return "(no memory store wired; using built-in local store)"

        try:
            s = repo.status()
            live = s.get("live_memories", 0)
        except Exception:
            live = "?"

        header = (
            "═══════════════════════════════════════════════\n"
            "MEMORY (your personal notes)\n"
            "═══════════════════════════════════════════════\n"
        )
        footer = f"\n(live: {live})\n"
        budget = self.char_limit - len(header) - len(footer)

        if budget <= 0:
            return header + "(memory block budget exhausted)\n" + footer

        try:
            query = _sanitize_query(self.injection_query)
            memories = repo.search(query, top_k=self.top_k, hybrid_text_weight=0.3)
            # Re-rank by priority score
            memories = sorted(memories, key=lambda m: _score_memory(m), reverse=True)
        except Exception as e:
            return header + f"(memory retrieval failed: {e})\n" + footer

        lines: list[str] = []
        used = 0
        for memory in memories:
            if memory.deleted:
                continue
            line = _format_bullet(memory)
            if used + len(line) + 1 > budget:
                # Try to fit a truncation note
                note = "… (more memories available)"
                if used + len(note) + 1 <= budget:
                    lines.append(note)
                break
            lines.append(line)
            used += len(line) + 1

        body = "\n".join(lines) if lines else "(no memories yet)"
        block = header + body + footer
        # Update header with actual usage
        pct = int(len(block) / self.char_limit * 100)
        header = (
            "═══════════════════════════════════════════════\n"
            f"MEMORY (your personal notes) [{pct}% — {len(block)}/{self.char_limit} chars]\n"
            "═══════════════════════════════════════════════\n"
        )
        return header + body + footer


def build_memory_block(repo: MemoryRepo | None, *, char_limit: int = 2200) -> str:
    """Convenience wrapper — matches hermes-memory signature."""
    layer = InjectionLayer(char_limit=char_limit)
    return layer.render(repo)
