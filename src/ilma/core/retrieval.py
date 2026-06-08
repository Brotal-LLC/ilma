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


def _sanitize_query(query: str) -> str:
    """Strip prompt contamination from search queries.

    Borrowed from MemPalace: contaminated queries can drop R@10
    from ~89.8% to ~1%.
    """
    # Strip system/developer/tool prefixes
    query = re.sub(r"^(system|developer|user|assistant|tool)\s*[:\-]\s*", "", query, flags=re.I)
    # Strip XML-like tags
    query = re.sub(r"<[^>]+>", "", query)
    # Collapse whitespace
    query = re.sub(r"\s+", " ", query).strip()
    return query


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
