"""Wiki repository interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol


@dataclass
class WikiDoc:
    """A wiki document — long-form structured knowledge."""

    id: int
    slug: str
    title: str
    body_md: str
    category: str | None = None
    tags: tuple[str, ...] = ()
    source_uri: str | None = None
    version: int = 1
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class WikiRepo(Protocol):
    """Abstract wiki repository."""

    def ingest(
        self,
        slug: str,
        title: str,
        body_md: str,
        *,
        category: str | None = None,
        tags: list[str] | None = None,
        source_uri: str | None = None,
    ) -> dict[str, Any]:
        """Store or update a wiki doc. Returns {document_id, version_id, chunks}."""
        ...

    def search(self, query: str, *, top_k: int = 5) -> list[dict[str, Any]]:
        """FTS + vector search over wiki chunks."""
        ...

    def get(self, slug: str) -> WikiDoc | None:
        """Fetch a doc by slug."""
        ...

    def suggest_links(self, doc_id: int, *, top_k: int = 5) -> list[dict[str, Any]]:
        """Suggest related docs via vector similarity."""
        ...
