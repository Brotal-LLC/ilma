"""Storage backend abstract base class.

Borrowed from MemPalace's backend plugin contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class StorageBackend(ABC):
    """Abstract storage backend."""

    name: str = ""

    @abstractmethod
    def add(
        self,
        collection: str,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add documents to a collection."""
        ...

    @abstractmethod
    def upsert(
        self,
        collection: str,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Upsert documents."""
        ...

    @abstractmethod
    def query(
        self,
        collection: str,
        query_embeddings: list[list[float]],
        *,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Vector query."""
        ...

    @abstractmethod
    def get(
        self, collection: str, ids: list[str] | None = None, *, where: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Fetch by id or filter."""
        ...

    @abstractmethod
    def delete(
        self, collection: str, ids: list[str] | None = None, *, where: dict[str, Any] | None = None
    ) -> None:
        """Delete documents."""
        ...

    @abstractmethod
    def count(self, collection: str) -> int:
        """Count documents in collection."""
        ...

    @abstractmethod
    def health(self) -> dict[str, Any]:
        """Backend health status."""
        ...
