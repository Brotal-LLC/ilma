"""Sessions repository interface."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol


@dataclass
class SessionMessage:
    id: int
    session_id: str
    role: str  # user | assistant | tool
    content: str
    created_at: datetime


class SessionsRepo(Protocol):
    def append(self, session_id: str, role: str, content: str) -> int:
        """Append a message to a session. Returns message id."""
        ...

    def get_session(self, session_id: str, *, limit: int = 100) -> list[SessionMessage]:
        """Fetch session messages."""
        ...

    def search(self, query: str, *, top_k: int = 10) -> list[SessionMessage]:
        """Search across all sessions."""
        ...

    def recent_sessions(self, *, limit: int = 10) -> list[str]:
        """List recent session ids."""
        ...
