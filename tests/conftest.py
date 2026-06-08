"""Shared test fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_memory():
    from ilma.core.memory import Memory

    return Memory(
        id=1,
        content="User prefers dark mode",
        tags=("user", "preference"),
        category="identity",
    )
