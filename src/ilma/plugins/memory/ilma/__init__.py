"""Ilma memory provider plugin shim for Hermes.

Re-exports IlmaMemoryProvider for the filesystem-based plugin loader
that scans ~/.hermes/plugins/memory/<name>/__init__.py at gateway boot.

The same class is also discoverable via the [project.entry-points.
"hermes_agent.plugins"] entry in pyproject.toml — both paths work.
"""

from ilma.adapters.hermes.memory_provider import IlmaMemoryProvider

__all__ = ["IlmaMemoryProvider"]
