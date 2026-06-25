"""ilma — Framework-agnostic agent memory system.

Postgres + pgvector backend. MCP-native. Hermes Agent, Claude, Cursor,
Codex — any MCP client.
"""

from ilma.service import method_description, method_to_pydantic_model, tools_dict

__version__ = "0.2.3"

__all__ = ["__version__", "method_description", "method_to_pydantic_model", "tools_dict"]
