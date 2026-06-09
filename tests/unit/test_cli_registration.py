from __future__ import annotations

from typing import Any

from ilma.api import cli
from ilma.service import tools_dict


class RegistrationService:
    def ilma_status(self) -> dict[str, Any]:
        return {"ok": True}

    def ilma_search(
        self, query: str, top_k: int = 10, hybrid_text_weight: float = 0.5
    ) -> dict[str, Any]:
        return {"ok": True, "query": query, "top_k": top_k, "hybrid_text_weight": hybrid_text_weight}

    def ilma_remember(
        self,
        content: str,
        tags: list[str] | None = None,
        category: str | None = None,
        source: str | None = "mcp",
    ) -> dict[str, Any]:
        return {"ok": True, "content": content, "tags": tags, "category": category, "source": source}

    def ilma_forget(self, memory_id: int) -> dict[str, Any]:
        return {"ok": True, "memory_id": memory_id}

    def ilma_doctor(self) -> dict[str, Any]:
        return {"ok": True}

    def ilma_repair(self, force: bool = False) -> dict[str, Any]:
        return {"ok": True, "force": force}

    def ilma_migrate(self) -> dict[str, Any]:
        return {"ok": True, "migrated": True}


def _registered_callbacks() -> dict[str, Any]:
    callbacks: dict[str, Any] = {}
    for command in cli.app.registered_commands:
        assert command.callback is not None
        name = command.name or command.callback.__name__.replace("_", "-")
        callbacks[name] = command.callback
    return callbacks


def test_tools_dict_cli_equivalents_are_registered() -> None:
    registered_names = set(_registered_callbacks())

    for tool_name in tools_dict(RegistrationService()):
        command_name = cli._CLI_TOOL_TO_COMMAND.get(tool_name)
        if command_name is None:
            continue
        assert command_name in registered_names, f"{tool_name} is missing CLI command {command_name}"


def test_cli_excluded_contains_only_four_hand_written_commands() -> None:
    assert frozenset({"init", "mcp", "serve", "migrate-config"}) == cli._CLI_EXCLUDED


def test_only_four_typer_commands_are_hand_written() -> None:
    callbacks = _registered_callbacks()
    hand_written = {
        name
        for name, callback in callbacks.items()
        if not getattr(callback, "__ilma_auto_registered__", False)
    }

    assert hand_written == {"init", "mcp", "serve", "migrate-config"}


def test_expected_service_commands_are_auto_registered() -> None:
    callbacks = _registered_callbacks()
    auto_registered = {
        name
        for name, callback in callbacks.items()
        if getattr(callback, "__ilma_auto_registered__", False)
    }

    assert auto_registered == {
        "status",
        "search",
        "remember",
        "forget",
        "doctor",
        "repair",
        "audit",
        "migrate",
    }
