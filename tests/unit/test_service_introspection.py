from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from ilma.service import (
    WRITE_TOOLS,
    IlmaService,
    _derive_write_tools,
    method_description,
    method_to_pydantic_model,
    tools_dict,
)


def test_method_to_pydantic_model_builds_fields_from_signature() -> None:
    def sample_tool(query: str, top_k: int = 10) -> dict[str, Any]:
        """Search memory.

        Args:
            query: Text to search for.
            top_k: Maximum number of results.
        """
        return {"query": query, "top_k": top_k}

    model = method_to_pydantic_model(sample_tool)

    args = model(query="hello")
    assert args.model_dump() == {"query": "hello", "top_k": 10}
    assert model.model_fields["query"].description == "Text to search for."
    assert model.model_fields["top_k"].description == "Maximum number of results."
    with pytest.raises(ValidationError):
        model()


def test_method_to_pydantic_model_supports_optional_and_list_args() -> None:
    def sample_tool(content: str, tags: list[str] | None = None) -> dict[str, Any]:
        """Store content.

        Args:
            content: Content to store.
            tags: Optional tags to attach.
        """
        return {"content": content, "tags": tags}

    model = method_to_pydantic_model(sample_tool)

    assert model(content="note").model_dump() == {"content": "note", "tags": None}
    assert model(content="note", tags=["a", "b"]).model_dump() == {
        "content": "note",
        "tags": ["a", "b"],
    }
    with pytest.raises(ValidationError):
        model(content="note", tags="not-a-list")


def test_method_to_pydantic_model_parses_sphinx_param_docstrings() -> None:
    def sample_tool(name: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Fetch a stored object.

        :param name: Object name to fetch.
        :param metadata: Optional metadata filter.
        """
        return {"name": name, "metadata": metadata}

    model = method_to_pydantic_model(sample_tool)

    assert model(name="python").model_dump() == {"name": "python", "metadata": None}
    assert model.model_fields["name"].description == "Object name to fetch."
    assert model.model_fields["metadata"].description == "Optional metadata filter."


def test_tools_dict_registers_one_entry_per_public_ilma_method() -> None:
    service = MiniService()

    tools = tools_dict(service)

    assert set(tools) == {"ilma_echo", "ilma_store"}
    assert tools["ilma_echo"]["description"] == "Echo a message."
    assert set(tools["ilma_store"]["input_model"].model_fields) == {"content", "tags"}


def test_tools_dict_handler_dispatches_to_method() -> None:
    """Handler validates args and calls the bound method directly.

    Audit + metrics wrapping is the service method's responsibility
    (e.g. IlmaService.ilma_remember calls self.call internally), not
    the handler's. The MiniService test fixture does not wrap
    ilma_store, so we assert on the result + a side-effect counter
    instead of service.calls.
    """
    service = MiniService()
    tool = tools_dict(service)["ilma_store"]

    result = tool["handler"](content="note", tags=["a"])

    assert result == {"ok": True, "content": "note", "tags": ["a"]}
    assert service.store_calls == [{"content": "note", "tags": ["a"]}]


def test_tools_dict_handler_returns_validation_error_shape_for_bad_args() -> None:
    service = MiniService()
    tool = tools_dict(service)["ilma_store"]

    result = tool["handler"](tags="not-a-list")

    assert result["error"] == "validation_error"
    assert {detail["loc"][0] for detail in result["details"]} == {"content", "tags"}


def test_method_description_uses_first_docstring_line_or_method_name() -> None:
    def documented() -> None:
        """First line.

        More detail.
        """

    def undocumented() -> None:
        pass

    assert method_description(documented) == "First line."
    assert method_description(undocumented) == "ilma: undocumented"


def test_write_tools_set_unchanged() -> None:
    previous_hand_maintained_mapping = {
        "ilma_remember": ("memory", "remember"),
        "ilma_forget": ("memory", "forget"),
        "ilma_wiki_create": ("wiki", "create"),
        "ilma_wiki_update": ("wiki", "update"),
        "ilma_kanban_create": ("kanban", "create"),
        "ilma_kanban_update": ("kanban", "update"),
        "ilma_kanban_complete": ("kanban", "complete"),
        "ilma_metrics_record": ("metrics", "record"),
        "ilma_obs_log": ("observability", "log"),
        "ilma_migrate": ("maintenance", "migrate"),
        "ilma_repair": ("maintenance", "repair"),
        "ilma_graph_rebuild": ("graph", "rebuild"),
    }

    assert dict(WRITE_TOOLS) == previous_hand_maintained_mapping


def test_write_tools_derive_from_class() -> None:
    assert _derive_write_tools(IlmaService) == dict(WRITE_TOOLS)


def test_adding_a_new_write_method_updates_dict_without_manual_edit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def ilma_dummy_write(self: Any) -> dict[str, Any]:
        return self.call("ilma_dummy_write", lambda: {"ok": True}, {})

    monkeypatch.setattr(IlmaService, "ilma_dummy_write", ilma_dummy_write, raising=False)

    derived = _derive_write_tools()

    assert derived["ilma_dummy_write"] == ("dummy", "write")
    assert "ilma_dummy_write" not in WRITE_TOOLS


class MiniService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.store_calls: list[dict[str, Any]] = []

    def call(self, tool_name: str, fn: Any, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"tool_name": tool_name, "payload": payload})
        return fn()

    def ilma_echo(self, message: str) -> dict[str, Any]:
        """Echo a message.

        Args:
            message: Message to echo.
        """
        return {"ok": True, "message": message}

    def ilma_store(self, content: str, tags: list[str] | None = None) -> dict[str, Any]:
        """Store content.

        Args:
            content: Content to store.
            tags: Tags to attach.
        """
        self.store_calls.append({"content": content, "tags": tags})
        return {"ok": True, "content": content, "tags": tags}

    def not_a_tool(self) -> dict[str, Any]:
        return {"ok": False}
