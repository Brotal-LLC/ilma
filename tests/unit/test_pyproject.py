"""Packaging metadata tests for ilma."""

from __future__ import annotations

import inspect
import tomllib
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
PLUGIN_SHIM = PROJECT_ROOT / "src" / "ilma" / "plugins" / "memory" / "ilma" / "__init__.py"
PLUGIN_YAML = PROJECT_ROOT / "src" / "ilma" / "plugins" / "memory" / "ilma" / "plugin.yaml"
EXPECTED_PROVIDER = "ilma.adapters.hermes.memory_provider:IlmaMemoryProvider"


def _pyproject() -> dict[str, object]:
    with PYPROJECT.open("rb") as file:
        return tomllib.load(file)


def test_pyproject_registers_ilma_memory_provider_entry_point() -> None:
    pyproject = _pyproject()
    entry_points = pyproject["project"]["entry-points"]["hermes_agent.plugins"]  # type: ignore[index]

    matches = {
        name: value
        for name, value in entry_points.items()
        if "ilma" in name.lower() and value == EXPECTED_PROVIDER
    }

    assert matches == {"ilma_memory": EXPECTED_PROVIDER}


def test_plugin_shim_file_exists() -> None:
    assert PLUGIN_SHIM.is_file()


def test_plugin_shim_reexports_ilma_memory_provider() -> None:
    import ilma.plugins.memory.ilma as plugin

    assert "IlmaMemoryProvider" in plugin.__all__
    assert plugin.IlmaMemoryProvider.__name__ == "IlmaMemoryProvider"


def test_plugin_yaml_contains_expected_metadata() -> None:
    metadata = yaml.safe_load(PLUGIN_YAML.read_text())

    assert metadata == {
        "name": "ilma",
        "version": "0.2.0",
        "kind": "exclusive",
        "description": "Postgres-backed vector memory provider",
    }


def test_importing_plugin_shim_does_not_raise() -> None:
    __import__("ilma.plugins.memory.ilma")


def test_ilma_memory_provider_is_a_class() -> None:
    import ilma.plugins.memory.ilma as plugin

    assert inspect.isclass(plugin.IlmaMemoryProvider)
