"""Tests for ilma.config.IlmaConfig.

The config dataclass resolves values in priority order: dataclass defaults,
YAML overlays, environment variables.  These tests exercise the full
resolution pipeline plus the source-tracking helpers.
"""

from __future__ import annotations

import os
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from ilma.config import (
    ApiConfig,
    IlmaConfig,
    MemoryConfig,
    PostgresConfig,
    VectorsConfig,
)


@pytest.fixture(autouse=True)
def _clean_ilma_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every key IlmaConfig reads from env so each test is isolated.

    IlmaConfig resolves ``postgres.dsn`` from ``ILMA_DSN``, ``PG_MEM_DB_CONN_STR``,
    and ``HERMES_PG_CONN_STR`` (in addition to the canonical ``ILMA_PG_DSN``).
    The fixture must clear all of them or the test inherits the operator's local
    DSN via whichever alias happens to be set in their shell.
    """

    aliases = (
        "ILMA_DSN",
        "PG_MEM_DB_CONN_STR",
        "HERMES_PG_CONN_STR",
        "MEMORY_PROVIDER",
        "HERMES_HOME",
    )
    for key in list(os.environ):
        if key.startswith("ILMA_") or key in aliases:
            monkeypatch.delenv(key, raising=False)


def test_defaults_when_no_yaml_no_env() -> None:
    config = IlmaConfig.from_env()

    assert config.postgres == PostgresConfig()
    assert config.memory == MemoryConfig()
    assert config.vectors == VectorsConfig()
    assert config.api == ApiConfig()


def test_yaml_overlay(tmp_path: Path) -> None:
    yaml_path = tmp_path / "ilma.yaml"
    yaml_path.write_text(
        "postgres:\n  dsn: postgres://yaml.example/db\n  pool_size: 9\n"
        "memory:\n  namespace: from-yaml\n",
        encoding="utf-8",
    )

    config = IlmaConfig.from_env(yaml_path=yaml_path)

    assert config.postgres.dsn == "postgres://yaml.example/db"
    assert config.postgres.pool_size == 9
    assert config.memory.namespace == "from-yaml"
    assert config.source_of("postgres.dsn") == "yaml"
    assert config.source_of("postgres.pool_size") == "yaml"
    assert config.source_of("memory.namespace") == "yaml"


def test_env_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    yaml_path = tmp_path / "ilma.yaml"
    yaml_path.write_text(
        "postgres:\n  dsn: postgres://yaml.example/db\n"
        "memory:\n  namespace: from-yaml\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ILMA_PG_DSN", "postgres://env.example/db")
    monkeypatch.setenv("ILMA_MEMORY_NAMESPACE", "from-env")

    config = IlmaConfig.from_env(yaml_path=yaml_path)

    assert config.postgres.dsn == "postgres://env.example/db"
    assert config.memory.namespace == "from-env"
    assert config.source_of("postgres.dsn") == "env"
    assert config.source_of("memory.namespace") == "env"


def test_nested_lookup() -> None:
    config = IlmaConfig.from_env()

    assert isinstance(config.postgres, PostgresConfig)
    assert isinstance(config.memory, MemoryConfig)
    assert isinstance(config.vectors, VectorsConfig)
    assert isinstance(config.api, ApiConfig)
    assert config.vectors.embedder == "openai"
    assert config.api.rate_limit_rps == 30.0


def test_source_of_default(monkeypatch) -> None:
    import os
    print("DEBUG: env HERMES_PG_CONN_STR:", os.environ.get("HERMES_PG_CONN_STR"))
    config = IlmaConfig.from_env()
    print("DEBUG: config.postgres.dsn:", repr(config.postgres.dsn))

    assert config.source_of("memory.namespace") == "default"
    assert config.source_of("postgres.dsn") == "default"
    assert config.source_of("vectors.embedder") == "default"


def test_source_of_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "ilma.yaml"
    yaml_path.write_text("postgres:\n  dsn: postgres://yaml/db\n", encoding="utf-8")

    config = IlmaConfig.from_env(yaml_path=yaml_path)

    assert config.source_of("postgres.dsn") == "yaml"
    # Untouched values stay default.
    assert config.source_of("memory.namespace") == "default"


def test_source_of_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ILMA_PG_DSN", "postgres://env/db")
    monkeypatch.setenv("ILMA_VECTORS_DIM", "768")

    config = IlmaConfig.from_env()

    assert config.source_of("postgres.dsn") == "env"
    assert config.source_of("vectors.dim") == "env"


def test_to_dict_roundtrip() -> None:
    original = IlmaConfig(
        postgres=PostgresConfig(dsn="postgres://x", pool_size=11),
        memory=MemoryConfig(namespace="ns", provider="ilma"),
        vectors=VectorsConfig(embedder="openai", dim=1536),
        api=ApiConfig(api_key="secret", rate_limit_rps=5.0),
    )

    rebuilt = IlmaConfig.from_dict(original.to_dict())

    assert rebuilt == original


def test_env_prefix_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ILMA_MEMORY_NAMESPACE", "n1")
    monkeypatch.setenv("ILMA_VECTORS_EMBEDDER", "custom-embedder")
    monkeypatch.setenv("ILMA_VECTORS_DIM", "768")

    config = IlmaConfig.from_env()

    assert config.memory.namespace == "n1"
    assert config.vectors.embedder == "custom-embedder"
    assert config.vectors.dim == 768
    # Untouched values still default.
    assert config.postgres.dsn == ""
    assert config.api.rate_limit_rps == 30.0


def test_frozen_dataclass() -> None:
    config = IlmaConfig.from_env()

    with pytest.raises(FrozenInstanceError):
        config.postgres.dsn = "x"  # type: ignore[misc]


def test_env_alias_first_match_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the operator sets both a canonical env key AND a legacy alias, the
    canonical one must win (declaration order in ``_ENV_TO_DOTTED``).

    This regression test guards against the bug where iteration over the
    alias dict caused the LATER-iterated alias (``PG_MEM_DB_CONN_STR``) to
    silently override the user's intent (``ILMA_DSN`` or ``ILMA_PG_DSN``).
    """

    monkeypatch.setenv("ILMA_PG_DSN", "postgres://canonical/db")
    monkeypatch.setenv("ILMA_DSN", "postgres://short-alias/db")
    monkeypatch.setenv("PG_MEM_DB_CONN_STR", "postgres://legacy-alias/db")
    monkeypatch.setenv("HERMES_PG_CONN_STR", "postgres://hermes-legacy/db")

    config = IlmaConfig.from_env()

    # Canonical name wins because it's declared first in ``_ENV_TO_DOTTED``.
    assert config.postgres.dsn == "postgres://canonical/db"


def test_env_alias_short_form_beats_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the canonical name, the short alias ``ILMA_DSN`` must win
    over the older ``PG_MEM_DB_CONN_STR`` / ``HERMES_PG_CONN_STR`` aliases.
    """

    monkeypatch.setenv("ILMA_DSN", "postgres://short-alias/db")
    monkeypatch.setenv("PG_MEM_DB_CONN_STR", "postgres://legacy-alias/db")
    monkeypatch.setenv("HERMES_PG_CONN_STR", "postgres://hermes-legacy/db")

    config = IlmaConfig.from_env()

    assert config.postgres.dsn == "postgres://short-alias/db"


def test_backward_compat_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    """HERMES_PG_CONN_STR and MEMORY_PROVIDER should still work as legacy aliases."""

    monkeypatch.setenv("HERMES_PG_CONN_STR", "postgres://legacy/db")
    monkeypatch.setenv("MEMORY_PROVIDER", "ilma")

    config = IlmaConfig.from_env()

    assert config.postgres.dsn == "postgres://legacy/db"
    assert config.memory.provider == "ilma"


def test_ilma_dsn_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ILMA_DSN", "postgres://alias/db")

    config = IlmaConfig.from_env()

    assert config.postgres.dsn == "postgres://alias/db"


def test_env_value_coercion_for_int(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ILMA_PG_POOL_SIZE", "12")
    monkeypatch.setenv("ILMA_RATE_LIMIT_RPS", "12.5")

    config = IlmaConfig.from_env()

    assert config.postgres.pool_size == 12
    assert config.api.rate_limit_rps == 12.5


def test_env_value_coercion_for_bool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ILMA_MEMORY_AUTO_BACKUP", "false")

    config = IlmaConfig.from_env()

    assert config.memory.auto_backup is False


def test_unknown_dotted_key_raises() -> None:
    config = IlmaConfig.from_env()

    with pytest.raises(KeyError):
        config.source_of("nope.invalid")


def test_replace_returns_new_instance() -> None:
    base = IlmaConfig.from_env()
    updated = base.replace("memory.namespace", "replaced", source="yaml")

    assert base.memory.namespace == "default"  # original unchanged
    assert updated.memory.namespace == "replaced"
    assert updated.source_of("memory.namespace") == "yaml"


def test_hermes_home_drives_yaml_search_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hermes_home = tmp_path / "hermes"
    (hermes_home / ".ilma").mkdir(parents=True)
    (hermes_home / ".ilma" / "config.yaml").write_text(
        "postgres:\n  dsn: postgres://hermes-home/db\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    config = IlmaConfig.from_env()

    assert config.postgres.dsn == "postgres://hermes-home/db"
    assert config.source_of("postgres.dsn") == "yaml"
