"""Unified runtime configuration for ilma.

Configuration is immutable and is resolved in priority order:

1. dataclass defaults
2. YAML overlays
3. environment variables

The public entry point is :meth:`IlmaConfig.from_env`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, get_args, get_origin, get_type_hints

import yaml

ConfigSource = Literal["default", "yaml", "env"]


@dataclass(frozen=True)
class PostgresConfig:
    """Postgres connection settings."""

    dsn: str = ""
    pool_size: int = 5
    pool_timeout_s: int = 10


@dataclass(frozen=True)
class MemoryConfig:
    """Memory runtime settings."""

    namespace: str = "default"
    backup_prefix: str = "_bak_"
    auto_backup: bool = True
    provider: str = "local"


@dataclass(frozen=True)
class VectorsConfig:
    """Embedding/vector runtime settings."""

    embedder: str = "openai"
    dim: int = 1536
    batch_size: int = 32
    openai_model: str = "text-embedding-3-small"
    base_url: str | None = None
    api_key: str | None = None


@dataclass(frozen=True)
class ApiConfig:
    """HTTP API runtime settings."""

    api_key: str | None = None
    rate_limit_rps: float = 30.0
    cors_origins: tuple[str, ...] = ("*",)


_ENV_TO_DOTTED: Mapping[str, str] = {
    # Canonical ilma names.
    "ILMA_PG_DSN": "postgres.dsn",
    "ILMA_PG_POOL_SIZE": "postgres.pool_size",
    "ILMA_PG_POOL_TIMEOUT_S": "postgres.pool_timeout_s",
    "ILMA_PG_POOL_TIMEOUT": "postgres.pool_timeout_s",
    "ILMA_MEMORY_NAMESPACE": "memory.namespace",
    "ILMA_MEMORY_BACKUP_PREFIX": "memory.backup_prefix",
    "ILMA_MEMORY_AUTO_BACKUP": "memory.auto_backup",
    "ILMA_MEMORY_PROVIDER": "memory.provider",
    "ILMA_VECTORS_EMBEDDER": "vectors.embedder",
    "ILMA_VECTORS_DIM": "vectors.dim",
    "ILMA_VECTORS_BATCH_SIZE": "vectors.batch_size",
    "ILMA_VECTORS_OPENAI_MODEL": "vectors.openai_model",
    "ILMA_VECTORS_BASE_URL": "vectors.base_url",
    "ILMA_VECTORS_API_KEY": "vectors.api_key",
    "ILMA_API_KEY": "api.api_key",
    "ILMA_RATE_LIMIT_RPS": "api.rate_limit_rps",
    "ILMA_CORS_ORIGINS": "api.cors_origins",
    # Backward-compatible aliases used by existing ilma/Hermes installs.
    "ILMA_DSN": "postgres.dsn",
    "PG_MEM_DB_CONN_STR": "postgres.dsn",
    "HERMES_PG_CONN_STR": "postgres.dsn",
    "ILMA_PG_POOL_MAX": "postgres.pool_size",
    "MEMORY_PROVIDER": "memory.provider",
    "ILMA_EMBED_PROVIDER": "vectors.embedder",
    "ILMA_EMBED_DIM": "vectors.dim",
    "ILMA_EMBED_MODEL": "vectors.openai_model",
    "ILMA_EMBED_BASE_URL": "vectors.base_url",
    "ILMA_EMBED_API_KEY": "vectors.api_key",
}


@dataclass(frozen=True)
class IlmaConfig:
    """Immutable ilma configuration with YAML/env overlays and source tracking."""

    postgres: PostgresConfig = field(default_factory=PostgresConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    vectors: VectorsConfig = field(default_factory=VectorsConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    _sources: Mapping[str, ConfigSource] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        sources = dict(self._sources) if self._sources else _default_sources(type(self))
        for dotted_key in _dotted_keys(type(self)):
            sources.setdefault(dotted_key, "default")
        object.__setattr__(self, "_sources", MappingProxyType(sources))

    @classmethod
    def from_env(
        cls,
        *,
        hermes_home: str | None = None,
        yaml_path: str | Path | None = None,
    ) -> IlmaConfig:
        """Resolve config from defaults, standard YAML locations, and process env.

        Standard YAML locations are ``~/.config/ilma/config.yaml`` with fallback to
        ``~/.ilma/config.yaml``. When ``HERMES_HOME`` (or ``hermes_home``) is set,
        ``$HERMES_HOME/.ilma/config.yaml`` and ``$HERMES_HOME/config.yaml`` are
        also loaded, with later files taking precedence. Environment variables are
        always applied last.
        """

        env = os.environ
        paths: list[Path] = []
        if yaml_path is not None:
            paths.append(Path(yaml_path).expanduser())
        else:
            xdg_path = Path("~/.config/ilma/config.yaml").expanduser()
            legacy_path = Path("~/.ilma/config.yaml").expanduser()
            paths.append(xdg_path if xdg_path.is_file() else legacy_path)

        resolved_hermes_home = hermes_home or env.get("HERMES_HOME", "").strip()
        if resolved_hermes_home:
            base = Path(resolved_hermes_home).expanduser()
            paths.extend([base / ".ilma" / "config.yaml", base / "config.yaml"])

        return cls._from_sources(env=env, yaml_paths=paths)

    @classmethod
    def from_yaml(cls, path: Path) -> IlmaConfig:
        """Resolve config from defaults plus one YAML file."""

        return cls._from_sources(env={}, yaml_paths=[Path(path).expanduser()])

    @classmethod
    def from_sources(
        cls,
        *,
        env: Mapping[str, str] | None = None,
        yaml_path: Path | None = None,
    ) -> IlmaConfig:
        """Defaults → YAML → env. Returns the merged config."""

        paths = [Path(yaml_path).expanduser()] if yaml_path is not None else []
        return cls._from_sources(env=env, yaml_paths=paths)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> IlmaConfig:
        """Build an :class:`IlmaConfig` from a nested mapping.

        This is primarily useful for tests and debug round-trips. Values loaded
        this way are marked as ``yaml`` when they differ from defaults because the
        input is a structured config mapping rather than process env.
        """

        return cls._from_sources(env={}, yaml_mappings=[data])

    @classmethod
    def _from_sources(
        cls,
        *,
        env: Mapping[str, str] | None = None,
        yaml_paths: Sequence[Path] = (),
        yaml_mappings: Sequence[Mapping[str, Any]] = (),
    ) -> IlmaConfig:
        values = cls().to_dict()
        sources = _default_sources(cls)

        for mapping in yaml_mappings:
            _apply_mapping(values, sources, mapping, "yaml")

        for path in yaml_paths:
            if not path.is_file():
                continue
            data = _read_yaml(path)
            _apply_mapping(values, sources, data, "yaml")

        if env is None:
            env = os.environ
        _apply_env(values, sources, env, "env")
        return cls._build(values, sources)

    @classmethod
    def _build(cls, values: Mapping[str, Any], sources: Mapping[str, ConfigSource]) -> IlmaConfig:
        return cls(
            postgres=PostgresConfig(**values.get("postgres", {})),
            memory=MemoryConfig(**values.get("memory", {})),
            vectors=VectorsConfig(**values.get("vectors", {})),
            api=ApiConfig(**values.get("api", {})),
            _sources=sources,
        )

    def source_of(self, dotted_key: str) -> ConfigSource:
        """Return the source that last set ``dotted_key``."""

        _validate_dotted_key(type(self), dotted_key)
        return self._sources.get(dotted_key, "default")

    def to_dict(self) -> dict[str, Any]:
        """Return a plain nested dict without source-tracking metadata."""

        result: dict[str, Any] = {}
        for section_field in fields(self):
            if section_field.name.startswith("_"):
                continue
            section_value = getattr(self, section_field.name)
            result[section_field.name] = _plain_dataclass_dict(section_value)
        return result

    def replace(self, dotted_key: str, value: Any, *, source: ConfigSource = "yaml") -> IlmaConfig:
        """Return a copy with one dotted key changed.

        This helper is intentionally small; config mutation is represented as a
        new immutable instance.
        """

        values = self.to_dict()
        sources = dict(self._sources)
        _set_dotted(values, sources, dotted_key, value, source)
        return type(self)._build(values, sources)


def _read_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        msg = f"ilma config must be a YAML mapping: {path}"
        raise ValueError(msg)
    return data


def _apply_mapping(
    values: dict[str, Any],
    sources: dict[str, ConfigSource],
    mapping: Mapping[str, Any],
    source: ConfigSource,
) -> None:
    for key, raw_value in mapping.items():
        if key in {"env", "environment"} and isinstance(raw_value, Mapping):
            _apply_env(values, sources, raw_value, source)
            continue
        if isinstance(key, str) and key in _ENV_TO_DOTTED:
            _set_dotted(values, sources, _ENV_TO_DOTTED[key], raw_value, source)
            continue
        if isinstance(key, str) and "." in key:
            if _is_known_dotted_key(IlmaConfig, key):
                _set_dotted(values, sources, key, raw_value, source)
            continue
        if not isinstance(raw_value, Mapping):
            continue
        section = str(key)
        if section not in values or not isinstance(values[section], dict):
            continue
        for field_name, field_value in raw_value.items():
            dotted_key = f"{section}.{field_name}"
            if _is_known_dotted_key(IlmaConfig, dotted_key):
                _set_dotted(values, sources, dotted_key, field_value, source)


def _apply_env(
    values: dict[str, Any],
    sources: dict[str, ConfigSource],
    env: Mapping[str, Any],
    source: ConfigSource,
) -> None:
    # First-match-wins within an alias group: if multiple env keys map to
    # the same dotted key (canonical name + aliases), the first key in
    # ``_ENV_TO_DOTTED`` declaration order that has a non-empty value wins.
    # This is how ``python-dotenv`` and most layered-config libraries behave,
    # and it ensures that an operator's ``ILMA_PG_DSN`` setting is not silently
    # overridden by a stray ``PG_MEM_DB_CONN_STR`` left in their shell.
    for env_key, dotted_key in _ENV_TO_DOTTED.items():
        if sources.get(dotted_key) == source:
            continue
        raw_value = env.get(env_key)
        if raw_value is None:
            continue
        if isinstance(raw_value, str) and not raw_value.strip():
            continue
        _set_dotted(values, sources, dotted_key, raw_value, source)


def _set_dotted(
    values: dict[str, Any],
    sources: dict[str, ConfigSource],
    dotted_key: str,
    raw_value: Any,
    source: ConfigSource,
) -> None:
    section, field_name = _split_dotted_key(dotted_key)
    section_values = values.setdefault(section, {})
    if not isinstance(section_values, dict):
        msg = f"config section is not a mapping: {section}"
        raise ValueError(msg)
    field_type = _field_type(section, field_name)
    section_values[field_name] = _coerce_value(raw_value, field_type, dotted_key)
    sources[dotted_key] = source


def _split_dotted_key(dotted_key: str) -> tuple[str, str]:
    parts = dotted_key.split(".", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        msg = f"unknown config key: {dotted_key}"
        raise KeyError(msg)
    return parts[0], parts[1]


def _validate_dotted_key(config_type: type[IlmaConfig], dotted_key: str) -> None:
    if not _is_known_dotted_key(config_type, dotted_key):
        msg = f"unknown config key: {dotted_key}"
        raise KeyError(msg)


def _is_known_dotted_key(config_type: type[IlmaConfig], dotted_key: str) -> bool:
    try:
        _split_dotted_key(dotted_key)
        _field_type(*_split_dotted_key(dotted_key))
    except (KeyError, ValueError):
        return False
    return dotted_key in _dotted_keys(config_type)


def _field_type(section: str, field_name: str) -> Any:
    section_field_type: Any | None = None
    section_types = _section_types(IlmaConfig)
    section_field_type = section_types.get(section)
    if section_field_type is None or not is_dataclass(section_field_type):
        msg = f"unknown config section: {section}"
        raise KeyError(msg)
    field_types = get_type_hints(section_field_type)
    for item in fields(section_field_type):
        if item.name == field_name:
            return field_types.get(item.name, item.type)
    msg = f"unknown config key: {section}.{field_name}"
    raise KeyError(msg)


def _coerce_value(raw_value: Any, target_type: Any, dotted_key: str) -> Any:
    origin = get_origin(target_type)
    args = get_args(target_type)
    if origin is Literal:
        return raw_value
    if origin is tuple:
        return _coerce_tuple(raw_value)
    if origin is list:
        return list(raw_value) if isinstance(raw_value, (list, tuple)) else [raw_value]
    if origin is not None and type(None) in args:
        non_none = next((arg for arg in args if arg is not type(None)), str)
        if raw_value is None:
            return None
        if isinstance(raw_value, str) and raw_value.strip().lower() in {"", "none", "null"}:
            return None
        return _coerce_value(raw_value, non_none, dotted_key)
    if target_type is bool:
        return _coerce_bool(raw_value, dotted_key)
    if target_type is int:
        return int(raw_value)
    if target_type is float:
        return float(raw_value)
    if target_type is str:
        return str(raw_value).strip() if isinstance(raw_value, str) else str(raw_value)
    return raw_value


def _coerce_bool(raw_value: Any, dotted_key: str) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, int):
        return bool(raw_value)
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    msg = f"invalid boolean value for {dotted_key}: {raw_value!r}"
    raise ValueError(msg)


def _coerce_tuple(raw_value: Any) -> tuple[str, ...]:
    if isinstance(raw_value, str):
        parts = [part.strip() for part in raw_value.split(",") if part.strip()]
        return tuple(parts or ["*"])
    if isinstance(raw_value, Sequence) and not isinstance(raw_value, (bytes, bytearray)):
        return tuple(str(item).strip() for item in raw_value if str(item).strip()) or ("*",)
    return (str(raw_value).strip(),)


def _plain_dataclass_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {
            item.name: _plain_dataclass_dict(getattr(value, item.name)) for item in fields(value)
        }
    if isinstance(value, tuple):
        return tuple(_plain_dataclass_dict(item) for item in value)
    if isinstance(value, list):
        return [_plain_dataclass_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain_dataclass_dict(item) for key, item in value.items()}
    return value


def _dotted_keys(config_type: type[IlmaConfig]) -> set[str]:
    keys: set[str] = set()
    section_types = _section_types(config_type)
    for section_name, section_type in section_types.items():
        if section_name.startswith("_") or not is_dataclass(section_type):
            continue
        for child_field in fields(section_type):
            keys.add(f"{section_name}.{child_field.name}")
    return keys


def _section_types(config_type: type[IlmaConfig]) -> dict[str, Any]:
    type_hints = get_type_hints(config_type)
    return {
        item.name: type_hints[item.name]
        for item in fields(config_type)
        if not item.name.startswith("_") and item.name in type_hints
    }


def _default_sources(config_type: type[IlmaConfig]) -> dict[str, ConfigSource]:
    return dict.fromkeys(_dotted_keys(config_type), "default")


__all__ = [
    "ApiConfig",
    "ConfigSource",
    "IlmaConfig",
    "MemoryConfig",
    "PostgresConfig",
    "VectorsConfig",
]
