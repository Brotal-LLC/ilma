"""Embedding providers and registry."""

from __future__ import annotations

import os
from typing import Protocol

import httpx

SUPPORTED_DIMS = (768, 1024, 1536)
DEFAULT_DIM = 1024


class EmbeddingError(Exception):
    """Raised when an embedder fails to produce a vector."""


class Embedder(Protocol):
    """Embedding provider protocol."""

    @property
    def dim(self) -> int:
        """Vector dimensionality."""
        ...

    @property
    def model(self) -> str:
        """Provider model identifier."""
        ...

    def embed(self, text: str) -> list[float]:
        """Embed text into a dense vector."""
        ...


class HttpEmbedder:
    """OpenAI-compatible HTTP embedder (Ollama, OpenAI, vLLM, etc.)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        dim: int,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        if dim not in SUPPORTED_DIMS:
            msg = f"unsupported dim {dim}; choose one of {SUPPORTED_DIMS}"
            raise ValueError(msg)
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dim = dim
        self._api_key = api_key
        self._client = httpx.Client(timeout=timeout)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model(self) -> str:
        return self._model

    def embed(self, text: str) -> list[float]:
        if not text or not text.strip():
            return [0.0] * self._dim

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            resp = self._client.post(
                f"{self._base_url}/embeddings",
                headers=headers,
                json={"model": self._model, "input": text},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            msg = f"embedder POST failed: {exc}"
            raise EmbeddingError(msg) from exc

        data = resp.json()
        try:
            vec = data["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            msg = f"embedder response missing 'data[0].embedding': {exc}"
            raise EmbeddingError(msg) from exc

        if len(vec) != self._dim:
            vec = vec[: self._dim] if len(vec) > self._dim else vec + [0.0] * (self._dim - len(vec))
        return [float(x) for x in vec]


class EmbedderRegistry:
    """Dispatch queries to embedders by dimensionality.

    This is the framework-agnostic port of the hermes-memory registry. It uses
    ILMA_* environment variables and supports OpenAI-compatible providers.
    """

    def __init__(self, embedders: dict[int, Embedder], *, default_dim: int = DEFAULT_DIM) -> None:
        if default_dim not in SUPPORTED_DIMS:
            msg = f"unsupported default dim {default_dim}; choose one of {SUPPORTED_DIMS}"
            raise ValueError(msg)
        if default_dim not in embedders:
            msg = f"default dim {default_dim} must be in the registry"
            raise ValueError(msg)
        self._embedders = embedders
        self._default_dim = default_dim

    @property
    def default_dim(self) -> int:
        return self._default_dim

    @classmethod
    def from_env(cls) -> EmbedderRegistry:
        provider = os.environ.get("ILMA_EMBED_PROVIDER", "ollama_local").strip()
        if provider == "ollama_local":
            base_url = os.environ.get("ILMA_EMBED_BASE_URL", "http://localhost:11434/v1")
            dim = int(os.environ.get("ILMA_EMBED_DIM", "1024"))
            model = os.environ.get(
                "ILMA_EMBED_MODEL",
                "bge-m3" if dim == 1024 else "nomic-embed-text-v2-moe",
            )
        elif provider == "openai":
            base_url = os.environ.get("ILMA_EMBED_BASE_URL", "https://api.openai.com/v1")
            dim = int(os.environ.get("ILMA_EMBED_DIM", "1536"))
            model = os.environ.get("ILMA_EMBED_MODEL", "text-embedding-3-small")
        elif provider == "http":
            base_url = os.environ["ILMA_EMBED_BASE_URL"]
            dim = int(os.environ["ILMA_EMBED_DIM"])
            model = os.environ["ILMA_EMBED_MODEL"]
        else:
            msg = (
                f"unknown ILMA_EMBED_PROVIDER: {provider!r} "
                "(expected ollama_local, openai, or http)"
            )
            raise ValueError(msg)
        api_key = os.environ.get("ILMA_EMBED_API_KEY")
        embedder = HttpEmbedder(base_url=base_url, model=model, dim=dim, api_key=api_key)
        return cls({dim: embedder}, default_dim=dim)

    def embed(self, text: str, *, dim: int | None = None) -> list[float]:
        target = dim or self._default_dim
        embedder = self._embedders.get(target)
        if embedder is None:
            msg = f"no embedder for dim={target}"
            raise EmbeddingError(msg)
        return embedder.embed(text)

    def get(self, dim: int) -> Embedder | None:
        return self._embedders.get(dim)


__all__ = [
    "DEFAULT_DIM",
    "SUPPORTED_DIMS",
    "Embedder",
    "EmbedderRegistry",
    "EmbeddingError",
    "HttpEmbedder",
]
