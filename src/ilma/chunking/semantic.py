"""Text chunking for memory embeddings."""

from __future__ import annotations

from dataclasses import dataclass

#: Approximate chars per token for English text.
CHUNKER_TOKEN_RATIO = 4
#: Default chunk window size in tokens.
DEFAULT_WINDOW_TOKENS = 512
#: Default overlap between adjacent chunks in tokens.
DEFAULT_OVERLAP_TOKENS = 50


@dataclass(frozen=True)
class Chunk:
    """A single chunk of a longer text."""

    index: int
    text: str
    token_count: int

    def __post_init__(self) -> None:
        if self.index < 0:
            msg = f"chunk index must be >= 0, got {self.index}"
            raise ValueError(msg)
        if not self.text or not self.text.strip():
            msg = f"chunk {self.index} has empty text"
            raise ValueError(msg)
        if self.token_count <= 0:
            msg = f"chunk {self.index} has non-positive token_count={self.token_count}"
            raise ValueError(msg)


def _approx_tokens(text: str) -> int:
    if not text or not text.strip():
        return 0
    return max(1, len(text) // CHUNKER_TOKEN_RATIO)


def _chars_for_tokens(tokens: int) -> int:
    return tokens * CHUNKER_TOKEN_RATIO


def chunk_text(
    text: str,
    window_tokens: int = DEFAULT_WINDOW_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Split text into overlapping 512-token windows with 50-token overlap."""
    if window_tokens <= 0:
        msg = f"window_tokens must be > 0, got {window_tokens}"
        raise ValueError(msg)
    if overlap_tokens < 0:
        msg = f"overlap_tokens must be >= 0, got {overlap_tokens}"
        raise ValueError(msg)
    if overlap_tokens >= window_tokens:
        msg = f"overlap_tokens ({overlap_tokens}) must be < window_tokens ({window_tokens})"
        raise ValueError(msg)

    if not text or not text.strip():
        return []

    window_chars = _chars_for_tokens(window_tokens)
    overlap_chars = _chars_for_tokens(overlap_tokens)
    step = window_chars - overlap_chars
    if step <= 0:
        raise ValueError("chunk step must be positive")

    chunks: list[Chunk] = []
    start = 0
    index = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + window_chars, text_len)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(
                Chunk(
                    index=index,
                    text=chunk,
                    token_count=min(_approx_tokens(chunk), window_tokens),
                )
            )
            index += 1
        if end == text_len:
            break
        start += step
        if start == end:
            break
    return chunks
