# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN python -m pip install --upgrade pip wheel

COPY pyproject.toml README.md ./
COPY src ./src

# Build ilma and all runtime wheels in one isolated stage. The runtime image
# installs only from /wheels and does not need compilers, source metadata, or pip
# network access.
RUN python -m pip wheel --wheel-dir /wheels ".[http,mcp]"

FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="ilma" \
      org.opencontainers.image.description="ilma HTTP API" \
      org.opencontainers.image.source="https://github.com/brotal/ilma"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    ILMA_PG_POOL_MIN=1 \
    ILMA_PG_POOL_MAX=8

WORKDIR /app

RUN groupadd --gid 1000 ilma \
    && useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin ilma

COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir --no-index --find-links=/wheels "ilma-agent[http,mcp]" \
    && rm -rf /wheels

USER 1000:1000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import sys, urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).read(); sys.exit(0)"

CMD ["ilma", "serve", "--host", "0.0.0.0", "--port", "8000"]
