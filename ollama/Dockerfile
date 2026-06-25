# syntax=docker/dockerfile:1.7
#
# ghcr.io/brotal-llc/ilma-ollama
#
# Thin wrapper over ollama/ollama that pre-pulls the bge-m3 model so ilma
# installations can point ILMA_EMBED_BASE_URL at this image and have a
# working embedder within seconds of `docker run`.
#
# Build strategy: ollama pull needs a running ollama server, so we can't
# just `RUN ollama pull bge-m3` directly. Instead:
#   1. Start ollama serve in the background.
#   2. Wait for the API to come up.
#   3. Pull bge-m3 into the image's /root/.ollama dir.
#   4. Stop the background server.
#
# The resulting image has the model pre-pulled at /root/.ollama. At runtime,
# the entrypoint pulls again only if the model is missing (e.g. user started
# with a fresh named volume). Persistent deployments keep the model in the
# named volume so startup is instant.

FROM ollama/ollama:latest

# Pull the model eagerly during the image build so the first `docker run`
# is instant. The model is ~2.2GB and downloads at build time, not at
# container start.
#
# Three stages: start ollama serve, wait for it, pull bge-m3, stop ollama.
# Buildkit's RUN supports inline bash, so we chain them with `&&`.
ARG OLLAMA_MODEL=bge-m3
RUN set -eux; \
    ollama serve >/tmp/ollama-build.log 2>&1 & \
    OLLAMA_PID=$!; \
    for i in $(seq 1 120); do \
        if curl -fsS http://localhost:11434/api/version >/dev/null 2>&1; then break; fi; \
        sleep 1; \
    done; \
    if ! curl -fsS http://localhost:11434/api/version >/dev/null 2>&1; then \
        echo "ollama server failed to come up during build:" >&2; \
        cat /tmp/ollama-build.log >&2; \
        kill $OLLAMA_PID 2>/dev/null || true; \
        exit 1; \
    fi; \
    ollama pull "$OLLAMA_MODEL"; \
    kill $OLLAMA_PID; \
    wait $OLLAMA_PID 2>/dev/null || true

# Healthcheck — Ollama's HTTP API responds on /api/tags once the server is up.
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=10 \
    CMD curl -fsS http://localhost:11434/api/tags || exit 1

# Expose the Ollama HTTP API on its default port.
EXPOSE 11434

# Runtime entrypoint: ensures bge-m3 is pulled (idempotent — noop if model
# is already in the named volume /root/.ollama), then execs the foreground
# `ollama serve` so Docker sees it as PID 1.
COPY ollama/bin/ilma-ollama-entrypoint.sh /usr/local/bin/ilma-ollama-entrypoint.sh
RUN chmod 0755 /usr/local/bin/ilma-ollama-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/ilma-ollama-entrypoint.sh"]
CMD ["serve"]
