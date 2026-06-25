# syntax=docker/dockerfile:1.7
#
# ghcr.io/brotal-llc/ilma-ollama
#
# Thin wrapper over ollama/ollama that pre-pulls the bge-m3 model on first
# start so ilma installations can point ILMA_EMBED_BASE_URL at this image
# and have a working embedder within a couple of minutes of `docker run`.
#
# First start: ollama server starts in the background, then a small
# init script polls until the API is up, runs `ollama pull bge-m3`, and
# exits 0. The Ollama server then runs in the foreground as PID 1.
#
# Subsequent starts: the named volume (/root/.ollama) already has the
# model on disk; the init script sees the model is present and exits in
# a few seconds.

FROM ollama/ollama:latest

# Pull the model eagerly during the image build so the first `docker run`
# is instant. The model is ~2.2GB and downloads at build time, not at
# container start. Subsequent builds reuse the cached layer.
#
# `OLLAMA_KEEP_ALIVE=-1` would normally keep the model loaded forever,
# but that's a runtime env, not a build-time concern.
ARG OLLAMA_MODEL=bge-m3
RUN ollama pull ${OLLAMA_MODEL}

# Healthcheck — Ollama's HTTP API responds on /api/tags once the server
# is up. The model may take a moment to load on first request, but
# the server itself comes up in 5-10s.
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=10 \
    CMD curl -fsS http://localhost:11434/api/tags || exit 1

# Expose the Ollama HTTP API on its default port.
EXPOSE 11434

# Wrapper entrypoint: start Ollama in the background, wait for the API
# to be reachable, run `ollama pull bge-m3` (idempotent — noop if the
# model is already in the named volume), then exec the foreground
# `ollama serve` as PID 1 so Docker sees it as the main process.
COPY ollama/bin/ilma-ollama-entrypoint.sh /usr/local/bin/ilma-ollama-entrypoint.sh
RUN chmod 0755 /usr/local/bin/ilma-ollama-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/ilma-ollama-entrypoint.sh"]
CMD ["serve"]
