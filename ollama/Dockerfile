# syntax=docker/dockerfile:1.7
#
# ghcr.io/brotal-llc/ilma-ollama
#
# Thin wrapper over ollama/ollama. The bge-m3 model is NOT pulled at
# image build time because (a) BuildKit's sandbox can't reliably reach
# a long-running background ollama server inside a single RUN step, and
# (b) pulling a 2.2GB model on every CI build is wasteful when the
# model rarely changes.
#
# Instead, the entrypoint (ollama/bin/ilma-ollama-entrypoint.sh) ensures
# bge-m3 is pulled on every container start. Idempotent: when the named
# volume /root/.ollama already has the model, the pull is a no-op.
# First start takes 1-3 minutes (download + extract); subsequent starts
# are instant.
#
# To pre-pull the model offline and bake it into a custom image, run:
#   docker run --rm -v ilma-ollama-data:/root/.ollama ghcr.io/brotal-llc/ilma-ollama:latest pull bge-m3
#   docker commit <container> my-llama:local
# Or set OLLAMA_MODEL to a different model via -e.

FROM ollama/ollama:latest

# Healthcheck — Ollama's HTTP API responds on /api/tags once the server is up.
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=10 \
    CMD curl -fsS http://localhost:11434/api/tags || exit 1

# Expose the Ollama HTTP API on its default port.
EXPOSE 11434

# Runtime entrypoint: ensures the configured model is pulled (idempotent
# — noop if model is already in the named volume /root/.ollama), then
# execs the foreground `ollama serve` so Docker sees it as PID 1.
ARG OLLAMA_MODEL=bge-m3
ENV OLLAMA_MODEL=${OLLAMA_MODEL}
COPY ollama/bin/ilma-ollama-entrypoint.sh /usr/local/bin/ilma-ollama-entrypoint.sh
RUN chmod 0755 /usr/local/bin/ilma-ollama-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/ilma-ollama-entrypoint.sh"]
CMD ["serve"]
