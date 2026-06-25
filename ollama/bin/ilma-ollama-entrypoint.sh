#!/bin/bash
#
# /usr/local/bin/ilma-ollama-entrypoint.sh
#
# Wrapper around the upstream ollama image entrypoint. Starts the
# Ollama server in the background, ensures the bge-m3 model is pulled,
# then execs the foreground `ollama serve` so Docker sees it as PID 1.
#
# Idempotent: if bge-m3 is already in the named volume, the pull step
# is a no-op (ollama reports "pulling manifest" then exits 0).
#
# Why bash /dev/tcp instead of curl: the ollama/ollama base image does
# NOT ship curl (or wget, or nc). Adding curl to the image would bloat
# it by ~1MB; bash's built-in /dev/tcp/host/port works everywhere with
# no extra binary.

set -e

# Defaults — overridable via docker run -e OLLAMA_MODEL=...
: "${OLLAMA_MODEL:=bge-m3}"

# Port ollama listens on. Default 11434; overridable for non-default
# deployments (uncommon).
: "${OLLAMA_PORT:=11434}"

# Wait for localhost:port to be reachable. Uses bash's built-in
# /dev/tcp pseudo-device so no curl/wget is needed.
wait_for_port() {
    local host="$1" port="$2" max="${3:-90}"
    for i in $(seq 1 "$max"); do
        if (echo > "/dev/tcp/${host}/${port}") 2>/dev/null; then
            return 0
        fi
        sleep 1
    done
    return 1
}

echo "[ilma-ollama] Starting Ollama server in background..."
ollama serve >/tmp/ollama.log 2>&1 &
OLLAMA_PID=$!

echo "[ilma-ollama] Waiting for Ollama API..."
if ! wait_for_port localhost "$OLLAMA_PORT" 90; then
    echo "[ilma-ollama] ERROR: Ollama API never came up. Server log:" >&2
    cat /tmp/ollama.log >&2
    kill "$OLLAMA_PID" 2>/dev/null || true
    exit 1
fi

# Pull the model if not already present. `ollama pull` is idempotent —
# if the model is in the named volume, this exits quickly.
echo "[ilma-ollama] Ensuring model '$OLLAMA_MODEL' is pulled..."
ollama pull "$OLLAMA_MODEL" || {
    echo "[ilma-ollama] ERROR: ollama pull $OLLAMA_MODEL failed. Server log:" >&2
    cat /tmp/ollama.log >&2
    kill "$OLLAMA_PID" 2>/dev/null || true
    exit 1
}
echo "[ilma-ollama] Model '$OLLAMA_MODEL' is ready."

# Stop the background server and exec the foreground one. The exec
# replaces the shell so Docker's PID-1 supervision sees the real
# ollama process. We strip the `serve` arg (which was added by CMD in
# the Dockerfile to make the image discoverable) so it doesn't fail
# with "accepts 0 arg(s), received 1".
echo "[ilma-ollama] Starting foreground Ollama server..."
kill "$OLLAMA_PID" 2>/dev/null || true
# Give the background server a moment to release the port.
sleep 2
exec ollama serve
