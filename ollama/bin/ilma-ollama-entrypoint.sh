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

set -e

# Defaults — overridable via docker run -e OLLAMA_MODEL=...
: "${OLLAMA_MODEL:=bge-m3}"

echo "[ilma-ollama] Starting Ollama server in background..."
ollama serve >/tmp/ollama.log 2>&1 &
OLLAMA_PID=$!

# Wait for the API to come up. /api/version is the lightest endpoint.
echo "[ilma-ollama] Waiting for Ollama API..."
for i in $(seq 1 60); do
    if curl -fsS http://localhost:11434/api/version >/dev/null 2>&1; then
        echo "[ilma-ollama] Ollama API is up."
        break
    fi
    sleep 1
done

if ! curl -fsS http://localhost:11434/api/version >/dev/null 2>&1; then
    echo "[ilma-ollama] ERROR: Ollama API never came up. Server log:"
    cat /tmp/ollama.log
    kill "$OLLAMA_PID" 2>/dev/null || true
    exit 1
fi

# Pull the model if not already present. `ollama pull` is idempotent —
# if the model is in the named volume, this exits quickly.
echo "[ilma-ollama] Ensuring model '$OLLAMA_MODEL' is pulled..."
ollama pull "$OLLAMA_MODEL" || {
    echo "[ilma-ollama] ERROR: ollama pull $OLLAMA_MODEL failed. Server log:"
    cat /tmp/ollama.log
    kill "$OLLAMA_PID" 2>/dev/null || true
    exit 1
}
echo "[ilma-ollama] Model '$OLLAMA_MODEL' is ready."

# Stop the background server and exec the foreground one. The exec
# replaces the shell so Docker's PID-1 supervision sees the real
# ollama process.
echo "[ilma-ollama] Starting foreground Ollama server..."
kill "$OLLAMA_PID" 2>/dev/null || true
# Give the background server a moment to release the port.
sleep 2
exec ollama serve "$@"
