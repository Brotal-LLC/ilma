#!/bin/bash
#
# /usr/local/bin/ilma-pg-entrypoint.sh
# ilma wrapper around the upstream pgvector docker-entrypoint.sh.
#
# Why this exists:
#   The upstream entrypoint only runs /docker-entrypoint-initdb.d/ scripts
#   on FIRST `initdb`. For a pre-initialized volume (e.g. live
#   `ilma-postgres-data` brought up from the old pgvector image), those
#   scripts never run, and `ilma_template` / `ilma_cron` DBs never get
#   created.
#
# What this does:
#   1. Symlinks /usr/local/bin/ilma-pg-init.sh into
#      /docker-entrypoint-initdb.d/99-ilma.sh so the upstream runs it on
#      FIRST init (alongside the rest of init.d).
#   2. Hands off to the upstream entrypoint with `exec`. The upstream
#      FOREGROUNDS postgres (which is what docker expects as PID 1).
#
#   For PRE-INITIALIZED volumes, the init.d scripts never run. The ilma
#   Python package's `initialize_schema()` call handles the actual ilma
#   schema/tables on first API connection. If you need ilma_template
#   pre-populated for clones, run the init manually:
#
#     docker exec <pg-container> /usr/local/bin/ilma-pg-init.sh
#
# Set ILMA_AUTO_INIT=1 in the container env to run the init script
# automatically on first start (after initdb completes).

set -e

# Symlink the init scripts into /docker-entrypoint-initdb.d/ so the
# upstream runs them on first init. The 99- prefix ensures they run
# AFTER any other init.d scripts (the upstream's own docker_setup_db
# runs first via docker_temp_server_start).
ln -sf /usr/local/bin/ilma-pg-init.sh /docker-entrypoint-initdb.d/99-ilma-init.sh
ln -sf /usr/local/bin/ilma-pg-cron.sh /docker-entrypoint-initdb.d/99-ilma-cron.sh

# Hand off to the upstream entrypoint. `exec` replaces the shell so
# docker's PID-1 supervision sees the real postgres process.
exec /usr/local/bin/docker-entrypoint.sh "$@"
