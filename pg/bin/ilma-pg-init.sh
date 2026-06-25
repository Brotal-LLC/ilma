#!/bin/bash
#
# /usr/local/bin/ilma-pg-init.sh
# Bundled in the image. Idempotent. Initializes the ilma_template DB
# with the extensions ilma needs. The ilma Python package's
# initialize_schema() creates the actual ilma schema and tables on
# first API connection; this script only ensures extensions exist and
# a template DB is ready for cloning.
#
# Run once after first boot:
#
#   docker exec <pg-container> /usr/local/bin/ilma-pg-init.sh
#
# Or set ILMA_AUTO_INIT=1 in the container env to run automatically
# on first start (after initdb completes).
#
# Concurrency: this script can be invoked from two places on a fresh
# container — the /docker-entrypoint-initdb.d/ symlink (runs in
# docker_temp_server during first boot) AND an explicit `docker exec`
# call (e.g. from CI smoke tests). These race on the same DBs. We
# serialize via a session-level advisory lock that is held for the
# entire duration of this script's work, and use \gexec to avoid the
# check-then-create TOCTOU window inside a single session.

set -e

DB="${ILMA_TEMPLATE_DB:-ilma_template}"

echo "[ilma-init] Target database: $DB"

# Wait for postgres to be ready
until pg_isready -U "$POSTGRES_USER" -d postgres > /dev/null 2>&1; do
    echo "[ilma-init] Waiting for postgres..."
    sleep 1
done

# Acquire advisory lock + create the template DB + release the lock,
# all in one psql session. The lock prevents a concurrent invocation
# of this script (e.g. the init.d symlink) from racing with us.
#   lockid = hashtext('ilma_init') & 0x7FFFFFFF
psql -U "$POSTGRES_USER" -d postgres -v ON_ERROR_STOP=1 <<SQL
SELECT pg_advisory_lock(hashtext('ilma_init')::int);
SELECT 'CREATE DATABASE $DB'
 WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname='$DB')
\gexec
SELECT pg_advisory_unlock(hashtext('ilma_init')::int);
SQL

# Install the extensions ilma needs. pg_cron is intentionally NOT
# installed here — it lives in ilma_cron (separate DB) so workers
# don't block CREATE DATABASE ... TEMPLATE clones.
echo "[ilma-init] Installing extensions..."
psql -U "$POSTGRES_USER" -d "$DB" -v ON_ERROR_STOP=1 <<'SQL'
-- pgvector: vector embeddings (768 / 1024 / 1536 dims)
CREATE EXTENSION IF NOT EXISTS vector;

-- pg_trgm: trigram indexes for fuzzy text match
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ltree: hierarchical category paths (memory.category)
-- Note: ltree moved out of core into an extension in PG 18.
CREATE EXTENSION IF NOT EXISTS ltree;

-- timescaledb: operational metrics hypertables
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- apache-age: graph layer (ilma's cypher queries)
CREATE EXTENSION IF NOT EXISTS age;
SQL

# Last step: install pg_cron in its own dedicated DB. This is what lets
# the workers idle without blocking CREATE DATABASE ... TEMPLATE clones.
echo "[ilma-init] Setting up pg_cron in ilma_cron..."
/usr/local/bin/ilma-pg-cron.sh

echo "[ilma-init] Done. $DB is ready for 'ilma profile create <name>'."
