#!/bin/bash
#
# /usr/local/bin/ilma-pg-cron.sh
# One-shot installer for the pg_cron worker DB. Creates the ilma_cron
# database (the one DB pg_cron is allowed to pin idle sessions in) and
# installs the pg_cron extension there.
#
# Called by ilma-pg-init.sh as the last step. Users add their own cron
# jobs via `SELECT cron.schedule(...)` against the ilma_cron DB.
#
# Note: pg_cron + TimescaleDB each keep one idle session pinned to
# cron.database_name, which blocks `CREATE DATABASE ... TEMPLATE`
# clones of ilma_template. That's why ilma_cron is separate from
# ilma_template: clones of ilma_template are clean, while cron workers
# idle in ilma_cron.

set -e

CRON_DB="ilma_cron"

echo "[ilma-cron] Target database: $CRON_DB"

# Wait for postgres to be ready
until pg_isready -U "$POSTGRES_USER" -d postgres > /dev/null 2>&1; do
    echo "[ilma-cron] Waiting for postgres..."
    sleep 1
done

# Create the cron DB (advisory-lock + TOCTOU-safe via \gexec)
psql -U "$POSTGRES_USER" -d postgres -v ON_ERROR_STOP=1 <<SQL
SELECT pg_advisory_lock(hashtext('ilma_cron_init')::int);
SELECT 'CREATE DATABASE $CRON_DB'
 WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname='$CRON_DB')
\gexec
SELECT pg_advisory_unlock(hashtext('ilma_cron_init')::int);
SQL

# Install pg_cron in the cron DB. Users add their own jobs after init.
echo "[ilma-cron] Installing pg_cron in $CRON_DB..."
psql -U "$POSTGRES_USER" -d "$CRON_DB" -v ON_ERROR_STOP=1 <<'SQL'
CREATE EXTENSION IF NOT EXISTS pg_cron;
SQL

echo "[ilma-cron] Done. pg_cron is loaded in $CRON_DB. Use:"
echo "  psql -U \$POSTGRES_USER -d $CRON_DB -c \"SELECT cron.schedule('job_name', '0 4 * * *', \$\$SQL\$\$);\""
