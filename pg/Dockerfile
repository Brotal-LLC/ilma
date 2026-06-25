# syntax=docker/dockerfile:1.7
#
# ghcr.io/brotal-llc/ilma-pg
#
# Postgres 18 + the extensions ilma needs out of the box:
#
#   - pgvector       — vector embeddings (768 / 1024 / 1536 dims)
#   - pg_trgm        — trigram indexes for wiki/journal FTS fallbacks
#   - ltree          — hierarchical labels (memory.category)
#   - timescaledb    — operational metrics hypertables
#   - pg_cron        — background job scheduler (loaded but jobs are
#                      managed by the user's application)
#   - apache-age     — graph layer (ilma's cypher queries)
#
# This is the production Postgres image for ilma. The image bootstraps
# extensions on first init (no SQL schema — ilma's own `initialize_schema()`
# call from the Python package creates the `ilma` schema and tables when
# the API process boots).
#
# Multi-stage build mirrors skb50bd/hermes-memory's Dockerfile pattern:
#   Stage 1 (builder): install curl + gnupg to fetch PGDG and
#                      TimescaleDB apt keys. Nothing else.
#   Stage 2 (runtime): clean pgvector base. apt-install the three packaged
#                      extensions (cron, timescaledb, age). Drop the
#                      builder's ca-certificates after install.
#
# That gives us a ~930MB image (vs ~2.4GB if we built AGE from source).

ARG PG_MAJOR=18
ARG TIMESCALEDB_VERSION=2.18.2

# ============================================================================
# Stage 1: builder — fetch PGDG and TimescaleDB apt keys + sources
# ============================================================================
FROM pgvector/pgvector:pg${PG_MAJOR} AS builder
ARG PG_MAJOR
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg \
    && rm -rf /var/lib/apt/lists/*

# PGDG keyring + sources
RUN install -d /usr/share/postgresql-common/pgdg \
    && curl -fsSL -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc \
       https://www.postgresql.org/media/keys/ACCC4CF8.asc \
    && echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt bookworm-pgdg main" \
       > /etc/apt/sources.list.d/pgdg.list

# TimescaleDB keyring + sources
RUN curl -fsSL https://packagecloud.io/timescale/timescaledb/gpgkey \
        | gpg --dearmor -o /etc/apt/trusted.gpg.d/timescaledb.gpg \
    && echo "deb https://packagecloud.io/timescale/timescaledb/debian/ bookworm main" \
       > /etc/apt/sources.list.d/timescaledb.list

# ============================================================================
# Stage 2: runtime — clean pgvector base + extensions installed via apt
# ============================================================================
FROM pgvector/pgvector:pg${PG_MAJOR} AS runtime
ARG PG_MAJOR

ENV DEBIAN_FRONTEND=noninteractive

# Bring forward the apt sources, keyrings, AND ca-certificates from
# the builder (which has ca-certificates installed). We need
# ca-certificates in the runtime base BEFORE running `apt-get update`
# because the pgvector base image ships with no Mozilla CA bundle, so
# HTTPS repos (apt.postgresql.org, packagecloud.io) would fail cert
# verification on the first update.
COPY --from=builder /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc \
                    /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc
COPY --from=builder /etc/apt/sources.list.d/pgdg.list          /etc/apt/sources.list.d/
COPY --from=builder /etc/apt/trusted.gpg.d/timescaledb.gpg     /etc/apt/trusted.gpg.d/
COPY --from=builder /etc/apt/sources.list.d/timescaledb.list   /etc/apt/sources.list.d/
COPY --from=builder /etc/ssl/certs/ca-certificates.crt         /etc/ssl/certs/ca-certificates.crt

# Install the three extension .debs. Each pulls its declared runtime
# dependencies via apt. We use --no-install-recommends to skip doc
# packages.
#
#   - postgresql-18-cron:      pg_cron background job scheduler
#   - postgresql-18-timescaledb: TimescaleDB hypertables
#   - postgresql-18-age:       Apache AGE graph extension
#
# pgvector:vector is already in the base image. pg_trgm, ltree, plpgsql
# are stock postgres extensions and need no extra package.
#
# Cleanup: drop ca-certificates after the install (it was only needed
# for the apt-get update step) and purge the apt cache.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        postgresql-${PG_MAJOR}-cron \
        postgresql-${PG_MAJOR}-timescaledb \
        postgresql-${PG_MAJOR}-age \
    && apt-get purge -y ca-certificates \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ilma init + cron scripts. Mode 0755 so postgres user (uid 999) can
# both read (to parse the shebang) and execute (to run the scripts).
# Mode 0711 would leave postgres unable to read the file even though it
# can execute it, which manifests as "Permission denied" at first boot.
COPY --chown=root:root pg/bin/ilma-pg-entrypoint.sh  /usr/local/bin/ilma-pg-entrypoint.sh
COPY --chown=root:root pg/bin/ilma-pg-init.sh        /usr/local/bin/ilma-pg-init.sh
COPY --chown=root:root pg/bin/ilma-pg-cron.sh        /usr/local/bin/ilma-pg-cron.sh
RUN chmod 0755 /usr/local/bin/ilma-pg-entrypoint.sh \
               /usr/local/bin/ilma-pg-init.sh \
               /usr/local/bin/ilma-pg-cron.sh

# Use the ilma entrypoint wrapper that runs ilma-pg-init.sh idempotently
# on every container start. The upstream pgvector entrypoint only runs
# /docker-entrypoint-initdb.d/ scripts on first `initdb`, which breaks
# pre-initialized volumes. The wrapper symlinks our init script into the
# upstream's init.d so first-start works.
ENTRYPOINT ["/usr/local/bin/ilma-pg-entrypoint.sh"]
CMD ["postgres"]

# postgresql.conf tunables. Order matters: shared_preload_libraries must
# list pg_cron, timescaledb, age in that order — pg_cron registers workers
# first, timescaledb hooks the executor, age hooks the parser last.
RUN { \
        echo "shared_preload_libraries = 'pg_cron,timescaledb,age'"; \
        echo "cron.database_name = 'ilma_cron'"; \
    } >> /usr/share/postgresql/postgresql.conf.sample

# ilma-specific tunables
RUN { \
        echo ""; \
        echo "# ---- ilma tunables ----"; \
        echo "max_connections = 200"; \
        echo "shared_buffers = 512MB"; \
        echo "effective_cache_size = 1GB"; \
        echo "work_mem = 8MB"; \
        echo "maintenance_work_mem = 128MB"; \
        echo "wal_compression = on"; \
        echo "logging_collector = on"; \
        echo "log_destination = 'stderr'"; \
        echo "log_min_duration_statement = 250ms"; \
        echo "log_line_prefix = '%m [%p] %q%u@%d from %h '"; \
        echo "lc_messages = 'C.UTF-8'"; \
        echo "lc_monetary = 'C.UTF-8'"; \
        echo "lc_numeric = 'C.UTF-8'"; \
        echo "lc_time = 'C.UTF-8'"; \
        echo "timescaledb.telemetry_level = off"; \
    } >> /usr/share/postgresql/postgresql.conf.sample

# Healthcheck (used by docker compose / docker ps)
HEALTHCHECK --interval=5s --timeout=5s --retries=20 \
    CMD pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
        || exit 1

EXPOSE 5432
