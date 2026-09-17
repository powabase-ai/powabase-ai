# syntax=docker/dockerfile:1
#
# The Postgres the `pg_search` CI job tests against: ParadeDB's
# paradedb/paradedb:0.25.9-pg15 image with its pg_search library rebuilt from
# the v0.25.9 source plus ParadeDB pull request #6211.
#
# Stock pg_search 0.25.9 on Postgres 15 and 16 fails a bm25
# CREATE INDEX CONCURRENTLY while another backend writes to the table: the
# build errors with "buffer ... is not owned by resource owner" (XX000, index
# left INVALID) or the server crashes. The partition-move tests build indexes
# under concurrent writes on purpose, so against the stock image they fail
# intermittently. #6211 fixes it upstream and is in no 0.25.x release; see
# the patch file's header.
#
# Everything except pg_search.so and its extension files comes from the
# unmodified upstream image. The compile takes 8 minutes with 16 parallel jobs
# and far longer on fewer CPUs; CI caches the layers (see
# .github/workflows/test.yml).
#
# pg_search is AGPL-3.0. This Dockerfile only builds it for tests; the image is
# not published.
#
# Local build:
#   docker build -t pg-search-ci:0.25.9-6211 ci/pg_search

ARG PARADEDB_IMAGE=paradedb/paradedb:0.25.9-pg15@sha256:2f4690be0f3175d453ac11efdb373f908427a8724d9356fb3a31069a7714fdbc

FROM ${PARADEDB_IMAGE} AS builder

# v0.25.9 ("chore: Prepare `0.25.9`.")
ARG PARADEDB_COMMIT=c727757be0aab7fbb17c6cc5f0360f22efdd3776
# The versions v0.25.9 pins: rust-toolchain.toml and the pgrx entry in Cargo.lock.
ARG RUST_TOOLCHAIN=1.97.1
ARG CARGO_PGRX_VERSION=0.19.0
# Unset by default (cargo uses every CPU); pass --build-arg CARGO_BUILD_JOBS=N
# to bound the compile's memory on a shared machine.
ARG CARGO_BUILD_JOBS

SHELL ["/bin/bash", "-o", "pipefail", "-euc"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential clang libclang-dev pkg-config libssl-dev cmake \
        postgresql-server-dev-15 libopenblas-dev gfortran curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain "${RUST_TOOLCHAIN}"
RUN cargo install --locked cargo-pgrx --version "${CARGO_PGRX_VERSION}" --debug \
    && cargo pgrx init --pg15=/usr/lib/postgresql/15/bin/pg_config

WORKDIR /src
RUN git init -q . \
    && git fetch -q --depth 1 https://github.com/paradedb/paradedb.git "${PARADEDB_COMMIT}" \
    && git checkout -q FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "${PARADEDB_COMMIT}"

COPY *.patch /patches/
RUN for patch in /patches/*.patch; do git apply --verbose "$patch"; done

# The features of a release build (pg_search's defaults), for pg15 instead of
# the default pg18. The target directory is a cache mount so a failed or
# repeated local build keeps its compiled crates; the artifacts are copied out
# in the same step.
WORKDIR /src/pg_search
RUN --mount=type=cache,target=/src/target \
    cargo pgrx package \
        --pg-config /usr/lib/postgresql/15/bin/pg_config \
        --no-default-features --features "pg15 deferred_wal" \
    && mkdir -p /out \
    && cp -r /src/target/release/pg_search-pg15 /out/

FROM ${PARADEDB_IMAGE}
COPY --from=builder /out/pg_search-pg15/usr/lib/postgresql/15/lib/pg_search.so /usr/lib/postgresql/15/lib/pg_search.so
COPY --from=builder /out/pg_search-pg15/usr/share/postgresql/15/extension/ /usr/share/postgresql/15/extension/
# The marker the service reads before it builds a bm25 index concurrently
# (pg_bm25_index.concurrent_build_safety): nothing pg_search itself reports
# tells this build from stock 0.25.9. initdb copies the sample into the data
# directory, so a new cluster starts with it; a custom setting with a dot in
# its name needs no extension to define it.
RUN printf '\n# pg_search here contains paradedb/paradedb#6211\npowabase.pg_search_cic_safe = on\n' \
        >> /usr/share/postgresql/postgresql.conf.sample
LABEL org.opencontainers.image.description="paradedb 0.25.9-pg15 with pg_search rebuilt from v0.25.9 + paradedb#6211, for tests only"
