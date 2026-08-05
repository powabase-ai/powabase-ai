# powabase-ai — the OSS AI backend service. Installs the powabase-agentic
# library from PyPI (the import module is `agentic`) and ships no billing or
# charging logic of any kind.
#   docker build -t powabase-ai .
FROM python:3.13-slim AS builder
WORKDIR /app

# uv 0.8.19, pinned. A floating tag would reintroduce exactly the kind of
# outside version channel this change exists to remove, and every uv behaviour
# relied on below was verified against this version.
COPY --from=ghcr.io/astral-sh/uv:0.8.19 /uv /uvx /bin/

# uv IGNORES VIRTUAL_ENV for project environments — it warns and creates
# ./.venv instead:
#   warning: `VIRTUAL_ENV=/opt/venv` does not match the project environment
#            path `.venv` and will be ignored; use `--active` ...
# The runtime stage copies /opt/venv, so with VIRTUAL_ENV it would copy nothing.
# UV_PROJECT_ENVIRONMENT is the variable that works.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml uv.lock README.md ./
COPY src/ src/
# --frozen: fail the build if uv.lock does not match pyproject.toml, so the
#   lock can never disagree with the image the way it did before this change.
# --no-editable: uv installs the root project EDITABLE by default (uv.lock
#   records `source = { editable = "." }`). Editable would leave a path
#   reference in the venv rather than the package, making the runtime stage's
#   `COPY src/ src/` load-bearing. Non-editable matches the previous
#   `pip install .` behaviour and lets that COPY be deleted.
# NO `--extra` flag: the extra is on a DEPENDENCY (see pyproject.toml), which
#   is a different namespace from this project's optional-dependencies.
RUN uv sync --frozen --no-editable

FROM python:3.13-slim
WORKDIR /app
ENV VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:$PATH" \
    FLASK_APP=agentic_project_service.main:create_app
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
# No `COPY src/ src/` here: `--no-editable` installs the package INTO /opt/venv,
# which the line above already carries. `migrations/` stays — alembic.ini lives
# there and is read at runtime.
COPY migrations/ migrations/
EXPOSE 5000
HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
  CMD curl -f http://localhost:5000/health || exit 1
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4", \
     "--worker-class", "gthread", "--timeout", "120", \
     "agentic_project_service.main:create_app()"]
