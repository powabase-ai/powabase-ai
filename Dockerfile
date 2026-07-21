# powabase-ai — the OSS AI backend service. Installs the powabase-agentic
# library from PyPI (the import module is `agentic`) and ships ZERO charging
# logic (this repo carries no billing_cloud — criterion 5).
#   docker build --build-arg AGENTIC_VERSION=0.1.0rc1 -t powabase-ai .
FROM python:3.13-slim AS builder
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:$PATH"
RUN python -m venv /opt/venv

ARG AGENTIC_VERSION
RUN pip install "powabase-agentic[rerankers]==${AGENTIC_VERSION}"

# The project service itself.
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install . --no-deps && pip install .

FROM python:3.13-slim
WORKDIR /app
ENV VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:$PATH" \
    FLASK_APP=agentic_project_service.main:create_app
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
COPY src/ src/
COPY migrations/ migrations/
EXPOSE 5000
HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
  CMD curl -f http://localhost:5000/health || exit 1
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4", \
     "--worker-class", "gthread", "--timeout", "120", \
     "agentic_project_service.main:create_app()"]
