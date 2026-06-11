# ── Build stage: CPU (amd64 + arm64) ─────────────────────────────────────
FROM --platform=$BUILDPLATFORM ubuntu:22.04 AS build

ARG TARGETARCH

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    libgl1 libglib2.0-0 libxext6 g++ tini \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && cp /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

# Copy everything first — uv sync needs the source to build the project
COPY pyproject.toml uv.lock ./
COPY if_curator/ if_curator/
COPY entrypoint.sh scheduler.py ./

RUN uv sync --extra object --frozen \
    && uv cache clean

RUN chmod +x /app/entrypoint.sh

# ── Runtime stage: CPU ────────────────────────────────────────────────────
FROM build AS runtime-cpu

RUN groupadd -g 568 apps && useradd -u 568 -g apps -m -s /bin/bash appuser \
    && mkdir -p /models/.insightface /models/huggingface \
    && chown -R appuser:apps /app /models

USER appuser
ENV HF_HOME=/models/huggingface INSIGHTFACE_HOME=/models

HEALTHCHECK CMD test -f /app/entrypoint.sh || exit 1
ENTRYPOINT ["tini", "--", "/app/entrypoint.sh"]

# ── Build stage: GPU (amd64 only) ────────────────────────────────────────
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04 AS build-gpu

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    libgl1 libglib2.0-0 libxext6 g++ tini \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && cp /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

# Copy everything first — uv sync needs the source to build the project
COPY pyproject.toml uv.lock ./
COPY if_curator/ if_curator/
COPY entrypoint.sh scheduler.py ./

RUN uv sync --extra gpu --extra object --frozen \
    && uv cache clean

RUN chmod +x /app/entrypoint.sh

# ── Runtime stage: GPU ───────────────────────────────────────────────────
FROM build-gpu AS runtime-gpu

RUN groupadd -g 568 apps && useradd -u 568 -g apps -m -s /bin/bash appuser \
    && mkdir -p /models/.insightface /models/huggingface \
    && chown -R appuser:apps /app /models

USER appuser
ENV HF_HOME=/models/huggingface INSIGHTFACE_HOME=/models

HEALTHCHECK CMD test -f /app/entrypoint.sh || exit 1
ENTRYPOINT ["tini", "--", "/app/entrypoint.sh"]

