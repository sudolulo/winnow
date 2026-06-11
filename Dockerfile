# ── Platform-conditional base image ──────────────────────────────────────
#   amd64:  NVIDIA CUDA 12.6 runtime (matches cu126 torch wheels)
#   arm64:  Plain Ubuntu (CPU-only; no CUDA on ARM64)

FROM --platform=$BUILDPLATFORM nvidia/cuda:12.6.3-runtime-ubuntu22.04 AS base-amd64
FROM --platform=$BUILDPLATFORM ubuntu:22.04 AS base-arm64

# ── Build stage ───────────────────────────────────────────────────────────
ARG TARGETARCH

FROM base-${TARGETARCH} AS build

ENV DEBIAN_FRONTEND=noninteractive

# Certificates + curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Python 3.12 + system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    libgl1 libglib2.0-0 libxext6 g++ tini \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && cp /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

# Copy project files for deterministic, cached builds
COPY pyproject.toml uv.lock README.md ./
COPY if_curator/ if_curator/
COPY entrypoint.sh scheduler.py ./

# Install dependencies — uv resolves per-platform via tool.uv.sources markers
#   amd64:  --extra gpu --extra object  → CUDA torch + onnxruntime-gpu
#   arm64:  --extra object              → CPU torch + onnxruntime
RUN if [ "$TARGETARCH" = "amd64" ]; then \
      uv sync --extra gpu --extra object; \
    else \
      uv sync --extra object; \
    fi \
    && uv add croniter \
    && uv cache clean

RUN chmod +x /app/entrypoint.sh

# ── Runtime stage ─────────────────────────────────────────────────────────
FROM build AS runtime

RUN groupadd -g 568 apps && useradd -u 568 -g apps -m -s /bin/bash appuser \
    && mkdir -p /models/.insightface /models/huggingface \
    && chown -R appuser:apps /app /models

USER appuser
ENV HF_HOME=/models/huggingface INSIGHTFACE_HOME=/models

HEALTHCHECK CMD test -f /app/entrypoint.sh || exit 1
ENTRYPOINT ["tini", "--", "/app/entrypoint.sh"]

