# ── Platform-conditional base ─────────────────────────────────────────────
#   amd64:  NVIDIA CUDA 12.6 (GPU acceleration when available, CPU fallback)
#   arm64:  Plain Ubuntu (CPU-only, no CUDA on ARM)

FROM --platform=$BUILDPLATFORM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 AS base-amd64
FROM --platform=$BUILDPLATFORM ubuntu:22.04 AS base-arm64

# ── Build stage ───────────────────────────────────────────────────────────
ARG TARGETARCH

FROM base-${TARGETARCH} AS build

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

COPY pyproject.toml uv.lock ./
COPY if_curator/ if_curator/
COPY entrypoint.sh scheduler.py ./

RUN uv sync --frozen \
    && uv cache clean

RUN chmod +x /app/entrypoint.sh

# ── Runtime stage ─────────────────────────────────────────────────────────
FROM build AS runtime

# Expose CUDA/cuDNN libraries from pip packages so onnxruntime-gpu
# can find libcublasLt.so.12 and libcudnn.so.9 at runtime
ENV LD_LIBRARY_PATH="/app/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/app/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH}"

RUN groupadd -g 568 apps && useradd -u 568 -g apps -m -s /bin/bash appuser \
    && mkdir -p /models/.insightface /models/huggingface \
    && chown -R appuser:apps /app /models

USER appuser
ENV HF_HOME=/models/huggingface INSIGHTFACE_HOME=/models

HEALTHCHECK CMD test -f /app/entrypoint.sh || exit 1
ENTRYPOINT ["tini", "--", "/app/entrypoint.sh"]

