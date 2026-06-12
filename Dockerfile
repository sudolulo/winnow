# ── Platform-conditional base ─────────────────────────────────────────────
#   amd64:  NVIDIA CUDA 13.3 (GPU acceleration when available, CPU fallback)
#   arm64:  Ubuntu 24.04 (CPU-only; no CUDA on ARM)

FROM --platform=$BUILDPLATFORM nvidia/cuda:13.3.0-cudnn-runtime-ubuntu22.04 AS base-amd64
FROM ubuntu:24.04 AS base-arm64

# ── Build stage ───────────────────────────────────────────────────────────
ARG TARGETARCH

FROM base-${TARGETARCH} AS build

ENV DEBIAN_FRONTEND=noninteractive

# Both bases (Ubuntu 22.04 CUDA / Ubuntu 24.04) need Python 3.13 from the
# deadsnakes PPA. GNUPGHOME is isolated to a tmpdir so gpg never tries to
# contact an agent socket, which fails silently under QEMU.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg software-properties-common \
    && GNUPGHOME=$(mktemp -d) add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.13 python3.13-venv python3.13-dev \
    libgl1 libglib2.0-0 libxext6 g++ \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.13 /usr/bin/python3

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && cp /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev \
    && uv cache clean

COPY winnow/ winnow/
COPY entrypoint.sh scheduler.py ./
RUN chmod +x /app/entrypoint.sh

# ── Runtime stage ─────────────────────────────────────────────────────────
#   Starts fresh from the base image — excludes build tools (g++,
#   python3.13-dev, gnupg, software-properties-common) not needed at runtime.

FROM base-${TARGETARCH} AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg software-properties-common tini \
    && GNUPGHOME=$(mktemp -d) add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.13 python3.13-venv \
    libgl1 libglib2.0-0 libxext6 \
    && apt-get purge -y --auto-remove curl gnupg software-properties-common \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.13 /usr/bin/python3

# Copy app (with .venv) and uv from build stage
COPY --from=build /app /app
COPY --from=build /usr/local/bin/uv /usr/local/bin/uv

# Expose CUDA/cuDNN libraries from pip packages so onnxruntime-gpu
# can find libcublasLt.so.12 and libcudnn.so.9 at runtime (amd64 only)
ENV LD_LIBRARY_PATH="/app/.venv/lib/python3.13/site-packages/nvidia/cudnn/lib:/app/.venv/lib/python3.13/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH}"

RUN groupadd -g 568 apps && useradd -u 568 -g apps -m -s /bin/bash appuser \
    && mkdir -p /models/.insightface /models/huggingface \
    && chown -R appuser:apps /app /models

WORKDIR /app
USER appuser
ENV HF_HOME=/models/huggingface INSIGHTFACE_HOME=/models/.insightface

HEALTHCHECK CMD test -f /app/entrypoint.sh || exit 1
ENTRYPOINT ["tini", "--", "/app/entrypoint.sh"]
