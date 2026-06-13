# ── Base images ───────────────────────────────────────────────────────────────
#   amd64 + gpu:   NVIDIA CUDA 12.8 + cuDNN (GPU acceleration via NVIDIA Container Toolkit)
#   amd64 + rocm:  Ubuntu 22.04 (AMD GPU via ROCm — pass /dev/kfd and /dev/dri)
#   amd64 + intel: Ubuntu 22.04 (Intel Arc / iGPU via OpenVINO — pass /dev/dri)
#   amd64 + cpu:   Ubuntu 22.04 (CPU-only, ~2 GB smaller image)
#   arm64:         Ubuntu 24.04 (CPU-only; no CUDA/ROCm wheels on ARM)

ARG VARIANT=gpu

FROM --platform=$BUILDPLATFORM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 AS base-amd64-gpu
FROM ubuntu:22.04 AS base-amd64-rocm
FROM ubuntu:22.04 AS base-amd64-intel
FROM ubuntu:22.04 AS base-amd64-cpu
FROM ubuntu:24.04 AS base-arm64-gpu
FROM ubuntu:24.04 AS base-arm64-rocm
FROM ubuntu:24.04 AS base-arm64-intel
FROM ubuntu:24.04 AS base-arm64-cpu

# ── Build stage ───────────────────────────────────────────────────────────────
ARG TARGETARCH

FROM base-${TARGETARCH}-${VARIANT} AS build

ARG VARIANT=gpu
ENV DEBIAN_FRONTEND=noninteractive

# Both Ubuntu 22.04 and 24.04 get Python 3.13 from the deadsnakes PPA.
# GNUPGHOME is isolated so gpg never contacts an agent socket under QEMU.
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

# Swap in the variant-specific pyproject and lockfile before syncing.
COPY pyproject.toml uv.lock pyproject-cpu.toml uv-cpu.lock \
     pyproject-rocm.toml uv-rocm.lock pyproject-intel.toml uv-intel.lock ./
RUN if [ "$VARIANT" = "cpu" ]; then \
        cp pyproject-cpu.toml pyproject.toml && cp uv-cpu.lock uv.lock; \
    elif [ "$VARIANT" = "rocm" ]; then \
        cp pyproject-rocm.toml pyproject.toml && cp uv-rocm.lock uv.lock; \
    elif [ "$VARIANT" = "intel" ]; then \
        cp pyproject-intel.toml pyproject.toml && cp uv-intel.lock uv.lock; \
    fi && \
    uv sync --frozen --no-dev \
    && uv cache clean

COPY winnow/ winnow/
COPY entrypoint.sh scheduler.py ./
RUN chmod +x /app/entrypoint.sh

# ── Runtime stage ─────────────────────────────────────────────────────────────
#   Starts fresh from the base image — excludes build tools (g++,
#   python3.13-dev, gnupg, software-properties-common) not needed at runtime.

FROM base-${TARGETARCH}-${VARIANT} AS runtime

ARG VARIANT=gpu
ARG VERSION=dev
LABEL org.opencontainers.image.title="winnow" \
      org.opencontainers.image.description="Selects diverse, high-quality photos from Immich as training data for Frigate face recognition and object classification." \
      org.opencontainers.image.source="https://github.com/sudolulo/winnow" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later" \
      org.opencontainers.image.version="${VERSION}"
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

# NVIDIA: register pip-installed nvidia lib/ dirs with ldconfig so onnxruntime-gpu
# and torch can find libcudnn, libcublas, etc. Skipped silently on other variants.
RUN if [ "$VARIANT" = "gpu" ]; then \
        find /app/.venv/lib/python3.*/site-packages/nvidia -type d -name "lib" \
            2>/dev/null > /etc/ld.so.conf.d/nvidia-pip.conf && ldconfig || true; \
    fi
# Intel: install GPU compute runtime so OpenVINO EP can target Intel Arc / iGPU.
# onnxruntime-openvino bundles OpenVINO itself; only the userspace GPU driver
# (OpenCL ICD + Level Zero) is needed from the OS.
# These packages aren't in Ubuntu 22.04 main, so this block adds Intel's
# official GPU repo first, then installs. libze-intel-gpu1 was renamed to
# level-zero in Intel's repo.
RUN if [ "$VARIANT" = "intel" ]; then \
        apt-get update \
        && apt-get install -y --no-install-recommends curl gnupg \
        && curl -fsSL https://repositories.intel.com/graphics/intel-graphics.key \
           | gpg --dearmor > /usr/share/keyrings/intel-graphics.gpg \
        && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] \
https://repositories.intel.com/graphics/ubuntu jammy flex" \
           > /etc/apt/sources.list.d/intel-graphics.list \
        && apt-get update \
        && apt-get install -y --no-install-recommends \
               intel-opencl-icd intel-level-zero-gpu level-zero \
        && apt-get remove -y --autoremove curl gnupg \
        && rm -rf /var/lib/apt/lists/*; \
    fi

RUN groupadd -g 568 apps && useradd -u 568 -g apps -m -s /bin/bash appuser \
    && mkdir -p /models/.insightface /models/huggingface \
    && chown -R appuser:apps /app /models

WORKDIR /app
USER appuser
# PYTHONPATH=/app makes the winnow package importable from the entry point script.
# uv sync builds the wheel before winnow/ is COPY'd, so site-packages has only
# the dist-info. Explicitly adding /app lets Python find winnow/__init__.py there.
ENV HF_HOME=/models/huggingface INSIGHTFACE_HOME=/models/.insightface PYTHONPATH=/app

HEALTHCHECK CMD test -f /app/entrypoint.sh || exit 1
ENTRYPOINT ["tini", "--", "/app/entrypoint.sh"]
