FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    libgl1 libglib2.0-0 libxext6 git curl g++ tini \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && cp /root/.local/bin/uv /usr/local/bin/uv \
    && chmod 755 /usr/local/bin/uv

WORKDIR /app

# Clone your specific repo
RUN git clone --depth 1 https://github.com/sudolulo/if_curator_headless.git . \
    && uv sync --extra gpu \
    && uv add croniter \
    && uv cache clean

# Permissions for entrypoint
RUN chmod +x entrypoint.sh

# Setup non-root user and persistent model paths
RUN groupadd -g 568 apps \
    && useradd -u 568 -g apps -m -s /bin/bash appuser \
    && mkdir -p /models/.insightface /models/huggingface \
    && chown -R appuser:apps /app /models

USER appuser

# Baked-in configurations
ENV FORCE_CPU=false \
    HF_HOME=/models/huggingface \
    INSIGHTFACE_HOME=/models

HEALTHCHECK --interval=5m --timeout=10s --start-period=600s --retries=3 \
  CMD test -f /app/entrypoint.sh || exit 1

ENTRYPOINT ["tini", "--", "/app/entrypoint.sh"]

