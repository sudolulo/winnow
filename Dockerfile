FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    libgl1 \
    libglib2.0-0 \
    libxext6 \
    git \
    curl \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

RUN git clone --depth 1 https://github.com/sudolulo/if_curator_headless.git . \
    && uv sync --extra gpu \
    && uv cache clean

RUN groupadd -g 568 apps \
    && useradd -u 568 -g apps -m -s /bin/bash appuser \
    && chown -R appuser:apps /app

USER appuser

ENV FORCE_CPU=false

ENTRYPOINT ["uv", "run", "if-curator"]
