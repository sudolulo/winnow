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

# Install uv to system path
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && cp /root/.local/bin/uv /usr/local/bin/uv \
    && chmod 755 /usr/local/bin/uv

WORKDIR /app

# Clone repo and install dependencies
RUN git clone --depth 1 https://github.com/sudolulo/if_curator_headless.git . \
    && uv sync --extra gpu \
    && uv add croniter \
    && uv cache clean

# Apply enhanced upload_to_frigate function in-place
RUN sed -i '/def upload_to_frigate/,/if failed == total_files/c\
def upload_to_frigate(jobs: list[dict]) -> None:\
    """Upload processed face crops to Frigate via API with detailed logging."""\
    frigate_url = os.environ.get("FRIGATE_URL", "")\
    if not frigate_url:\
        rprint("[yellow]⚠️  FRIGATE_URL not set, skipping upload.[/yellow]")\
        return\
\
    rprint("\\n[bold cyan]📤 Uploading to Frigate[/bold cyan]")\
    rprint(f"  Target: [dim]{frigate_url}[/dim]")\
\
    total_files = 0\
    for job in jobs:\
        name = job["person"]["name"]\
        person_dir = os.path.join(Config.OUTPUT_DIR, name)\
        if not os.path.isdir(person_dir): continue\
        total_files += sum(1 for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))\
\
    if total_files == 0:\
        rprint("  [yellow]No images found to upload.[/yellow]")\
        return\
\
    rprint(f"  People: [bold]{len(jobs)}[/bold], Total images: [bold]{total_files}[/bold]")\
    uploaded, failed = 0, 0\
\
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(), console=console) as progress:\
        upload_task = progress.add_task("[green]Uploading to Frigate", total=total_files)\
        for job in jobs:\
            name = job["person"]["name"]\
            encoded_name = quote(name, safe="")\
            person_dir = os.path.join(Config.OUTPUT_DIR, name)\
            person_files = sorted(f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))) if os.path.isdir(person_dir) else []\
            for fname in person_files:\
                fpath = os.path.join(person_dir, fname)\
                try:\
                    with open(fpath, "rb") as f:\
                        resp = requests.post(f"{frigate_url}/api/faces/train/{encoded_name}/classify", files={"file": (fname, f, "image/jpeg")}, timeout=30)\
                    if resp.status_code == 200: uploaded += 1\
                    else: failed += 1\
                except Exception: failed += 1\
                progress.advance(upload_task)\
\
    rprint("\\n  [bold]Frigate Upload Summary:[/bold]\\n    ✅ Succeeded: [green]{uploaded}[/green]\\n    ❌ Failed:    [red]{failed}[/red]")' /app/if_curator/cli.py

# Copy scripts
COPY entrypoint.sh scheduler.py /app/
RUN chmod +x /app/entrypoint.sh

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

