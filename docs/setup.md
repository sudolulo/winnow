# Setup Guide

## Prerequisites

- [Immich](https://immich.app) v1.106+ with face recognition enabled and people tagged
- [Frigate](https://frigate.video) v0.16+ (face mode only)
- Docker with the NVIDIA container toolkit (optional but strongly recommended)

---

## 1. Get your Immich API key

1. Open Immich → **Account Settings** → **API Keys**
2. Click **New API Key**, give it a name (e.g. `winnow`), copy the key

---

## 2. Get your Frigate URL

This is the base URL of your Frigate instance, e.g. `http://192.168.1.10:5000`. Only needed for face mode — omit it entirely if you're using object mode.

---

## 3. Deploy with Docker Compose

Copy [`compose.yml`](../compose.yml) and [`.env.example`](../.env.example) to a directory on your host:

```bash
mkdir winnow && cd winnow
curl -O https://raw.githubusercontent.com/sudolulo/winnow/main/compose.yml
curl -O https://raw.githubusercontent.com/sudolulo/winnow/main/.env.example
cp .env.example .env
```

Edit `.env` with your values:

```bash
IMMICH_URL=http://192.168.1.10:2283
API_KEY=your-immich-api-key
FRIGATE_URL=http://192.168.1.10:5000
```

Edit the volume paths in `compose.yml` to point to directories on your host where models, cache, and output crops should be stored:

```yaml
volumes:
  - /your/path/to/models:/models
  - /your/path/to/cache:/app/.if_cache
  - /your/path/to/output:/app/frigate_train
```

These directories will be created automatically by Docker if they don't exist.

Start it:

```bash
docker compose up -d
```

Logs:

```bash
docker compose logs -f winnow
```

---

## 4. First run

On the first run, winnow downloads the embedding models (~1–2 GB) from HuggingFace and InsightFace. This happens once — subsequent runs use the cached models from your mounted volume and start immediately.

---

## 5. Scheduling

Set `CRON_SCHEDULE` in your `.env` to keep winnow running on a schedule:

```
CRON_SCHEDULE=0 3 * * 0   # Every Sunday at 3 AM
```

Without `CRON_SCHEDULE`, the container runs once and exits.

---

## GPU passthrough

To enable GPU acceleration, include the `deploy` block in `compose.yml` (already present in the example) and ensure the NVIDIA container toolkit is installed on your host:

```bash
# Verify GPU is accessible to Docker
docker run --rm --gpus all nvidia/cuda:12.9.2-base-ubuntu22.04 nvidia-smi
```

CPU mode works without any GPU setup — set `FORCE_CPU=true` to disable GPU explicitly.
