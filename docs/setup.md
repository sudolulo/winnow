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

Edit the volume paths in `compose.yml` to match your storage layout (replace `/mnt/<pool>` with your actual path).

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

## TrueNAS Scale

The included `compose.yml` uses TrueNAS-style volume paths. Replace `<pool>` with your pool name:

```yaml
volumes:
  - /mnt/tank/winnow/models:/models
  - /mnt/tank/winnow/embeddings:/app/.if_cache
  - /mnt/tank/winnow/output:/app/frigate_train
```

GPU passthrough on TrueNAS requires the NVIDIA app to be installed from the TrueNAS catalog and the `deploy.resources.reservations.devices` block in `compose.yml` (already included).
