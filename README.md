# winnow

[![Docker](https://github.com/sudolulo/winnow/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/sudolulo/winnow/actions/workflows/docker-publish.yml) [![Test](https://github.com/sudolulo/winnow/actions/workflows/test.yml/badge.svg)](https://github.com/sudolulo/winnow/actions/workflows/test.yml) [![GitHub release](https://img.shields.io/github/v/release/sudolulo/winnow)](https://github.com/sudolulo/winnow/releases/latest) [![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE) [![Immich](https://img.shields.io/badge/Immich-v1.106%2B-blueviolet)](https://immich.app) [![Frigate](https://img.shields.io/badge/Frigate-Ready-brightgreen)](https://frigate.video)

**Docs:** [Setup](https://github.com/sudolulo/winnow/wiki/Setup) · [Troubleshooting](https://github.com/sudolulo/winnow/wiki/Troubleshooting) · [FAQ](https://github.com/sudolulo/winnow/wiki/FAQ)

`winnow` pulls photos from your [Immich](https://immich.app) library, selects the most diverse and highest-quality subset using AI embeddings, and delivers them as training data for [Frigate](https://frigate.video)'s face recognition and object classification models.

Frigate's face recognition is only as good as its training data — and the key quality metric is **diversity**, not volume. A hundred photos from the same week teach the model one lighting condition. What you need is a spread: different years, different angles, different lighting, different contexts. Your photo library already has that data. winnow finds and delivers the right subset automatically.

---

## How It Works

```
Immich library
      │
      ▼
1. Fetch all assets tagged with this person
      │
      ▼
2. Filter by recency (configurable years window)
      │
      ▼
3. Skip already-uploaded assets (persistent tracker)
      │
      ▼
4. Quality filter — reject:
   • Blurry images (Laplacian variance)
   • Grayscale / infrared (channel similarity)
   • Over- or underexposed
   • Low detection confidence
   • Face crops below minimum pixel size
      │
      ▼
5. Compute embeddings
   • Faces   → InsightFace (ArcFace / Buffalo_L)
   • Objects → SigLIP (Vision Transformer)
      │
      ▼
6. Diversity selection
   • K-Medoids clustering → one representative per natural group
   • Farthest Point Sampling → fill remaining slots with maximally spread picks
   • Hard example weighting — unusual angles and low-confidence detections
     are biased toward selection, since those are where models tend to fail
   • Auto mode: stops when the next candidate is too similar to what's already chosen
      │
      ▼
7. Crop and deliver
   • Face mode: aligned 112×112 crops uploaded directly to Frigate's face training API
   • Object mode: YOLO-detected crops saved to disk
```

Uploaded asset IDs are persisted so the same image is never uploaded twice, even across runs weeks apart.

---

## Modes

**Face mode** (default) — extracts face crops using Immich's bounding box metadata, applies EXIF orientation correction, and aligns them to ArcFace's standard 112×112 format using 5-point facial landmarks. Crops are uploaded directly to Frigate's face registration API.

**Object mode** — runs each image through YOLOv9c to detect instances of a target class (dog, cat, car, etc.), crops each detection, and saves it to the output directory. Frigate has no API for uploading object training data; place the crops into your Frigate data directory manually.

---

## Running in Docker

### Image Tags

| Tag | Arch | Acceleration |
| :-- | :-- | :-- |
| `:latest` | amd64 + arm64 | NVIDIA CUDA 13.3 (amd64) · requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) |
| `:rocm` | amd64 | AMD ROCm · pass `/dev/kfd` + `/dev/dri` |
| `:intel` | amd64 | Intel Arc / iGPU via OpenVINO · pass `/dev/dri`, set `OPENVINO_DEVICE=GPU` |
| `:cpu` | amd64 + arm64 | CPU only · ~2 GB smaller · no GPU required |

### Quick Start

**NVIDIA:**
```yaml
services:
  winnow:
    image: ghcr.io/sudolulo/winnow:latest
    environment:
      - IMMICH_URL=http://192.168.1.10:2283
      - API_KEY=your-immich-api-key
      - FRIGATE_URL=http://192.168.1.10:5000
      - CRON_SCHEDULE=0 3 * * 0
    volumes:
      - /path/to/models:/models
      - /path/to/cache:/app/.if_cache
      - /path/to/output:/app/frigate_train
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

**AMD (`:rocm`):** use `image: ghcr.io/sudolulo/winnow:rocm` and replace the `deploy:` block with:
```yaml
    devices:
      - /dev/kfd
      - /dev/dri
    group_add:
      - video
      - render
```

**Intel (`:intel`):** use `image: ghcr.io/sudolulo/winnow:intel` and replace the `deploy:` block with:
```yaml
    devices:
      - /dev/dri
    group_add:
      - render
    environment:
      - OPENVINO_DEVICE=GPU   # omit to run OpenVINO inference on CPU (default)
```

**CPU (`:cpu`):** use `image: ghcr.io/sudolulo/winnow:cpu`, remove the `deploy:` block, and add `mem_limit: 2g` to prevent OOM on large libraries.

See [compose.yml](compose.yml) for the full annotated example with all options.

### Scheduling

`CRON_SCHEDULE` controls container lifetime:

| `CRON_SCHEDULE` value | Behaviour |
| :-- | :-- |
| *(unset)* | Run once on startup, then exit |
| *(empty string)* | Stay alive, run nothing — trigger manually with `docker exec -it winnow winnow` |
| Cron expression | Run on startup, then repeat on schedule |

In scheduled mode the process (and loaded models) stays resident between runs. The first run after a fresh install downloads the embedding models (~1–2 GB); subsequent runs use the cached models from the mounted volume.

---

## Environment Variables

### Connection

| Variable | Default | Description |
| :--- | :--- | :--- |
| `IMMICH_URL` | *(required)* | Full URL to your Immich instance |
| `API_KEY` | *(required)* | Immich API key |
| `FRIGATE_URL` | *(unset)* | Frigate URL — required for face upload; omit to skip |

### Mode & Strategy

| Variable | Default | Description |
| :--- | :--- | :--- |
| `TRAINING_MODE` | `face` | `face` — upload crops to Frigate; `object` — save crops to disk |
| `STRATEGY` | `auto` | `auto` (adaptive), `standard` (30 images), `broad` (100 images) |
| `LIMIT` | *(unset)* | Exact image count — overrides `STRATEGY` |
| `OBJECT_CLASS` | `dog` | Target class for object mode (any YOLO class: `dog`, `cat`, `car`, etc.) |
| `AUTO_MODE` | *(auto)* | Force non-interactive mode in a terminal; auto-detected otherwise |
| `VERBOSE` | `false` | Enable DEBUG-level console output (log file is always DEBUG) |

### People Filtering

| Variable | Default | Description |
| :--- | :--- | :--- |
| `ONLY_PEOPLE` | *(unset)* | Comma-separated whitelist — process only these people |
| `SKIP_PEOPLE` | *(unset)* | Comma-separated list — skip these people |
| `MIN_FACE_COUNT` | `0` | Skip people with fewer than N tagged assets in Immich |
| `YEARS_FILTER` | `10` | Ignore images older than N years |

### Image Quality

| Variable | Default | Description |
| :--- | :--- | :--- |
| `MIN_FACE_WIDTH` | `50` | Minimum face crop width in pixels |
| `FACE_MARGIN` | `0.15` | Padding around bounding box crop (fraction of face size) |
| `ENABLE_FACE_ALIGNMENT` | `true` | Align to ArcFace 112×112 format using facial landmarks |
| `USE_FULL_RESOLUTION` | `true` | Download full-resolution originals rather than preview thumbnails |
| `MIN_CONFIDENCE` | `0.7` | Minimum Immich face detection confidence |
| `BLUR_THRESHOLD` | `100.0` | Laplacian variance threshold — lower accepts more blur |
| `MAX_AUTO_IMAGES` | `80` | Maximum training images per person in Frigate |
| `QUALITY_REPLACEMENT` | `true` | When at cap, swap the lowest-quality uploaded image for a better candidate. Never touches manually added Frigate files. Set `false` to skip people already at cap |

### GPU & Models

| Variable | Default | Description |
| :--- | :--- | :--- |
| `FORCE_CPU` | `false` | Disable GPU — fall back to CPU for all inference |
| `OPENVINO_DEVICE` | `CPU` | Intel variant only: set `GPU` to use Arc or iGPU; default runs on CPU |
| `ENABLE_CACHE` | `true` | Cache computed embeddings to disk (speeds up re-runs on the same library) |
| `CACHE_DIR` | `.if_cache` | Path for embedding cache and upload tracker files |
| `HF_HOME` | *(system)* | HuggingFace model cache path (SigLIP) |
| `INSIGHTFACE_HOME` | *(system)* | InsightFace model cache path (Buffalo_L) |

### Tracker Overrides *(one-shot — remove after use)*

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DRY_RUN` | `false` | Preview selection without downloading or uploading |
| `RETRY_REJECTED` | `false` | Re-attempt assets previously rejected by Frigate |
| `RESET_PERSON` | *(unset)* | Clear upload and rejection history for one person by name |

### Scheduling

| Variable | Default | Description |
| :--- | :--- | :--- |
| `CRON_SCHEDULE` | *(unset)* | Unset = run once and exit; empty = stay alive; cron expression = scheduled |

---

## Local Install

```bash
git clone https://github.com/sudolulo/winnow.git
cd winnow
uv sync
uv run winnow
```

Requires Python 3.13+ and [uv](https://astral.sh/uv). An NVIDIA, AMD, or Intel GPU is recommended — CPU mode works but embedding computation is slower.

---

## Requirements

- **Immich** v1.106+
- **Frigate** v0.16+ (face mode only — object mode has no Frigate dependency)
- **GPU** recommended: NVIDIA (CUDA), AMD (ROCm), or Intel (Arc / iGPU via OpenVINO)
- **Python** 3.13+

---

## Getting Help

- **[GitHub Discussions](https://github.com/sudolulo/winnow/discussions)** — questions, setup help, and general discussion
- **[Wiki](https://github.com/sudolulo/winnow/wiki)** — setup guide, troubleshooting, and FAQ
- **[Issues](https://github.com/sudolulo/winnow/issues)** — bugs and feature requests only

---

## Attribution

Based on [if_curator](https://github.com/ds-sebastian/if_curator) by Sebastian, licensed MIT.
