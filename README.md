# winnow

[![Docker](https://github.com/sudolulo/winnow/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/sudolulo/winnow/actions/workflows/docker-publish.yml) [![Test](https://github.com/sudolulo/winnow/actions/workflows/test.yml/badge.svg)](https://github.com/sudolulo/winnow/actions/workflows/test.yml) [![Immich](https://img.shields.io/badge/Immich-v1.106%2B-blueviolet)](https://immich.app) [![Frigate](https://img.shields.io/badge/Frigate-Ready-brightgreen)](https://frigate.video)

**Docs:** [Setup](https://github.com/sudolulo/winnow/wiki/Setup) · [Troubleshooting](https://github.com/sudolulo/winnow/wiki/Troubleshooting) · [FAQ](https://github.com/sudolulo/winnow/wiki/FAQ)

`winnow` pulls photos of people and objects from your [Immich](https://immich.app) library, selects the most diverse and highest-quality subset using AI embeddings, and delivers them as training data for [Frigate](https://frigate.video)'s face recognition and object classification models.

It runs fully headless in Docker, is configured entirely through environment variables, and can run on a schedule — no interactive prompts, no manual steps.

---

## The Problem

Frigate's face recognition model (ArcFace) and object classifier are only as good as the training data you give them. The instinct is to feed them as many photos as possible, but volume is not what matters — **diversity is**.

If you upload 100 photos from the same week, the model learns the lighting in your living room and the jacket you wore that month. It struggles the moment anything changes. What you actually want is a spread: different years, different lighting conditions, different angles, different contexts.

This is especially true for people who have never been to your property, or who visit rarely — family members, friends, anyone Frigate has never seen in person. Live detections alone will never build a reliable model for these people. Your photo library already has the data; winnow finds and delivers the right subset of it.

Finding that spread manually across a library of thousands of photos is not practical. `winnow` does it automatically.

---

## How It Works

For each person (or object) you configure, the tool runs this pipeline:

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
   • Grayscale / infrared (channel similarity check)
   • Over- or underexposed
   • Low detection confidence
   • Face crops below minimum pixel size
      │
      ▼
5. Compute embeddings for remaining candidates
   • Faces   → InsightFace (ArcFace / Buffalo_L)
   • Objects → SigLIP (Vision Transformer)
      │
      ▼
6. Diversity selection
   • K-Medoids clustering to find natural groupings
   • Farthest Point Sampling (FPS) to pick maximally spread representatives
   • Hard example weighting — unusual angles, partial occlusions,
     and low-confidence detections are biased toward selection
   • Auto mode: keeps selecting until marginal diversity drops off
      │
      ▼
7. Crop and export
   • Face mode: aligned 112×112 crops (ArcFace standard input),
     uploaded directly to Frigate's face training API
   • Object mode: YOLO-detected crops saved to disk
```

Uploaded asset IDs are recorded so the same image is never uploaded twice, even across runs weeks apart.

---

## Note on Crop Quality

winnow works well, but no automated pipeline is perfect. Occasionally a bad crop will slip through quality filtering — a partial face, someone in the background, a blurry frame. After a run it's worth a quick review in Frigate's face management UI to remove anything that doesn't belong.

Issues and feedback welcome via [GitHub Issues](https://github.com/sudolulo/winnow/issues).

---

## Modes

### Face Mode (default)

Extracts face crops using Immich's bounding box metadata, scales them to the source image resolution, applies EXIF orientation correction, then either aligns them to the standard ArcFace 112×112 format using 5-point facial landmarks or falls back to a margin-padded bounding box crop.

Crops are uploaded directly to Frigate's face registration API (`POST /api/faces/{name}/register`). After each successful upload the asset ID is marked in the tracker so future runs skip it.

### Object Mode

Runs each full image through YOLOv9c to detect instances of a target class (dog, cat, car, etc.), then crops each detection and saves it to the output directory. Frigate has no API for uploading object training images, so the crops are saved for you to place into your Frigate data directory manually.

---

## Diversity Selection in Detail

The core of the tool is the embedding-based selection. Rather than picking images at random or evenly across time, it computes a vector embedding for each candidate image that encodes what the face or object actually looks like — the angle, lighting, expression, background context.

It then:

1. **Clusters** those embeddings using K-Medoids to find natural groups (e.g. "holiday photos", "outdoor summer shots", "indoor low light")
2. **Selects one representative** from each cluster — the most central image in each group
3. **Fills remaining slots** using Farthest Point Sampling, iteratively picking whichever image is most different from everything already selected
4. **Weights toward hard examples** — images with unusual angles, partial occlusions, or borderline detection confidence are more likely to be picked, because those edge cases are where models fail

In **Auto mode**, there is no fixed limit. The tool keeps selecting until the most-different remaining image is already close to something already in the set — at that point adding more would be redundant. This is capped at `MAX_AUTO_IMAGES` (default 80) as a safety limit.

If the embedding model is unavailable, the tool falls back to **time spread**: evenly distributing picks across the date range of your photos.

---

## Running in Docker

### Image Tags

| Tag | Arch | GPU | Notes |
| :-- | :-- | :-- | :-- |
| `:latest` | amd64 + arm64 | CUDA 13.3 (amd64) | Requires NVIDIA Container Toolkit on amd64 |
| `:cpu` | amd64 | None | ~2 GB smaller; use if you have no NVIDIA GPU |

### Quick Start

```yaml
services:
  winnow:
    image: ghcr.io/sudolulo/winnow:latest   # or :cpu for CPU-only amd64
    environment:
      - IMMICH_URL=http://192.168.1.10:2283
      - API_KEY=your-immich-api-key
      - FRIGATE_URL=http://192.168.1.10:5000
      - CRON_SCHEDULE=0 3 * * 0   # Every Sunday at 3 AM
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

> **CPU users (`:cpu` tag):** remove the `deploy.resources` block — no NVIDIA runtime needed.

See [compose.yml](compose.yml) for the full annotated example.

### Scheduling Behaviour

`CRON_SCHEDULE` controls container lifetime:

| `CRON_SCHEDULE` value | Behaviour |
| :-- | :-- |
| *(unset)* | Run once on startup, then exit |
| *(empty string)* | Stay alive, run nothing — trigger manually with `docker exec -it winnow winnow` |
| Cron expression | Run on startup, then repeat on schedule |

In scheduled mode the process (and loaded models) stays resident between runs. In manual mode the container idles indefinitely with `sleep infinity` — useful when you want to trigger runs interactively on demand without pulling a new container each time.

The first run after a fresh install downloads the embedding models (~1-2 GB). Subsequent runs use the cached models from the mounted volume and start immediately.

---

## Environment Variables

### Mode & Strategy

| Variable | Default | Description |
| :--- | :--- | :--- |
| `AUTO_MODE` | *(auto)* | Force non-interactive mode even in a terminal; auto-detected otherwise (no TTY = auto) |
| `VERBOSE` | `false` | Set to `true` to enable DEBUG-level console output |
| `TRAINING_MODE` | `face` | `face` — upload crops to Frigate API; `object` — save crops to disk |
| `STRATEGY` | `auto` | `auto` (adaptive), `standard` (30 images), `broad` (100 images) |
| `LIMIT` | *(unset)* | Exact image count — overrides `STRATEGY` |
| `OBJECT_CLASS` | `dog` | Target class for object mode (any YOLO class: `dog`, `cat`, `car`, etc.) |

### People Filtering

| Variable | Default | Description |
| :--- | :--- | :--- |
| `ONLY_PEOPLE` | *(unset)* | Comma-separated whitelist — only these people are processed |
| `SKIP_PEOPLE` | *(unset)* | Comma-separated list of people to skip |
| `MIN_FACE_COUNT` | `0` | Skip people with fewer than N tagged assets in Immich |
| `YEARS_FILTER` | `10` | Ignore images older than N years |

### Connection

| Variable | Default | Description |
| :--- | :--- | :--- |
| `IMMICH_URL` | *(required)* | Full URL to your Immich instance |
| `API_KEY` | *(required)* | Immich API key |
| `FRIGATE_URL` | *(unset)* | Frigate URL — required for face upload; omit to skip upload |

### Image Quality

| Variable | Default | Description |
| :--- | :--- | :--- |
| `MIN_FACE_WIDTH` | `50` | Minimum face crop width in pixels |
| `FACE_MARGIN` | `0.15` | Padding added around the bounding box crop (fraction of face size) |
| `ENABLE_FACE_ALIGNMENT` | `true` | Align to ArcFace 112×112 format using facial landmarks |
| `USE_FULL_RESOLUTION` | `true` | Download full-resolution originals rather than preview thumbnails |
| `MIN_CONFIDENCE` | `0.7` | Minimum Immich face detection confidence |
| `BLUR_THRESHOLD` | `100.0` | Laplacian variance threshold — lower accepts more blur |
| `MAX_AUTO_IMAGES` | `80` | Maximum images in auto-diversity mode |

### Caching & Models

| Variable | Default | Description |
| :--- | :--- | :--- |
| `FORCE_CPU` | `false` | Disable GPU — fall back to CPU for embedding computation |
| `ENABLE_CACHE` | `true` | Cache computed embeddings to disk (speeds up re-runs on the same library) |
| `CACHE_DIR` | `.if_cache` | Path for embedding cache and upload tracker files |
| `HF_HOME` | *(system)* | HuggingFace model cache location (SigLIP) |
| `INSIGHTFACE_HOME` | *(system)* | InsightFace model cache location (Buffalo_L) |

### Tracker Overrides *(one-shot — remove after use)*

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DRY_RUN` | `false` | Show what would be selected and uploaded without doing it |
| `RETRY_REJECTED` | `false` | Re-attempt assets previously rejected by Frigate |
| `RESET_PERSON` | *(unset)* | Clear upload and rejection history for one person by name |

### Scheduling

| Variable | Default | Description |
| :--- | :--- | :--- |
| `CRON_SCHEDULE` | *(unset)* | Unset = run once and exit; empty = stay alive for manual `docker exec`; cron expression = scheduled |

---

## Local Install

```bash
git clone https://github.com/sudolulo/winnow.git
cd winnow
uv sync
uv run winnow
```

Requires Python 3.13+ and [uv](https://astral.sh/uv/). An NVIDIA GPU is strongly recommended — CPU mode works but embedding computation is significantly slower.

---

## Requirements

- **Immich** v1.106+
- **Frigate** v0.16+ (face mode only — object mode has no Frigate API dependency)
- **NVIDIA GPU** recommended (CUDA 12.x)
- **Python 3.13+**

---

## Attribution

Based on [if_curator](https://github.com/ds-sebastian/if_curator) by Sebastian, licensed MIT.
