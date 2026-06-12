[![Publish Docker Image](https://github.com/sudolulo/if_curator_headless/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/sudolulo/if_curator_headless/actions/workflows/docker-publish.yml) [![Release](https://github.com/sudolulo/if_curator_headless/actions/workflows/release.yml/badge.svg)](https://github.com/sudolulo/if_curator_headless/actions/workflows/release.yml) [![Lint](https://github.com/sudolulo/if_curator_headless/actions/workflows/lint.yml/badge.svg)](https://github.com/sudolulo/if_curator_headless/actions/workflows/lint.yml) [![Test](https://github.com/sudolulo/if_curator_headless/actions/workflows/test.yml/badge.svg)](https://github.com/sudolulo/if_curator_headless/actions/workflows/test.yml)

### if-curator-headless

> *A specialized tool to extract **high-quality, diverse** training images from your Immich library for Frigate's Face Recognition (ArcFace) and Object/State Classification models.*

Runs headless in Docker with full GPU acceleration, automatic Frigate upload, cron scheduling, and env-var-driven configuration — no interactive prompts required.

---

## Why This Tool?

> **"Diversity matters far more than volume."** — *Frigate Developer Tips*

Training AI models on bulk data is often harmful. Feed the model 50 images from the same 10-second video clip and it learns the *lighting and background*, not the actual *face* or *object*.

`if-curator-headless` solves this using **AI-powered diversity selection** and **quality filtering**:

| Mode | Embedding Model | Algorithm |
| :--- | :--- | :--- |
| **👤 Face** | InsightFace (ArcFace) | K-Medoids Clustering + FPS + Hard Example Weighting |
| **🐶 Object** | SigLIP (Vision Transformer) | K-Medoids Clustering + FPS |

---

## Features

### Smart Selection
- **Auto Diversity [Recommended]**: Clusters images by visual similarity, selects representatives from each cluster, then fills with maximally-diverse picks until redundancy starts (capped at 80)
- **Standard (30 images)**: Balanced set using Smart Diversity
- **Broad (100 images)**: Extensive set using Smart Diversity
- **Custom Count**: Set `LIMIT=N` for any exact number

### Quality Filtering
Images are automatically rejected if they are:
- **Blurry** — Laplacian variance below threshold
- **Grayscale / IR** — ArcFace is trained on color images only
- **Over/Underexposed** — Washed-out or too dark to use
- **Low confidence** — Partial or occluded face detections
- **Too small** — Faces under the minimum pixel width lack features

### Face Recognition Prep
- Uses InsightFace embeddings on **face crops** (not full images — avoids wrong-face in group photos)
- **Hard example prioritization** — unusual angles, sunglasses, and low-confidence detections are biased for selection
- **Face alignment** via InsightFace landmarks (standard 112×112 ArcFace input)
- **EXIF orientation** applied before cropping so bounding boxes align correctly
- Downloads **full-resolution** originals for final crops (falls back to JPEG preview for HEIC/RAW)

### Object/State Classification Prep
- Uses **SigLIP** embeddings for semantic diversity
- **YOLOv9c** to detect and crop specific objects (dogs, cars, etc.)
- Captures variation in poses, lighting, and backgrounds

> **Note:** Frigate does not support uploading custom images for object classification via the UI or API. Object mode crops are saved to disk for manual YOLO model training.

### Performance
- Concurrent thumbnail downloads (8 parallel workers)
- Batch-capable SigLIP embeddings for GPU efficiency
- Optional disk-based embedding cache for faster re-runs
- Upload deduplication — already-uploaded assets are skipped on future runs
- Multi-person batch mode

---

## Requirements

- **NVIDIA GPU** (recommended) — auto-detects CUDA. CPU mode available via `FORCE_CPU=true` but significantly slower.
- **Python 3.12+**
- **[uv](https://astral.sh/uv/)**
- **Immich Server** (v1.106+)
- **Frigate** (v0.16+, for face recognition API — optional)

---

## Environment Variables

### Mode & Strategy

| Variable | Default | Description |
| :--- | :--- | :--- |
| `AUTO_MODE` | `false` | Run without interactive prompts |
| `TRAINING_MODE` | `face` | `face` or `object` |
| `STRATEGY` | `auto` | `auto`, `standard` (30), or `broad` (100) |
| `LIMIT` | *(unset)* | Custom image count — overrides `STRATEGY` |
| `OBJECT_CLASS` | `dog` | Object label for object mode (e.g. `dog`, `cat`, `car`) |

### People Filtering

| Variable | Default | Description |
| :--- | :--- | :--- |
| `ONLY_PEOPLE` | *(unset)* | Comma-separated whitelist of people to process |
| `SKIP_PEOPLE` | *(unset)* | Comma-separated list of people to skip |
| `MIN_FACE_COUNT` | `0` | Skip people with fewer than N assets in Immich |
| `YEARS_FILTER` | `10` | Only include images from the last N years |

### Connection

| Variable | Default | Description |
| :--- | :--- | :--- |
| `IMMICH_URL` | *(required)* | Full URL to Immich (e.g. `http://192.168.1.10:2283`) |
| `API_KEY` | *(required)* | Your Immich API Key |
| `FRIGATE_URL` | *(unset)* | Frigate server URL — enables automatic face upload when set |

### Image Quality

| Variable | Default | Description |
| :--- | :--- | :--- |
| `MIN_FACE_WIDTH` | `50` | Minimum face crop width in pixels |
| `FACE_MARGIN` | `0.15` | Padding around face crop as a fraction of face size |
| `ENABLE_FACE_ALIGNMENT` | `true` | Align face to ArcFace 112×112 format before cropping |
| `USE_FULL_RESOLUTION` | `true` | Download full-resolution originals for final crops |
| `MIN_CONFIDENCE` | `0.7` | Minimum Immich face detection confidence |
| `BLUR_THRESHOLD` | `100.0` | Laplacian variance threshold — lower accepts more blur |
| `MAX_AUTO_IMAGES` | `80` | Hard cap on auto-diversity selection |

### Caching & Models

| Variable | Default | Description |
| :--- | :--- | :--- |
| `FORCE_CPU` | `false` | Disable GPU acceleration |
| `ENABLE_CACHE` | `false` | Cache embeddings to disk for faster re-runs |
| `CACHE_DIR` | `.if_cache` | Directory for embedding cache |
| `HF_HOME` | *(system)* | Override HuggingFace model cache location |
| `INSIGHTFACE_HOME` | *(system)* | Override InsightFace model cache location |

### Tracker Overrides *(one-shot — remove after use)*

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DRY_RUN` | `false` | Preview selection without downloading or uploading |
| `RETRY_REJECTED` | `false` | Re-attempt previously rejected images |
| `RESET_PERSON` | *(unset)* | Clear uploaded + rejected history for one person by name |

### Scheduling

| Variable | Default | Description |
| :--- | :--- | :--- |
| `CRON_SCHEDULE` | *(unset)* | Cron expression — unset runs once and exits |

---

## Docker

```bash
docker pull ghcr.io/sudolulo/if-curator-headless:latest
```

See [compose.yml](compose.yml) for a full example with volume mounts and GPU support.

---

## Local Install

```bash
git clone https://github.com/sudolulo/if_curator_headless.git
cd if_curator_headless
uv sync
uv run if-curator
```

---

## Attribution

Based on [if-curator](https://github.com/ds-sebastian/if_curator) by Sebastian, licensed MIT.
