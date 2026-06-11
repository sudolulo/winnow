[![Publish Docker Image](https://github.com/sudolulo/if_curator_headless/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/sudolulo/if_curator_headless/actions/workflows/docker-publish.yml)
### if-curator-headless

Headless fork of [if-curator](https://github.com/ds-sebastian/if_curator) with automatic Frigate face training upload and Docker support.

> *A specialized tool to extract **high-quality, diverse** training images from your Immich library for Frigate's Face Recognition (ArcFace) and Object/State Classification models.*

## What This Adds

- **Headless mode** — runs without interactive prompts, suitable for Docker and cron
- **Auto-discover people** — processes all named people from Immich automatically instead of picking one at a time
- **Frigate upload** — sends curated face crops directly to Frigate's face training API after processing
- **People filtering** — skip specific people, whitelist only certain people, or set a minimum photo count
- **Docker** — pre-built image with NVIDIA GPU support, ready for TrueNAS
- **Scheduling** - Use cron notation to schedule runs

All of if-curator's original functionality is unchanged.

---

## Why This Tool?

> **"Diversity matters far more than volume."** — *Frigate Developer Tips*

Training AI models on "bulk" data is often harmful. If you feed the model 50 images from the same 10-second video clip, it learns to recognize the *lighting and background*, not the actual *face* or *object*.

`if-curator` solves this using **AI-powered diversity selection** and **quality filtering**:

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
- **Custom Count**: You choose the limit

### Quality Filtering
Bad training data hurts ArcFace models. Images are automatically rejected if they are:
- **Blurry** — Laplacian variance below threshold
- **Grayscale / IR** — ArcFace is trained on color images only
- **Over/Underexposed** — Washed-out or too dark to use
- **Low confidence** — Partial or occluded face detections
- **Too small** — Faces under 100px (configurable) lack features

### Face Recognition Prep
- Uses InsightFace embeddings on **face crops** (not full images — avoids wrong-face in group photos)
- **Hard example prioritization** — unusual angles, sunglasses, and low-confidence detections are biased for selection
- **Face alignment** via InsightFace landmarks (standard 112×112 ArcFace input)
- Downloads **full-resolution** originals for final crops (falls back to JPEG preview for HEIC/RAW)

### Object/State Classification Prep
- Uses **SigLIP** embeddings for semantic diversity
- **YOLOv9c** to detect and crop specific objects (dogs, cars, etc.)
- Captures variation in poses, lighting, and backgrounds

> **Note:** Frigate does not support uploading custom images for object classification training via the UI or API. Object mode crops are saved to disk for manual YOLO model training.

### Performance
- Concurrent thumbnail downloads (8 parallel workers)
- Batch-capable SigLIP embeddings for GPU efficiency
- Optional disk-based embedding cache for faster re-runs
- Multi-person batch mode

---

## Requirements

- **NVIDIA GPU** (recommended) — auto-detects CUDA for faster embedding computation. CPU mode is available via `FORCE_CPU=true` but significantly slower.
- **Python 3.12+**
- **[uv](https://astral.sh/uv/)**
- **Immich Server** (v1.106+)
- **Frigate** (v0.16+, for face recognition API — optional)

---

## Environment Variables

### New in This Fork

| Variable | Default | What it does |
| :--- | :--- | :--- |
| `AUTO_MODE` | `false` | Run without prompts |
| `FRIGATE_URL` | *(empty)* | Frigate server URL — auto-uploads faces when set |
| `TRAINING_MODE` | `face` | `face` or `object` |
| `STRATEGY` | `auto` | `auto`, `standard`, or `broad` |
| `SKIP_PEOPLE` | *(empty)* | People to skip |
| `ONLY_PEOPLE` | *(empty)* | People to process (whitelist) |
| `MIN_FACE_COUNT` | `3` | Minimum photos required to process a person |
| `OBJECT_CLASS` | `dog` | Object type (only for object mode) |

### Original if-curator Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `IMMICH_URL` | *(prompted)* | Full URL to Immich (e.g. `http://192.168.1.10:2283`) |
| `API_KEY` | *(prompted)* | Your Immich API Key |
| `FORCE_CPU` | `false` | Disable GPU acceleration |
| `MIN_FACE_WIDTH` | `100` | Minimum face crop size (pixels) |
| `BLUR_THRESHOLD` | `100.0` | Laplacian variance threshold for blur detection |
| `MIN_CONFIDENCE` | `0.7` | Minimum Immich detection confidence |
| `MAX_AUTO_IMAGES` | `80` | Safety cap for auto-diversity mode |
| `FACE_MARGIN` | `0.15` | Crop margin around face (fraction) |
| `USE_FULL_RESOLUTION` | `true` | Download originals for final crops |
| `ENABLE_FACE_ALIGNMENT` | `true` | Align faces to ArcFace 112×112 format |
| `ENABLE_CACHE` | `false` | Cache embeddings to disk for faster re-runs |
| `CACHE_DIR` | `.if_cache` | Directory for embedding cache |

---

## Docker

[ghcr.io/sudolulo/if-curator-headless](ghcr.io/sudolulo/if-curator-headless)

---

## Local Install

```bash
git clone https://github.com/sudolulo/if_curator_headless.git
cd if_curator_headless
uv sync
