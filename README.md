# winnow

[![Docker](https://github.com/sudolulo/winnow/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/sudolulo/winnow/actions/workflows/docker-publish.yml) [![Test](https://github.com/sudolulo/winnow/actions/workflows/test.yml/badge.svg)](https://github.com/sudolulo/winnow/actions/workflows/test.yml) [![GitHub release](https://img.shields.io/github/v/release/sudolulo/winnow)](https://github.com/sudolulo/winnow/releases/latest) [![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE) [![Immich](https://img.shields.io/badge/Immich-v1.106%2B-blueviolet)](https://immich.app) [![Frigate](https://img.shields.io/badge/Frigate-Ready-brightgreen)](https://frigate.video)

> **Note:** winnow's approach to training Frigate face recognition is not an officially documented workflow — results may vary.

> **Early Development — Use With Caution**
> winnow is in an unfinished state and maturing. Features that modify your Frigate training data — quality replacement, stale mapping cleanup — can remove images from your dataset and are not yet battle-tested at scale. Review the logs after each run and keep backups of your Frigate face training directory until you are confident in the results.

**Docs:** [Setup](https://github.com/sudolulo/winnow/wiki/Setup) · [Troubleshooting](https://github.com/sudolulo/winnow/wiki/Troubleshooting) · [FAQ](https://github.com/sudolulo/winnow/wiki/FAQ)

`winnow` pulls photos from your [Immich](https://immich.app) library, selects the most diverse and highest-quality subset using AI embeddings, and delivers them as training data for [Frigate](https://frigate.video)'s face recognition.

The best Frigate training data is images you curate manually — photos taken specifically for recognition, in controlled conditions, uploaded directly through Frigate's UI. Winnow is meant to supplement people in your library, not replace manual training. In some cases one has people they would like to recognize that do not occur in detections often enough to train a diverse dataset. This is meant to fill that gap.

> **winnow only touches files it uploaded.** Faces added to Frigate manually through its UI are never deleted, replaced, or modified — not by quality replacement, not by `RESET_PERSON`, not by stale cleanup. Your manually curated images are always the primary dataset; winnow only adds to it.

---

## How It Works

```
Immich library
      │
      ▼
1. Fetch all assets tagged with this person
      │
      ▼
2. Filter by recency (YEARS_FILTER) and skip already-uploaded
   and rejected assets (persistent tracker in DATA_DIR)
      │
      ▼
3. Quality filter — download preview thumbnails and reject:
   • Blurry images (Laplacian variance)
   • Grayscale / infrared (channel similarity)
   • Over- or underexposed
   • Low detection confidence
   • Face crops below minimum pixel size
      │
      ▼
4. Compute embeddings from the same preview thumbnails
   • InsightFace (ArcFace / Buffalo_L) → 512-dim vector
      │
      ▼
5. Near-duplicate removal — greedy cosine-distance pass drops burst shots
   and near-identical photos before clustering runs; the highest-quality
   image from each near-duplicate group is kept
      │
      ▼
6. Diversity selection
   • K-Medoids clustering → one representative per natural group
   • Farthest Point Sampling → fill remaining slots with maximally spread picks
   • Hard example weighting — low-confidence detections get a distance boost
     so unusual angles and harder looks are preferred over easy frontals
   • Adaptive mode: stops when the next candidate is too similar to those already
     selected (distance threshold = 20 % of median pairwise distance)
      │
      ▼
7. Download full-resolution originals from Immich
      │
      ▼
8. Crop and process — EXIF-corrected, landmark-aligned 112×112 crop (ArcFace format)
      │
      ▼
9. Deliver — upload crops to Frigate's face registration API
   ↳ below MAX_AUTO_IMAGES — upload, unless the novelty gate
     (FRIGATE_SCORE_CEILING) determines the candidate is already
     covered by the current training set
   ↳ at cap + QUALITY_REPLACEMENT=true — with Frigate scoring active,
     swap the most redundant tracked image (highest pre-upload recognize
     score) if the candidate is more novel (lower score); falling back to
     blur-score comparison when no Frigate scores are available; manually
     added files are never touched
   ↳ at cap + QUALITY_REPLACEMENT=false — skip this person
```

Uploaded and rejected asset IDs are persisted across runs in a SQLite database (`winnow_tracker.db` in `DATA_DIR`). The same image is never processed twice; rejected assets are permanently skipped unless `RETRY_REJECTED=true`.

---

## Running in Docker

### Image Tags

| Tag | Arch | Acceleration |
| :-- | :-- | :-- |
| `:latest` | amd64 | NVIDIA CUDA 12.8 · requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) |
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
      - /path/to/models:/models    # INSIGHTFACE_HOME — persists Buffalo_L model (~300 MB)
      - /path/to/data:/app/data
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

In scheduled mode the process (and loaded models) stays resident between runs. The first run after a fresh install downloads InsightFace Buffalo_L (~300 MB); subsequent runs use the cached model from the mounted volume.

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
| `STRATEGY` | `adaptive` | `adaptive` — embedding-based diversity selection, stops when candidates become redundant; `standard` — fixed 30 images; `broad` — fixed 100 images |
| `LIMIT` | *(unset)* | Exact image count — overrides `STRATEGY` |
| `AUTO_MODE` | *(auto)* | Skip interactive prompts and process all people unattended — auto-detected when no TTY is present (Docker, cron); set `true` to force in a terminal |
| `VERBOSE` | `false` | Enable DEBUG-level console output (log file is always DEBUG) |

### People Filtering

| Variable | Default | Description |
| :--- | :--- | :--- |
| `ONLY_PEOPLE` | *(unset)* | Comma-separated whitelist — process only these people |
| `SKIP_PEOPLE` | *(unset)* | Comma-separated list — skip these people |
| `MIN_FACE_COUNT` | `3` | Skip people with fewer than N tagged assets in Immich |
| `MERGE_DUPLICATE_PEOPLE` | `false` | When Immich has duplicate entries for the same person (same face split across multiple names), merge their asset pools before processing. Without this, each duplicate group emits a warning and is skipped |
| `YEARS_FILTER` | `10` | Ignore images older than N years |

> **Duplicate people detection** — winnow warns at startup if the same name appears on multiple Immich person records (a common side-effect of Immich's face clustering creating separate pools for the same individual). By default (`false`) it logs the duplicates, keeps only the person with the most assets, and skips the rest — no data is changed. Set `MERGE_DUPLICATE_PEOPLE=true` to permanently merge each duplicate group inside Immich (the person with the most assets absorbs the others). **This modifies Immich and cannot be undone.** Only enable it once you've verified the duplicates are actually the same person.

### Image Quality

| Variable | Default | Description |
| :--- | :--- | :--- |
| `MAX_AUTO_IMAGES` | `20` | Maximum training images per person in Frigate |
| `QUALITY_REPLACEMENT` | `true` | When at cap, swap a weaker tracked image for a better candidate. With Frigate scoring active, targets the most redundant image (highest pre-upload recognize score); otherwise uses blur score. Never touches manually added Frigate files. Set `false` to skip people at cap |

#### Advanced Tuning *(calibrated — do not adjust)*

These defaults are tuned for Frigate's ArcFace requirements. winnow will warn on launch if any are set. Image quality issues caused by non-default values will not be investigated.

| Variable | Default | Description |
| :--- | :--- | :--- |
| `ENABLE_FRIGATE_SCORES` | `true` | Call Frigate's recognize endpoint pre-upload to store diversity scores used for quality replacement. Adds ~200 ms per upload. Disabling also disables the below-cap novelty gate |
| `FRIGATE_SCORE_CEILING` | *(unset)* | Below-cap novelty gate. Unset: dynamic — skips candidates whose Frigate score exceeds the most-redundant tracked file's score, auto-calibrates each run. `0`: disable entirely. Positive value (e.g. `0.85`): fixed hard ceiling |
| `MIN_FACE_WIDTH` | `90` | Minimum face crop width in pixels |
| `FACE_MARGIN` | `0.15` | Padding around bounding box crop (fraction of face size) |
| `ENABLE_FACE_ALIGNMENT` | `true` | Align to ArcFace 112×112 format using facial landmarks |
| `USE_FULL_RESOLUTION` | `true` | Download full-resolution originals rather than preview thumbnails |
| `MIN_CONFIDENCE` | `0.7` | Minimum Immich face detection confidence |
| `BLUR_THRESHOLD` | `120.0` | Laplacian variance threshold — lower accepts more blur |

### GPU & Models

| Variable | Default | Description |
| :--- | :--- | :--- |
| `FORCE_CPU` | `false` | Disable GPU — fall back to CPU for all inference |
| `OPENVINO_DEVICE` | `CPU` | Intel variant only: set `GPU` to use Arc or iGPU; default runs on CPU |
| `ENABLE_CACHE` | `true` | Cache computed embeddings to disk (speeds up re-runs on the same library) |
| `DATA_DIR` | `data` | Path for embedding cache and upload tracker database (`winnow_tracker.db`) |
| `INSIGHTFACE_HOME` | *(system)* | InsightFace model cache path (Buffalo_L) |

### Output

| Variable | Default | Description |
| :--- | :--- | :--- |
| `OUTPUT_DIR` | `./frigate_train` | Directory where face crops are staged before upload and where `winnow.log` is written. In Docker, set this via the volume mount instead. |

### Tracker Overrides *(one-shot — remove after use)*

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DRY_RUN` | `false` | Preview selection without downloading or uploading |
| `RETRY_REJECTED` | `false` | Re-attempt all previously rejected assets (low-confidence skips, Frigate rejections, and other permanent exclusions) |
| `RESET_PERSON` | *(unset)* | Set to a person's name to clear their upload history and delete their winnow-managed Frigate training files so the next run starts fresh. Set to `*` to reset all tracked people at once. Manually added Frigate files are never touched |
| `TRACE_CROP_SIZE` | *(unset)* | Debug: print all tracked crops whose width or height matches this pixel value, then exit |

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

When run with a terminal attached, winnow starts an interactive session: select which people to process and choose a strategy (adaptive, standard, broad, or a custom count) per person. Without a TTY — Docker, cron, or `AUTO_MODE=true` — it processes all people unattended using the configured defaults.

---

## Requirements

- **Immich** v1.106+
- **Frigate** v0.16+
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
