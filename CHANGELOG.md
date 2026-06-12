# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-06-12

First release of winnow. Forked from [if-curator](https://github.com/ds-sebastian/if_curator) by Sebastian and rewritten for headless Docker deployment.

### Added

**Headless operation**
- `AUTO_MODE` env var — runs without any interactive prompts; required for Docker/cron use
- `DRY_RUN` env var — previews selection without downloading, cropping, or uploading anything
- `RETRY_REJECTED` env var — re-attempts assets previously rejected by Frigate's face API
- `RESET_PERSON` env var — clears upload and rejection history for one named person

**Docker and scheduling**
- `Dockerfile` — multi-stage build (CUDA 12.9 on amd64, plain Ubuntu on arm64); runtime stage excludes build tools (g++, python3.12-dev, curl, gnupg)
- `compose.yml` — fully annotated with inline comments grouped by concern; TrueNAS volume paths in the example
- `entrypoint.sh` — runs the tool once on startup, then hands off to the scheduler if `CRON_SCHEDULE` is set
- `scheduler.py` — in-process cron scheduler that keeps the container (and loaded models) alive between runs
- `CRON_SCHEDULE` env var — standard cron expression for recurring runs; unset exits after first run
- `.dockerignore` — keeps `.venv`, `__pycache__`, test files, and logs out of the image context
- Multi-arch image: `linux/amd64` and `linux/arm64` built and merged into a single manifest on GHCR
- `tini` as PID 1 init process for correct signal handling
- Non-root container user (`appuser`, uid 568) matching TrueNAS default app UID
- `HEALTHCHECK` in Dockerfile

**Object mode**
- `TRAINING_MODE=object` — runs YOLOv9c detection on full images, crops each detected instance of a target class, and saves crops to the output volume (Frigate has no training API for objects — crops are placed manually)
- `OBJECT_CLASS` env var — target YOLO class label (e.g. `dog`, `cat`, `car`); defaults to `dog`

**People filtering**
- `ONLY_PEOPLE` env var — comma-separated whitelist; only these people are processed
- `SKIP_PEOPLE` env var — comma-separated list; these people are skipped
- `MIN_FACE_COUNT` env var — skip people with fewer than N tagged assets in Immich
- `YEARS_FILTER` env var — ignore assets older than N years (default: 10)

**Image quality controls** (previously hardcoded)
- `BLUR_THRESHOLD` env var — Laplacian variance threshold for blur rejection
- `MIN_CONFIDENCE` env var — minimum Immich face detection confidence
- `MAX_AUTO_IMAGES` env var — hard cap on auto-diversity selection
- `FACE_MARGIN` env var — padding around bounding box crops as a fraction of face size
- `USE_FULL_RESOLUTION` env var — download full-res originals vs preview thumbnails
- `ENABLE_FACE_ALIGNMENT` env var — align to ArcFace 112×112 format via InsightFace landmarks

**GPU and model configuration**
- `FORCE_CPU` env var — disable GPU; fall back to CPU for embedding computation
- `INSIGHTFACE_HOME` env var — controls model persistence for Buffalo_L
- `HF_HOME` env var — HuggingFace model cache path for SigLIP
- `LD_LIBRARY_PATH` set in the image to expose CUDA and cuDNN pip libraries so `onnxruntime-gpu` can find them at runtime

**Caching and upload tracking**
- `ENABLE_CACHE` / `CACHE_DIR` env vars — opt-in embedding cache to skip recomputation on reruns
- Per-person upload tracker persisted as JSON; prevents the same asset from being uploaded twice across runs weeks apart, even if the container is recreated
- Startup summary showing uploaded and rejected counts per person
- `LIMIT` env var — exact image count overriding `STRATEGY` preset

**CI/CD**
- `docker-publish.yml` — builds multi-arch image and pushes to GHCR on push to `main` (`:latest`) or `dev` (`:dev`)
- `release.yml` — triggered by `v*` tags or `workflow_dispatch`; creates a GitHub Release, extracts changelog notes, builds and pushes versioned image to GHCR
- `lint.yml` — runs Ruff on push/PR to `main` and `dev`
- `test.yml` — runs pytest on push/PR to `main` and `dev`
- `update-lockfile.yml` — regenerates `uv.lock` and commits it when `pyproject.toml` changes
- Dependabot: weekly grouped PRs for Python dependencies (uv ecosystem) and GitHub Actions versions

**Testing**
- 24 unit tests across four modules: `test_config`, `test_immich_api`, `test_jobs`, `test_upload_tracker`

**Documentation**
- `docs/setup.md` — step-by-step install guide with TrueNAS and bare-Docker sections
- `docs/troubleshooting.md` — common failure modes with fixes
- `docs/faq.md` — answers to questions new users will ask
- `.env.example` — copy-paste starting point with every env var and inline comments
- README rewritten: pipeline diagram, env var reference tables, scheduling behaviour, requirements

### Changed

- `cli.py` split into three focused modules — `cli.py` (entry point), `jobs.py` (configuration and strategy resolution), `executor.py` (download, crop, upload)
- Dependency management replaced with [`uv`](https://astral.sh/uv); `uv.lock` pins the full transitive graph for reproducible builds
- `compose.yml` fully annotated; all env vars documented with inline comments
- `LD_LIBRARY_PATH` extended to include both cuDNN and CUDA runtime libraries

### Fixed

- **EXIF orientation**: PIL opens JPEGs without applying rotation metadata; Immich computes face bounding boxes on orientation-corrected images, so portrait photos produced misaligned crops. `ImageOps.exif_transpose()` now normalizes orientation before any coordinate math.
- **Model persistence**: `FaceAnalysis` was initialized with `root="~/.insightface"` (hardcoded), ignoring `INSIGHTFACE_HOME`. Buffalo_L was re-downloaded into the container on every run instead of persisting to the mounted volume.
- **Upload deduplication**: `upload_to_frigate()` scanned the output directory with `os.listdir()`, picking up leftover files from previous runs and re-uploading them. Now only files created in the current run are uploaded.
- **RGBA images**: Images in RGBA mode raised an error when encoding to JPEG. All images are now converted to RGB before saving.
- **Object mode uploads**: Object mode incorrectly called the Frigate face registration API. Frigate has no API for object training data — object mode now only saves crops to disk.
- **Stale output files**: The output directory was not cleaned between runs, causing crops to accumulate. Now wiped at the start of each face-mode run.
- **JSON decode errors**: `get_people()` and `fetch_all_assets()` only caught `RequestException`, leaving `JSONDecodeError` unhandled on non-JSON 200 responses from Immich.
- **Spaces in names**: People names with spaces caused downstream errors.
- **Inconsistent headers**: Some API calls used a raw header dict instead of `get_headers()`.
- **Docker layer caching**: `uv sync` was placed after `COPY if_curator/`, so any source change invalidated the 800 MB dependency cache. Dependencies are now installed before source is copied.

### Security

- CUDA base image bumped from `nvidia/cuda:12.6.3` to `nvidia/cuda:12.9.2-cudnn-runtime-ubuntu22.04`, picking up Ubuntu security patches flagged by Dependabot.
