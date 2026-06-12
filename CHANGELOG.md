# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.8] - 2026-06-12

### Fixed

- **Dockerfile: use `add-apt-repository` with isolated GNUPGHOME**: the `curl | gpg --dearmor` approach was silently failing — gpg exits 0 on bad/empty input, leaving an invalid keyring and causing apt to skip the deadsnakes PPA entirely. Reverted to `add-apt-repository ppa:deadsnakes/ppa` with `GNUPGHOME=$(mktemp -d)` to prevent gpg from touching any pre-existing agent socket (the original QEMU crash cause).
- **arm64 base: removed `--platform=$BUILDPLATFORM`**: the Ubuntu 24.04 base for arm64 is now pulled for the target platform, so the arm64 image contains real arm64 binaries rather than amd64 binaries in an arm64 manifest.

### Changed

- **License changed from MIT to AGPLv3+**: source and network use of winnow now require derivative works to be open-sourced under the same terms.

## [0.2.7] - 2026-06-12

### Fixed

- **arm64 base reverted to Ubuntu 24.04**: Ubuntu 26.04 ships Python 3.14, not 3.13 — `python3.13` was not locatable in its repos. Reverted arm64 base to `ubuntu:24.04`.
- **Deadsnakes PPA now uses curl+gpg instead of `add-apt-repository`**: `add-apt-repository` spawns a gpg-agent which crashes under QEMU (arm64 CI). The PPA is now added by fetching the key via `curl` and piping through `gpg --dearmor` — no agent, works on both architectures.
- **PPA conditional removed**: both amd64 (Ubuntu 22.04 CUDA) and arm64 (Ubuntu 24.04) now go through the same deadsnakes install path, eliminating the per-arch branching and the `ARG TARGETARCH` dependency in RUN commands.

## [0.2.6] - 2026-06-12

### Fixed

- **Dockerfile: `ARG TARGETARCH` re-declared in each stage**: Docker's automatic platform ARGs are only in scope for `FROM` instructions, not `RUN` commands. The `$TARGETARCH` variable in the deadsnakes PPA conditional was silently empty, so the PPA was never added and `python3.13` could not be located on the Ubuntu 22.04 CUDA base. Adding `ARG TARGETARCH` at the top of both the `build` and `runtime` stage bodies fixes the amd64 build.

## [0.2.5] - 2026-06-12

### Changed

- **CUDA base upgraded to 13.3.0**: amd64 base image bumped from `nvidia/cuda:12.9.2-cudnn-runtime-ubuntu22.04` to `nvidia/cuda:13.3.0-cudnn-runtime-ubuntu22.04`. Python packages (torch, onnxruntime-gpu) still target CUDA 12.6 via pip-installed libraries; the base image upgrade requires a host GPU driver that supports CUDA 13+.
- **`torchvision>=0.27.0` added as explicit dependency**: routed through the `pytorch-cu126` index for linux/x86_64, ensuring it resolves to `0.27.0+cu126` (paired with `torch 2.12.0+cu126`) rather than being pulled from PyPI where the older `0.21.0` wheel would downgrade torch to 2.6.0.
- **`torch>=2.12.0`** floor raised from 2.6.0 to prevent silent downgrade.
- **`nvidia-cudnn-cu12>=9.0.0`** floor kept intentionally loose — torch pins an exact cuDNN ABI version (`9.10.2.21`) and must own that constraint.

## [0.2.4] - 2026-06-12

### Changed

- **Python 3.13 across all platforms**: bumped from 3.12 to 3.13. amd64 installs Python 3.13 from the deadsnakes PPA on the Ubuntu 22.04 CUDA base; arm64 uses the Python 3.13 package available natively in Ubuntu 26.04. All confirmed dependencies (`insightface 1.0.1`, `onnxruntime-gpu 1.26.0`, `torch 2.12.0+cu126`) have Python 3.13 wheels.
- **arm64 base: Ubuntu 26.04**: Python 3.12 was removed from Ubuntu 26.04's default repos; upgrading the base pulls Python 3.13 without a PPA.
- **`requires-python = ">=3.13"`** and `target-version = "py313"` updated in `pyproject.toml`.
- **CI updated to Python 3.13**: `test.yml` and `update-lockfile.yml` now install Python 3.13.
- **`uv.lock` regenerated** for Python 3.13.5.

## [0.2.3] - 2026-06-12

### Fixed

- **Clear-text API key storage**: `config.py` no longer writes `API_KEY` to `.immich_config.json`. The key must come from an environment variable or `.env` file. Interactive mode now prints a tip directing users to `.env`. Resolves CodeQL `py/clear-text-storage-sensitive-data`.

### Changed

- **CI workflow permissions**: `test.yml` and `lint.yml` now declare `permissions: contents: read`, following least-privilege principle and resolving `actions/missing-workflow-permissions` scanner alerts.
- **CI lockfile race condition**: removed the `verify-lockfile` pre-job from `docker-publish.yml` and `release.yml`. The `update-lockfile.yml` bot maintains the lockfile; the verify step raced against it on the same push event and caused false failures. `release.yml` now runs `uv lock` inline so tag-triggered builds are always self-consistent.
- **Docs moved to wiki**: `docs/` folder removed from the repository. Setup, Troubleshooting, and FAQ pages are now at the [GitHub wiki](https://github.com/sudolulo/winnow/wiki).

## [0.2.2] - 2026-06-12

### Added

- **Confidence scores in upload tracker**: Immich face confidence scores are now stored per asset in `frigate_uploaded_ids.json` under `by_person[name].scores`. Lays the groundwork for future replacement logic (remove low-confidence uploads when better images are found).
- **Frigate-authoritative capacity tracking**: At startup, `GET /api/faces` is queried on the Frigate host to retrieve the actual number of trained images per person from the `train` directory (pending/unclassified queue is excluded). This count is stored as `frigate_count` in the tracker JSON so it survives Frigate downtime.
- **Lifetime cap uses Frigate count**: `MAX_AUTO_IMAGES` is now enforced against Frigate's live training image count rather than the local uploaded-asset tally. Fallback priority: live Frigate API → last cached `frigate_count` in JSON → local uploaded count.
- **Startup summary shows Frigate count**: Tracker summary at startup now includes the last known Frigate training count per person (e.g. `78 uploaded, 2 rejected, 42 in Frigate`).
- **`winnow/frigate_api.py`**: new module encapsulating Frigate API helpers; currently exposes `get_frigate_face_counts()`.

### Changed

- `upload_tracker.py`: `by_person` entries migrated from flat list to `{asset_ids, scores, frigate_count}` dict. Old list format is read and migrated transparently on first write.
- `mark_uploaded()` now accepts an optional `score` keyword argument.
- `get_person_summary()` now returns `frigate_count` and `scores` fields alongside `uploaded` and `rejected`.

## [0.2.1] - 2026-06-12

### Fixed

- **Container startup reinstalling packages**: `entrypoint.sh` used `uv run`, which performs a sync check on every startup and re-downloaded `ruff` and rebuilt the package each time. Replaced with direct `.venv/bin/python` calls to skip the sync entirely.
- **InsightFace double `models/` path**: `INSIGHTFACE_HOME=/models` caused InsightFace to download Buffalo_L to `/models/models/buffalo_l` (InsightFace always appends `models/` to the root). Updated default in `compose.yml` and `.env.example` to `/models/.insightface`.
- **Lint errors in CI**: unused imports in `tests/test_config.py` and `tests/test_upload_tracker.py`, unsorted imports in `scheduler.py` — all would have failed the ruff CI check.

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
- `compose.yml` — fully annotated with inline comments grouped by concern
- `entrypoint.sh` — runs the tool once on startup, then hands off to the scheduler if `CRON_SCHEDULE` is set
- `scheduler.py` — in-process cron scheduler that keeps the container (and loaded models) alive between runs
- `CRON_SCHEDULE` env var — standard cron expression for recurring runs; unset exits after first run
- `.dockerignore` — keeps `.venv`, `__pycache__`, test files, and logs out of the image context
- Multi-arch image: `linux/amd64` and `linux/arm64` built and merged into a single manifest on GHCR
- `tini` as PID 1 init process for correct signal handling
- Non-root container user (`appuser`, uid 568)
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
- `docs/setup.md` — step-by-step install and GPU passthrough guide
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
