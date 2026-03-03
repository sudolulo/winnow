<div align="center">

# 🖼️ if-curator
### Immich to Frigate Curator

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Immich](https://img.shields.io/badge/Immich-v1.106%2B-violet?style=for-the-badge)](https://immich.app)
[![Frigate](https://img.shields.io/badge/Frigate-Ready-green?style=for-the-badge)](https://frigate.video)

*A specialized tool to extract **high-quality, diverse** training images from your Immich library for Frigate's Face Recognition (ArcFace) and Object/State Classification models.*

</div>

> [!WARNING]
> **Regarding Object Classification**
>
> Frigate **does not support** uploading custom images for object classification training via the UI or API.
> This tool currently prepares the dataset (crops and categorizes images) for training external models (like YOLO) manually.

---

## ⚡ Why This Tool?

> **"Diversity matters far more than volume."** — *Frigate Developer Tips*

Training AI models on "bulk" data is often harmful. If you feed the model 50 images from the same 10-second video clip, it learns to recognize the *lighting and background*, not the actual *face* or *object*.

`if-curator` solves this using **AI-powered diversity selection** and **quality filtering**:

| Mode | Embedding Model | Algorithm |
| :--- | :--- | :--- |
| **👤 Face** | InsightFace (ArcFace) | K-Medoids Clustering + FPS + Hard Example Weighting |
| **🐶 Object** | SigLIP (Vision Transformer) | K-Medoids Clustering + FPS |

The pipeline uses **K-Medoids clustering** to guarantee coverage of every distinct "look", then fills the remaining budget with **Farthest Point Sampling (FPS)** biased toward **hard examples** (unusual angles, partial occlusions). An **adaptive threshold** stops selection automatically when adding more images becomes redundant.

---

## ✨ Features

### 🎯 Smart Selection
- **Auto Diversity [Recommended]**: Clusters images by visual similarity, selects representatives from each cluster, then fills with maximally-diverse picks until redundancy starts (capped at 80)
- **Standard (30 images)**: Balanced set using Smart Diversity
- **Broad (100 images)**: Extensive set using Smart Diversity
- **Custom Count**: You choose the limit

### � Quality Filtering
Bad training data hurts ArcFace models. Images are automatically rejected if they are:
- **Blurry** — Laplacian variance below threshold
- **Grayscale / IR** — ArcFace is trained on color images only
- **Over/Underexposed** — Washed-out or too dark to use
- **Low confidence** — Partial or occluded face detections
- **Too small** — Faces under 100px (configurable) lack features

### 👤 Face Recognition Prep
- Uses **InsightFace** (ArcFace/Buffalo_L) embeddings on **face crops** (not full images — avoids wrong-face in group photos)
- **Hard example prioritization** — unusual angles, sunglasses, and low-confidence detections are biased for selection
- **Face alignment** via InsightFace landmarks (standard 112×112 ArcFace input)
- Downloads **full-resolution** originals for final crops (falls back to JPEG preview for HEIC/RAW)
- Configurable crop margin (default 15%)

### 📦 Object/State Classification Prep
- Uses **SigLIP** (Vision Transformer) embeddings for semantic diversity
- **YOLOv9c** to detect and crop specific objects (dogs, cars, etc.)
- Captures variation in poses, lighting, and backgrounds

### ⚡ Performance
- **Concurrent thumbnail downloads** (8 parallel workers)
- **Batch-capable** SigLIP embeddings for GPU efficiency
- Optional **disk-based embedding cache** for faster re-runs
- **Multi-person batch mode** — process multiple people in one session

### 📋 Preview Before Download
After selection, a summary table shows what will be processed:
```
                📋 Training Job Preview
┏━━━━━━━━━━━┳━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Person    ┃ Mode ┃ Images ┃ Date Range              ┃
┡━━━━━━━━━━━╇━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Sebastian │ face │     80 │ 2021-04-03 → 2026-02-18 │
└───────────┴──────┴────────┴─────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites
- **Python 3.12+**
- **[uv](https://astral.sh/uv/)** (highly recommended)
- **Immich Server** (v1.106+)

### Setup

```bash
git clone <repository_url>
cd if-curator
uv sync
```

### 🏎️ GPU Support (Recommended)
For faster embedding computation, install with GPU extras:

```bash
uv sync --extra gpu
```
*Automatically detects CUDA (NVIDIA), ROCm (AMD), or MPS (macOS).*

---

## 💻 Usage

```bash
uv run if-curator
```

### Interactive Flow
The tool will guide you through:
1.  **Select Person** — Choose from your Immich people (supports multi-person batch)
2.  **Training Mode** — Face (Recognition) or Object (Classification)
3.  **Strategy** — Auto, Standard, Broad, or Custom
4.  **Preview** — Review the selection summary before downloading
5.  **Execute** — Downloads and processes images with progress tracking

```text
Using InsightFace (face embeddings) for diversity analysis...
Quality filtering removed 76 images.
Adaptive threshold: 0.1721 (median_dist=0.8605, fraction=0.2)
Clustering 223 embeddings into 20 groups (K-Medoids)...
Selected 20 cluster medoids as initial picks.
Selection complete: 80 images (0 hard examples with confidence < 0.85).
```

---

## 🛠️ Configuration

The tool prompts for your Immich URL and API Key on the first run and saves them to `.immich_config.json`.

### Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `IMMICH_URL` | — | Full URL to Immich (e.g. `http://192.168.1.10:2283`) |
| `API_KEY` | — | Your Immich API Key |
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

## 🧠 Technical Details

### Models
- **InsightFace (Buffalo_L)** — Face detection and embedding (ArcFace, 512-d)
- **SigLIP** — Visual embeddings via `transformers` (google/siglip-base-patch16-224, 768-d)
- **YOLOv9c** — Object detection for cropping

### Algorithms
- **K-Medoids Clustering** — Groups embeddings into k clusters using cosine distance, selecting actual data points (medoids) as cluster centers. Guarantees one representative from every distinct "look"
- **Farthest Point Sampling** — After medoid selection, fills remaining budget by iteratively selecting the most distant point from the current set
- **Hard Example Weighting** — Candidates with detection confidence < 0.85 get a 1.2–1.5× distance boost, biasing selection toward challenging images (unusual angles, occlusions)
- **Adaptive Auto-Threshold** — Computed as 20% of the median pairwise cosine distance; stops when the next-best image is too similar
- **Quality Filtering** — Blur (Laplacian), grayscale/IR (channel comparison), exposure (histogram), confidence (Immich metadata)
- **Face Crop Embedding** — Extracts the target person's face (using Immich bbox) before embedding, preventing wrong-face selection in group photos

### Architecture
```
Immich API ─► Fetch Assets by Person ─► Time Filter
                                            │
                                  Concurrent Thumbnail Download (8 workers)
                                            │
                                  Quality Filtering (blur, IR, exposure...)
                                            │
                                  Face Crop Extraction (bbox from Immich metadata)
                                            │
                                  Compute Embeddings (InsightFace / SigLIP)
                                            │
                                  K-Medoids Clustering → FPS + Hard Example Weighting
                                            │
                                  Preview Summary Table
                                            │
                                  Download Full-Res ─► Face Alignment ─► Save
```
