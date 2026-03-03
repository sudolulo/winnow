# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-03

### Added
- Initial release of `if-curator` — Immich to Frigate training set curator
- **Face recognition prep**: InsightFace (ArcFace/Buffalo_L) embeddings on face crops for Frigate face recognition training
- **Object/state classification prep**: SigLIP (Vision Transformer) embeddings with YOLOv9c object detection
- **Smart diversity selection**: K-Medoids clustering + Farthest Point Sampling (FPS) with hard-example weighting
- **Adaptive auto-threshold**: Automatically stops selection when adding more images becomes redundant (capped at 80)
- **Quality filtering**: Automatic rejection of blurry, grayscale/IR, over/underexposed, low-confidence, and too-small images
- **Face alignment**: Align faces to ArcFace 112×112 format via InsightFace landmarks
- **Full-resolution downloads**: Downloads originals for final crops (falls back to JPEG preview for HEIC/RAW)
- **Concurrent downloads**: 8 parallel thumbnail workers for performance
- **Embedding cache**: Optional disk-based cache for faster re-runs
- **Multi-person batch mode**: Process multiple people in one session
- **Interactive CLI**: Rich terminal UI with preview summary table before downloading
- **GPU support**: Optional CUDA/ROCm/MPS acceleration via `onnxruntime-gpu` extra
- **Environment variable configuration**: Full control via `.env` or shell environment
