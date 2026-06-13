#!/usr/bin/env python3
"""
winnow inference benchmark: GPU vs CPU throughput.

Measures InsightFace (face mode) and SigLIP (object mode) latency and
throughput. Run with FORCE_CPU=true for CPU-only baseline.

Usage inside container:
    # GPU mode:
    docker exec winnow python /app/scripts/benchmark.py

    # CPU mode:
    docker exec -e FORCE_CPU=true winnow python /app/scripts/benchmark.py
"""

import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw


def _mode_label() -> str:
    if os.getenv("FORCE_CPU", "").lower() in ("true", "1", "yes"):
        return "CPU (FORCE_CPU=true)"
    return "GPU (auto)"


def make_face_image(size: int = 640) -> Image.Image:
    """Synthetic face-like image: skin-tone rectangle with landmark blobs."""
    img = Image.new("RGB", (size, size), (200, 170, 140))
    draw = ImageDraw.Draw(img)
    # Head oval
    cx, cy = size // 2, size // 2
    hw, hh = int(size * 0.3), int(size * 0.38)
    draw.ellipse([cx - hw, cy - hh, cx + hw, cy + hh], fill=(220, 185, 155))
    # Eyes
    for ex in [cx - int(size * 0.1), cx + int(size * 0.1)]:
        ey = cy - int(size * 0.05)
        r = max(4, size // 40)
        draw.ellipse([ex - r, ey - r, ex + r, ey + r], fill=(40, 30, 20))
    # Nose
    draw.ellipse([cx - 5, cy + 5, cx + 5, cy + 15], fill=(180, 140, 110))
    # Mouth
    draw.arc([cx - 20, cy + 25, cx + 20, cy + 45], start=0, end=180, fill=(160, 80, 80), width=3)
    return img


def make_random_image(width: int = 224, height: int = 224) -> Image.Image:
    rng = np.random.default_rng(42)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8), "RGB")


def _stats(times_s: list[float]) -> dict:
    arr = np.array(times_s) * 1000  # ms
    return {
        "median_ms": float(np.median(arr)),
        "mean_ms": float(np.mean(arr)),
        "min_ms": float(np.min(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "ips": 1000.0 / float(np.median(arr)),
    }


def bench_insightface(n_warmup: int = 5, n_runs: int = 30) -> None:
    import cv2

    import winnow.embeddings as emb_mod
    from winnow.embeddings import get_insightface_app

    # Reset singleton so we get a fresh load
    emb_mod._insightface_app = None
    emb_mod._insightface_loaded = False

    print("  Loading model...")
    t_load = time.perf_counter()
    app = get_insightface_app()
    load_s = time.perf_counter() - t_load

    if app is None:
        print("  SKIP: InsightFace failed to load")
        return

    img_pil = make_face_image(640)
    img_bgr = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)

    # Warmup
    for _ in range(n_warmup):
        app.get(img_bgr)

    # Timed — single image 640×640
    times: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        app.get(img_bgr)
        times.append(time.perf_counter() - t0)

    s = _stats(times)
    print(f"  Model load time : {load_s:.2f} s")
    print("  Input size      : 640×640")
    print(f"  Runs            : {n_runs} (after {n_warmup} warmup)")
    print(f"  Median latency  : {s['median_ms']:.1f} ms")
    print(f"  Mean / p95      : {s['mean_ms']:.1f} ms / {s['p95_ms']:.1f} ms")
    print(f"  Min latency     : {s['min_ms']:.1f} ms")
    print(f"  Throughput      : {s['ips']:.1f} images/s")

    # Also test at 320×320
    img_sm = make_face_image(320)
    img_sm_bgr = cv2.cvtColor(np.asarray(img_sm), cv2.COLOR_RGB2BGR)
    for _ in range(n_warmup):
        app.get(img_sm_bgr)
    times_sm: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        app.get(img_sm_bgr)
        times_sm.append(time.perf_counter() - t0)
    s2 = _stats(times_sm)
    print(f"  320×320 median  : {s2['median_ms']:.1f} ms  ({s2['ips']:.1f} img/s)")


def bench_siglip(
    n_warmup: int = 3,
    n_runs: int = 20,
    batch_sizes: tuple = (1, 4, 8, 16, 32),
) -> None:
    import torch

    import winnow.embeddings as emb_mod
    emb_mod._siglip_model = None
    emb_mod._siglip_processor = None
    emb_mod._siglip_loaded = False

    print("  Loading model...")
    t_load = time.perf_counter()
    model, processor = emb_mod.get_siglip_model()
    load_s = time.perf_counter() - t_load

    if model is None:
        print("  SKIP: SigLIP failed to load")
        return

    device = next(model.parameters()).device
    print(f"  Model load time : {load_s:.2f} s  (device: {device})")

    print(f"  {'Batch':>5}  {'ms/batch':>10}  {'ms/img':>8}  {'img/s':>8}  {'p95/img':>9}")
    for bs in batch_sizes:
        imgs = [make_random_image(224, 224) for _ in range(bs)]
        inputs = processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Warmup
        for _ in range(n_warmup):
            with torch.no_grad():
                model(**inputs)
        if str(device) != "cpu":
            torch.cuda.synchronize()

        times: list[float] = []
        for _ in range(n_runs):
            if str(device) != "cpu":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                model(**inputs)
            if str(device) != "cpu":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        s = _stats(times)
        print(
            f"  {bs:>5}  {s['median_ms']:>10.1f}  {s['median_ms']/bs:>8.2f}"
            f"  {bs * 1000 / s['median_ms']:>8.1f}  {s['p95_ms']/bs:>9.2f}"
        )


def main() -> None:
    print("=" * 56)
    print("  winnow inference benchmark")
    print(f"  Mode: {_mode_label()}")
    print("=" * 56)
    print()

    print("── InsightFace Buffalo_L  (face detection + ArcFace) ──")
    bench_insightface()
    print()

    print("── SigLIP google/siglip-base-patch16-224  (objects) ───")
    bench_siglip()
    print()


if __name__ == "__main__":
    # Add winnow to path when run directly inside container
    sys.path.insert(0, "/app")
    main()
