# FAQ

## Does winnow modify my Immich library?

No. winnow only reads from Immich (assets, people, face bounding boxes). It never writes back to Immich or deletes anything.

---

## How many images should I upload to Frigate?

The `auto` strategy decides this for you — it keeps selecting until adding more images would be redundant. In practice this is usually 20–60 per person. You can cap it with `MAX_AUTO_IMAGES` (default 80).

Quality and diversity matter far more than volume. 30 well-spread images outperform 200 from the same week.

---

## What's the difference between face mode and object mode?

- **Face mode**: Extracts and aligns face crops, uploads them directly to Frigate's face training API. This is for teaching Frigate to recognize specific people.
- **Object mode**: Runs YOLO detection on full images and saves crops of a target class (dog, cat, car, etc.) to disk. Frigate has no API for object training data, so you place them manually.

---

## Can I run it without Frigate?

Yes — in object mode, `FRIGATE_URL` is not used and crops are saved to the output volume. In face mode you need Frigate to receive the uploads, but you can use `DRY_RUN=true` to preview selection without uploading.

---

## How does auto-diversity mode work?

winnow computes a vector embedding for each candidate image (what the face/object actually looks like — angle, lighting, expression). It then clusters those embeddings and picks representatives that are maximally spread across the embedding space. It stops when the next-most-different image is already close to something already selected. See the README for the full pipeline.

---

## Does it support multiple people in one run?

Yes. By default it processes every named person in your Immich library. Use `ONLY_PEOPLE` to whitelist specific names or `SKIP_PEOPLE` to exclude them.

---

## What GPU is needed?

Any NVIDIA GPU with CUDA 12.x support. The models (InsightFace Buffalo_L + SigLIP) fit comfortably in 4 GB VRAM. CPU mode works but is significantly slower.

ARM builds (linux/arm64) use CPU-only — CUDA is not available on ARM.

---

## Does it work on Unraid / Proxmox / bare Docker?

Yes — the `compose.yml` uses standard Docker volume mounts. Replace the example paths with whatever absolute paths suit your setup.

---

## How do I update winnow?

```bash
docker compose pull
docker compose up -d
```

The `latest` tag on GHCR tracks the `main` branch. Pinning to a version tag (e.g. `ghcr.io/sudolulo/winnow:v0.2.0`) is recommended for stability.
