# Troubleshooting

## Container exits immediately

Check logs:
```bash
docker compose logs winnow
```

Common causes:
- **Missing required env var** — `IMMICH_URL` or `API_KEY` not set
- **Cannot reach Immich** — check the URL and that Immich is running; use `http://` not `https://` unless you have TLS set up

---

## "No people found" / nothing processed

- Make sure Immich has completed face recognition and you have named people in your library
- `YEARS_FILTER` defaults to 10 years — increase it if your tagged photos are older
- `MIN_FACE_COUNT` skips people with few photos — lower or remove it

---

## Frigate upload fails

- Confirm `FRIGATE_URL` is reachable from inside the container: `docker exec winnow curl $FRIGATE_URL/api/stats`
- Check Frigate v0.16+ — older versions don't have the face training API
- Set `DRY_RUN=true` to verify selection without uploading

---

## Models fail to download

winnow downloads InsightFace and HuggingFace (SigLIP) models on first run.

- Ensure the container has internet access
- Confirm the model volume is mounted and writable
- If behind a proxy, set `HTTP_PROXY` / `HTTPS_PROXY` env vars

---

## Running on CPU (no GPU)

Set `FORCE_CPU=true`. Everything works but embedding computation is slower — expect several minutes per person instead of seconds.

If you have a GPU but it's not being used:
- Confirm the NVIDIA container toolkit is installed: `docker run --rm --gpus all nvidia/cuda:12.9.2-base-ubuntu22.04 nvidia-smi`
- Confirm the `deploy.resources.reservations.devices` block is present in `compose.yml`

---

## Same images uploaded every run

The upload tracker is stored in `CACHE_DIR` (`/app/.if_cache` by default). If this volume isn't persisted between runs, the tracker resets and images are re-uploaded.

Make sure `/app/.if_cache` is mounted to a persistent host path.

---

## Re-uploading a specific person

To clear the upload history for one person and start fresh:

```env
RESET_PERSON=John
```

Remove this after one run — it clears the history and then processes normally.

---

## Image quality issues

- **Too blurry**: Lower `BLUR_THRESHOLD` (default 100) — e.g. `50` accepts more blur
- **Face too small**: Lower `MIN_FACE_WIDTH` (default 50px)
- **Low confidence detections included**: Raise `MIN_CONFIDENCE` (default 0.7)
- **Rejected images being re-tried**: Set `RETRY_REJECTED=true` for one run
