"""Execution phase: image processing and Frigate upload."""

import logging
import os
import shutil
import time
from io import BytesIO
from urllib.parse import quote

import requests
from PIL import Image
from rich import print as rprint
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from .config import Config, get_headers
from .frigate_api import delete_frigate_person_files, get_frigate_person_files, recognize_face
from .image_processing import process_face_mode, process_full_mode, process_object_mode
from .immich_api import fetch_face_data, fetch_full_image
from .log_config import console
from .quality import assess_quality
from .upload_tracker import (
    get_frigate_filename_for_asset,
    get_lowest_quality_mapped_file,
    get_min_frigate_score,
    get_tracked_frigate_file_count,
    get_tracked_frigate_filenames,
    has_frigate_scores,
    mark_rejected,
    mark_uploaded,
    reclassify_as_rejected,
    record_frigate_file,
    remove_frigate_file,
)

logger = logging.getLogger(__name__)


def _reconcile_frigate_mappings(
    person_name: str,
    known_files_before: set[str],
    uploaded: list[tuple[str, str | None]],
) -> None:
    """Map Frigate filenames to asset IDs after a batch of uploads.

    Polls until all expected new files appear in the Frigate API, then maps
    them to asset IDs by filename timestamp order (Frigate processes the
    upload queue in FIFO order, so earlier uploads get earlier timestamps).

    KNOWN LIMITATION — race condition with external uploads:
    If another client uploads a face file for this person concurrently, the
    count of new files will exceed `len(uploaded)` and we bail out entirely
    (the "> target" branch). That's safe — we never record a wrong mapping —
    but those uploads become permanently unmapped (they won't be eligible for
    quality replacement). The right fix is a Frigate API that returns the
    filename in the upload response, removing the need for any post-upload
    diffing. Until then, the external-upload guard keeps mappings correct at
    the cost of occasionally missing them when another client is active.
    """
    target = len(uploaded)
    current_files: set[str] = set()

    for delay in (1, 2, 4, 8):
        time.sleep(delay)
        fresh = get_frigate_person_files(person_name)
        if fresh is None:
            logger.warning(
                f"{person_name}: Frigate API unreachable during mapping reconciliation"
                " — quality replacement won't target these files"
            )
            return
        current_files = set(fresh)
        if len(current_files - known_files_before) >= target:
            break

    new_files = current_files - known_files_before

    if len(new_files) == target:
        def _ts(fname: str) -> float:
            try:
                return float(fname.rsplit("_", 1)[-1].replace(".webp", ""))
            except (ValueError, IndexError):
                return 0.0

        for (fname, asset_id), frigate_file in zip(uploaded, sorted(new_files, key=_ts)):
            if asset_id:
                record_frigate_file(person_name, frigate_file, asset_id)
        logger.debug(f"{person_name}: batch-mapped {target} Frigate file(s)")
    elif len(new_files) > target:
        logger.info(
            f"{person_name}: {len(new_files)} new Frigate files for {target} uploads"
            " (external upload detected) — skipping file mapping"
        )
    else:
        logger.warning(
            f"{person_name}: only {len(new_files)} of {target} expected Frigate files"
            " appeared after reconciliation — mapping skipped"
        )


def _enrich_asset_with_face_data(asset: dict, person: dict) -> dict:
    """Enrich an asset dict with face bounding box data from the Immich faces API.

    The search/metadata endpoint does not include face bounding box data,
    so we fetch it from GET /api/faces?id={asset_id} and inject it into
    the asset's "people" field so process_face_mode can find it.

    Returns the enriched asset dict (modifies in place and returns it).
    """
    person_id = person["id"]
    face_data = fetch_face_data(asset["id"], person_id=person_id)

    if face_data is None:
        logger.debug(f"No face data returned for {person.get('name')} in asset {asset.get('id')}")
        # Clean any None entries from the people list (can come from Immich API)
        if "people" in asset:
            asset["people"] = [p for p in asset["people"] if p is not None]
        return asset

    # Skip zero-area bounding boxes (face detection failed or no face found)
    if face_data.bbox == (0, 0, 0, 0):
        logger.debug(f"Zero-area bounding box for {person.get('name')} in asset {asset.get('id')}")
        # Clean any None entries from the people list (can come from Immich API)
        if "people" in asset:
            asset["people"] = [p for p in asset["people"] if p is not None]
        return asset

    face_info = {
        "boundingBoxX1": face_data.bbox[0],
        "boundingBoxY1": face_data.bbox[1],
        "boundingBoxX2": face_data.bbox[2],
        "boundingBoxY2": face_data.bbox[3],
        "imageWidth": face_data.image_width,
        "imageHeight": face_data.image_height,
    }

    # Inject into asset so process_face_mode can find it via asset["people"]
    asset["people"] = [{"id": person_id, "faces": [face_info]}]
    asset["face_confidence"] = face_data.confidence
    return asset


def execute_jobs(jobs: list[dict]) -> None:
    """Download and process images for all jobs.

    Builds an asset_map per job (filename → Immich asset ID) so that
    upload_to_frigate() can mark assets as uploaded after success.
    """
    if not jobs:
        return

    console.rule("[bold blue]Execution Phase")

    use_full_res = Config.USE_FULL_RESOLUTION

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        grand_total = sum(j["limit"] for j in jobs)
        overall_task = progress.add_task("[green]Overall Progress", total=grand_total)

        for job in jobs:
            person, assets, config = job["person"], job["assets"], job["config"]
            name, mode = person["name"], config.get("mode", "face")

            job_task = progress.add_task(f"Processing {name}...", total=len(assets))
            person_dir = os.path.join(Config.OUTPUT_DIR, name)
            # Face crops are transient (uploaded then discarded); wipe before each run.
            # Object crops are the deliverable; preserve them across runs.
            if mode == "face" and os.path.isdir(person_dir):
                shutil.rmtree(person_dir)
            os.makedirs(person_dir, exist_ok=True)

            # Track filename → asset_id, filename → confidence score, filename → crop dims
            asset_map: dict[str, str] = {}
            score_map: dict[str, float | None] = {}
            dims_map: dict[str, tuple[int, int]] = {}

            count = 0
            for asset in assets:
                try:
                    # For face mode, enrich the asset with face bounding box data
                    # from the Immich faces API (not included in search/metadata results)
                    if mode == "face":
                        asset = _enrich_asset_with_face_data(asset, person)

                    # Use full-resolution for final output when configured
                    if use_full_res:
                        img = fetch_full_image(asset["id"])
                    else:
                        resp = requests.get(
                            f"{Config.IMMICH_URL}/api/assets/{asset['id']}/thumbnail?size=preview&format=JPEG",
                            headers=get_headers(),
                            timeout=30,
                        )
                        img = Image.open(BytesIO(resp.content)) if resp.ok else None

                    if img is None:
                        progress.console.print(f"[red]Failed download {asset['id']}[/red]")
                    else:
                        saved = (
                            process_face_mode(img, asset, person, person_dir, count)
                            if mode == "face"
                            else process_object_mode(img, config, person_dir, count)
                            if mode == "object"
                            else process_full_mode(img, person_dir, count)
                        )
                        if saved:
                            # Record which asset produced which output file
                            filename = f"{count}.jpg"
                            asset_map[filename] = asset["id"]
                            score_map[filename] = asset.get("quality_score")
                            if mode == "face" and isinstance(saved, tuple):
                                dims_map[filename] = saved
                            # Time-spread path: compute blur score from the downloaded
                            # image. Cap at 1440px so the scale matches the preview
                            # thumbnails the embedding path uses for scoring — Laplacian
                            # variance grows with resolution, making full-res and
                            # thumbnail scores incomparable if left uncapped.
                            if mode == "face" and score_map[filename] is None:
                                try:
                                    score_img = img.convert("RGB") if img.mode != "RGB" else img
                                    if score_img.width > 1440 or score_img.height > 1440:
                                        score_img = score_img.copy()
                                        score_img.thumbnail((1440, 1440), Image.LANCZOS)
                                    score_map[filename] = assess_quality(score_img).blur_score
                                except Exception as exc:
                                    logger.debug(f"Quality score fallback for {asset['id']}: {exc}")
                                    score_map[filename] = 0.0  # unknown quality — treat as lowest
                            # Also record object-mode variant filenames
                            if mode == "object":
                                for f in sorted(os.listdir(person_dir)):
                                    if f.startswith(f"{count}_") and f not in asset_map:
                                        asset_map[f] = asset["id"]
                                        score_map[f] = asset.get("face_confidence")

                            count += 1
                        else:
                            progress.console.print(
                                f"[yellow]Skipped {asset['id']} (no usable face data)[/yellow]"
                            )
                except Exception as e:
                    logger.error(f"Failed to process asset {asset['id']}: {e}")

                progress.advance(job_task)
                progress.advance(overall_task)

            # Store maps on the job so upload_to_frigate can use them
            job["asset_map"] = asset_map
            job["score_map"] = score_map
            job["dims_map"] = dims_map

            progress.remove_task(job_task)

            # Log how many images were actually saved vs selected
            if count < len(assets):
                logger.info(f"{name}: saved {count}/{len(assets)} selected images")


def upload_to_frigate(jobs: list[dict]) -> None:
    """Upload processed face crops to Frigate via API with detailed logging.

    Only runs for face-mode jobs. Object-mode crops are saved to the output
    directory as the deliverable and must be copied to Frigate manually.

    After each successful upload, records the Immich asset ID in the
    upload tracker so it is skipped on future runs.
    """
    face_jobs = [j for j in jobs if j["config"].get("mode", "face") == "face"]

    if not face_jobs:
        rprint("[dim]No face-mode jobs to upload.[/dim]")
        return

    # Notify user about object-mode jobs that were skipped
    object_jobs = [j for j in jobs if j["config"].get("mode") == "object"]
    for job in object_jobs:
        name = job["person"]["name"]
        person_dir = os.path.join(Config.OUTPUT_DIR, name)
        rprint(f"  [dim]📁 {name} (object): crops saved to {person_dir} — copy to Frigate manually[/dim]")

    frigate_url = os.environ.get("FRIGATE_URL", "")
    if not frigate_url:
        rprint("[yellow]⚠️  FRIGATE_URL not set, skipping upload.[/yellow]")
        return

    rprint("\n[bold cyan]📤 Uploading to Frigate[/bold cyan]")
    rprint(f"  Target: [dim]{frigate_url}[/dim]")

    # Build a mapping of output filenames → Immich asset IDs
    # from the asset_map stored on each job during execute_jobs()
    filename_to_asset_id: dict[str, dict[str, str]] = {}
    total_files = 0
    for job in face_jobs:
        name = job["person"]["name"]
        asset_map = job.get("asset_map", {})
        filename_to_asset_id[name] = asset_map
        total_files += len(asset_map)

    if total_files == 0:
        rprint("  [yellow]No images found to upload.[/yellow]")
        return

    rprint(f"  People: [bold]{len(face_jobs)}[/bold], Total images: [bold]{total_files}[/bold]")

    uploaded, failed, gate_total = 0, 0, 0
    max_retries = 2

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        upload_task = progress.add_task("[green]Uploading to Frigate", total=total_files)

        for job in face_jobs:
            name = job["person"]["name"]
            # URL-encode the name for the API (handles spaces, special chars)
            encoded_name = quote(name, safe="")
            if " " in name:
                progress.console.print(f"  ℹ️  URL-encoded name for Frigate API: '{name}' → '{encoded_name}'")

            person_dir = os.path.join(Config.OUTPUT_DIR, name)
            if not os.path.isdir(person_dir):
                progress.console.print(f"  [dim]⏭️  {name}: no output directory, skipping[/dim]")
                continue

            asset_map = filename_to_asset_id.get(name, {})
            score_map = job.get("score_map", {})
            dims_map = job.get("dims_map", {})
            person_files = sorted(asset_map.keys())

            if not person_files:
                progress.console.print(f"  [dim]⏭️  {name}: no images found[/dim]")
                continue

            progress.console.print(f"  📁 {name}: uploading {len(person_files)} image(s)...")
            person_uploaded = 0
            person_failed = 0

            # Snapshot live Frigate files for post-upload reconciliation diff only.
            # effective_count is sourced from the tracker (mapped files) so that
            # manually-added Frigate files don't consume winnow's managed quota.
            _snapshot = get_frigate_person_files(name)
            if _snapshot is None:
                # Frigate GET is down; fall back to the tracker's mapped filenames
                # as the pre-upload baseline. reconciliation will still work unless
                # there are concurrent manual uploads (handled by >target guard).
                logger.warning(
                    f"{name}: Frigate API unreachable at upload start"
                    " — using tracker baseline for post-upload reconciliation"
                )
                known_frigate_files_at_start: set[str] = get_tracked_frigate_filenames(name)
            else:
                known_frigate_files_at_start: set[str] = set(_snapshot)
                # Remove tracker mappings for files that no longer exist in Frigate
                # (manually deleted, or cleaned up outside winnow). This corrects the
                # effective_count so those slots are available for new uploads.
                stale = get_tracked_frigate_filenames(name) - known_frigate_files_at_start
                for stale_fn in stale:
                    remove_frigate_file(name, stale_fn)
                if stale:
                    progress.console.print(
                        f"  [dim]{name}: cleared {len(stale)} stale mapping(s)"
                        " (file(s) no longer in Frigate)[/dim]"
                    )
            effective_count = get_tracked_frigate_file_count(name)
            pre_run_count = effective_count
            quality_replacement = job.get("config", {}).get("quality_replacement", False)
            # Dynamic floor is always active once scores exist — new images must score
            # at least as well as the weakest image already in the set.
            # FRIGATE_SCORE_THRESHOLD adds an explicit absolute minimum on top.
            _dynamic = get_min_frigate_score(name)
            effective_threshold = max(Config.FRIGATE_SCORE_THRESHOLD, _dynamic or 0.0)
            if _dynamic is not None and pre_run_count > 0:
                if Config.FRIGATE_SCORE_THRESHOLD > 0 and _dynamic > Config.FRIGATE_SCORE_THRESHOLD:
                    progress.console.print(
                        f"  [dim]{name}: quality gate floor raised to {_dynamic:.2f}"
                        f" (min stored score, above configured {Config.FRIGATE_SCORE_THRESHOLD:.2f})[/dim]"
                    )
                elif Config.FRIGATE_SCORE_THRESHOLD == 0:
                    progress.console.print(
                        f"  [dim]{name}: quality gate active at {_dynamic:.2f} (min stored score)[/dim]"
                    )
            if Config.ENABLE_FRIGATE_SCORES and pre_run_count == 0:
                progress.console.print(
                    f"  [dim]{name}: first run — quality gate will apply from the next run[/dim]"
                )
            actually_uploaded: list[tuple[str, str | None]] = []
            failed_deletes: set[str] = set()
            quality_gate_failed: set[str] = set()
            min_quality_score_for_slot: float | None = None

            for fname in person_files:
                fpath = os.path.join(person_dir, fname)

                # If a previous replacement delete succeeded but that upload failed,
                # require the next candidate to beat the deleted file's score so the
                # freed slot isn't filled with something worse than what we removed.
                if min_quality_score_for_slot is not None:
                    file_score = score_map.get(fname)
                    if file_score is None or file_score <= min_quality_score_for_slot:
                        score_str = f"{file_score:.3f}" if file_score is not None else "N/A"
                        progress.console.print(
                            f"    [dim]⏭  {fname}: score {score_str} ≤ freed slot floor"
                            f" {min_quality_score_for_slot:.3f}, skipping[/dim]"
                        )
                        progress.advance(upload_task)
                        continue

                at_cap = effective_count >= Config.MAX_AUTO_IMAGES
                if at_cap:
                    if not quality_replacement:
                        progress.console.print(f"    [dim]⏭  {fname}: at cap, quality replacement disabled[/dim]")
                        progress.advance(upload_task)
                        continue
                    using_fscore = has_frigate_scores(name) and Config.ENABLE_FRIGATE_SCORES
                    if using_fscore:
                        candidate_score = recognize_face(fpath)
                        if candidate_score is None:
                            progress.console.print(
                                f"    [dim]⏭  {fname}: Frigate recognize unavailable, skipping replacement[/dim]"
                            )
                            progress.advance(upload_task)
                            continue
                        score_label = "frigate"
                    else:
                        candidate_score = score_map.get(fname)
                        if candidate_score is None:
                            progress.console.print(
                                f"    [dim]⏭  {fname}: no quality score, skipping replacement[/dim]"
                            )
                            progress.advance(upload_task)
                            continue
                        score_label = "blur"
                    # Skip replacement if candidate would fail the quality gate —
                    # deleting the worst then gating the new one is a net slot loss.
                    if using_fscore and effective_threshold > 0 and pre_run_count > 0 and candidate_score < effective_threshold:
                        progress.console.print(
                            f"    [dim]⏭  {fname}: frigate {candidate_score:.3f} below gate threshold"
                            f" {effective_threshold:.2f}, skipping replacement[/dim]"
                        )
                        progress.advance(upload_task)
                        continue
                    worst = get_lowest_quality_mapped_file(name, exclude=failed_deletes)
                    if worst is None or candidate_score <= worst[2]:
                        worst_score_str = f"{worst[2]:.3f}" if worst is not None else "N/A"
                        progress.console.print(
                            f"    [dim]⏭  {fname}: {score_label} {candidate_score:.3f} ≤ worst"
                            f" {worst_score_str}, skipping[/dim]"
                        )
                        progress.advance(upload_task)
                        continue
                    worst_frigate_file, _worst_asset_id, worst_score = worst
                    progress.console.print(
                        f"    🔄 {fname}: {score_label} {candidate_score:.3f} > {worst_score:.3f},"
                        f" replacing {worst_frigate_file}"
                    )
                    if delete_frigate_person_files(name, [worst_frigate_file]):
                        remove_frigate_file(name, worst_frigate_file)
                        effective_count -= 1
                        # Slot floor guard uses blur scores only — frigate_score mode
                        # will re-evaluate the next candidate via recognize_face anyway.
                        min_quality_score_for_slot = score_map.get(fname) if not using_fscore else None
                    else:
                        logger.warning(f"Failed to delete {worst_frigate_file} for {name}, skipping replacement")
                        failed_deletes.add(worst_frigate_file)
                        progress.advance(upload_task)
                        continue

                for attempt in range(1, max_retries + 1):
                    try:
                        with open(fpath, "rb") as f:
                            resp = requests.post(
                                f"{frigate_url}/api/faces/{encoded_name}/register",
                                files={"file": (fname, f, "image/jpeg")},
                                timeout=30,
                            )
                        if resp.status_code == 200:
                            uploaded += 1
                            person_uploaded += 1
                            effective_count += 1
                            min_quality_score_for_slot = None

                            asset_id = asset_map.get(fname)
                            if asset_id:
                                post_fscore = recognize_face(fpath) if Config.ENABLE_FRIGATE_SCORES else None
                                mark_uploaded(
                                    asset_id,
                                    person_name=name,
                                    score=score_map.get(fname),
                                    crop_dims=dims_map.get(fname),
                                    frigate_score=post_fscore,
                                )
                                actually_uploaded.append((fname, asset_id))

                                # Flag for post-reconcile removal if below threshold.
                                # We don't know the Frigate filename yet — reconcile maps
                                # it first, then we delete using the mapped name.
                                if (
                                    effective_threshold > 0
                                    and pre_run_count > 0
                                    and post_fscore is not None
                                    and post_fscore < effective_threshold
                                ):
                                    quality_gate_failed.add(asset_id)
                                    progress.console.print(
                                        f"    [yellow]⚠  {fname}: Frigate score {post_fscore:.2f}"
                                        f" < threshold {effective_threshold:.2f}, will remove after mapping[/yellow]"
                                    )

                            break
                        else:
                            if attempt < max_retries:
                                logger.warning(
                                    f"Upload attempt {attempt}/{max_retries} for {fname}:"
                                    f" HTTP {resp.status_code}, retrying..."
                                )
                                continue
                            failed += 1
                            person_failed += 1
                            progress.console.print(
                                f"    [red]✗ {fname}: HTTP {resp.status_code} (after {max_retries} attempts)[/red]"
                            )
                            try:
                                error_detail = resp.json().get("message", resp.text[:100])
                                progress.console.print(f"      [dim]{error_detail}[/dim]")
                            except Exception:
                                error_detail = resp.text[:100]
                                progress.console.print(f"      [dim]{error_detail}[/dim]")
                            if resp.status_code == 400 and "face" in error_detail.lower():
                                asset_id = asset_map.get(fname)
                                if asset_id:
                                    mark_rejected(asset_id, person_name=name)
                    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                        if attempt < max_retries:
                            logger.warning(
                                f"Upload attempt {attempt}/{max_retries} for {fname}:"
                                f" {type(exc).__name__}, retrying..."
                            )
                            continue
                        failed += 1
                        person_failed += 1
                        label = (
                            "Connection refused"
                            if isinstance(exc, requests.exceptions.ConnectionError)
                            else "Request timed out (30s)"
                        )
                        progress.console.print(
                            f"    [red]✗ {fname}: {label} (after {max_retries} attempts)[/red]"
                        )
                    except Exception as e:
                        if attempt < max_retries:
                            logger.warning(
                                f"Upload attempt {attempt}/{max_retries} for {fname}:"
                                f" {type(e).__name__}, retrying..."
                            )
                            continue
                        failed += 1
                        person_failed += 1
                        progress.console.print(
                            f"    [red]✗ {fname}: {type(e).__name__} - {e} (after {max_retries} attempts)[/red]"
                        )

                progress.advance(upload_task)

            if min_quality_score_for_slot is not None:
                logger.warning(
                    f"{name}: freed replacement slot (floor {min_quality_score_for_slot:.3f})"
                    " was not filled this run — will be available next run"
                )

            # Batch-map Frigate filenames to asset IDs now that all uploads are done.
            if actually_uploaded:
                _reconcile_frigate_mappings(name, known_frigate_files_at_start, actually_uploaded)

            # Post-reconcile quality gate: filenames are now mapped, so we can delete.
            gate_removed = 0
            if quality_gate_failed:
                to_delete: list[tuple[str, str]] = []  # (frigate_fn, asset_id)
                for asset_id in quality_gate_failed:
                    frigate_fn = get_frigate_filename_for_asset(name, asset_id)
                    if frigate_fn:
                        to_delete.append((frigate_fn, asset_id))
                    else:
                        logger.warning(
                            f"{name}: could not remove low-score file for {asset_id}"
                            " — no Frigate filename mapped (reconciliation race?)"
                        )
                if to_delete:
                    if delete_frigate_person_files(name, [fn for fn, _ in to_delete]):
                        for frigate_fn, aid in to_delete:
                            remove_frigate_file(name, frigate_fn)
                            reclassify_as_rejected(aid, name)
                        gate_removed = len(to_delete)
                        effective_count -= gate_removed
                        gate_total += gate_removed
                    else:
                        logger.warning(
                            f"{name}: batch delete of {len(to_delete)} low-score file(s) failed"
                        )

            # Per-person summary
            if person_failed == 0 and gate_removed == 0:
                progress.console.print(
                    f"  ✅ {name}: {person_uploaded}/{person_uploaded} uploaded"
                )
            elif person_failed == 0:
                net = person_uploaded - gate_removed
                progress.console.print(
                    f"  [yellow]✅ {name}: {person_uploaded} uploaded,"
                    f" {gate_removed} removed by quality gate (score < {effective_threshold:.2f})"
                    f" → {net} net[/yellow]"
                )
            else:
                gate_note = f", {gate_removed} removed by quality gate" if gate_removed else ""
                progress.console.print(
                    f"  ⚠️  {name}: {person_uploaded} succeeded, {person_failed} failed{gate_note}"
                )

    # Grand summary
    rprint("\n  [bold]Frigate Upload Summary:[/bold]")
    rprint(f"    ✅ Succeeded: [green]{uploaded}[/green]")
    if failed:
        rprint(f"    ❌ Failed:    [red]{failed}[/red]")
    else:
        rprint("    ❌ Failed:    0")
    if gate_total:
        rprint(f"    🗑  Removed (quality gate): [yellow]{gate_total}[/yellow]")

    if failed > 0:
        rprint("  [yellow]Check logs above for per-file error details.[/yellow]")

    if failed == total_files and total_files > 0:
        rprint("  [bold red]All uploads failed. Verify FRIGATE_URL is reachable and API is enabled.[/bold red]")
