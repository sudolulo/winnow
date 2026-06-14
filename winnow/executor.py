"""Execution phase: image processing and Frigate upload."""

import logging
import os
import shutil
from io import BytesIO
from urllib.parse import quote

import requests
from PIL import Image
from rich import print as rprint
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from .config import Config, get_headers
from .frigate_api import (
    delete_frigate_person_files,
    get_all_frigate_person_files,
    get_frigate_person_files,
    get_frigate_version,
    recognize_face,
)
from .image_processing import process_face_mode
from .immich_api import fetch_full_image
from .log_config import console
from .quality import assess_quality
from .reconcile import enrich_asset_with_face_data, reconcile_frigate_mappings
from .upload_tracker import (
    get_lowest_quality_mapped_file,
    get_most_redundant_mapped_file,
    get_tracked_frigate_file_count,
    get_tracked_frigate_filenames,
    has_frigate_scores,
    mark_rejected,
    mark_uploaded,
    remove_frigate_file,
)

logger = logging.getLogger(__name__)


def _safe_person_dir(output_dir: str, person_name: str) -> str:
    """Return the output subdirectory for a person, raising ValueError on path traversal.

    os.path.join silently discards output_dir when person_name is absolute,
    and '../..' sequences resolve outside the tree. Both are rejected here.
    """
    candidate = os.path.realpath(os.path.join(output_dir, person_name))
    base = os.path.realpath(output_dir)
    # Use the base path as its own prefix when it's the filesystem root ("/"),
    # otherwise append os.sep — avoids the false "//" double-slash when base == "/".
    base_prefix = base if base == os.sep else base + os.sep
    if not candidate.startswith(base_prefix) and candidate != base:
        raise ValueError(f"Person name {person_name!r} escapes output directory — skipping")
    return candidate


def execute_jobs(jobs: list[dict]) -> None:
    """Download and process images for all jobs.

    Builds an asset_map per job (filename → Immich asset ID) so that
    upload_to_frigate() can mark assets as uploaded after success.
    """
    if not jobs:
        return

    console.rule("[bold blue]Execution Phase")

    use_full_res = Config.USE_FULL_RESOLUTION

    # Load InsightFace app for landmark-based crop alignment.
    # The model is already resident from the diversity/embedding phase, so this
    # is just a singleton lookup — no load cost.
    insightface_app = None
    if Config.ENABLE_FACE_ALIGNMENT:
        try:
            from .embeddings import get_insightface_app

            insightface_app = get_insightface_app()
        except Exception as e:
            logger.debug("InsightFace unavailable for crop alignment: %s", e)

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
            person, assets = job["person"], job["assets"]
            name = person["name"]

            job_task = progress.add_task(f"Processing {name}...", total=len(assets))
            try:
                person_dir = _safe_person_dir(Config.OUTPUT_DIR, name)
            except ValueError as e:
                logger.error(str(e))
                continue
            # Face crops are transient (uploaded then discarded); wipe before each run.
            if os.path.isdir(person_dir):
                shutil.rmtree(person_dir)
            os.makedirs(person_dir, exist_ok=True)

            # Track filename → asset_id, filename → confidence score, filename → crop dims
            asset_map: dict[str, str] = {}
            score_map: dict[str, float | None] = {}
            dims_map: dict[str, tuple[int, int]] = {}

            count = 0
            for asset in assets:
                try:
                    # Enrich the asset with face bounding box data from the Immich
                    # faces API (not included in search/metadata results).
                    asset = enrich_asset_with_face_data(asset, person)
                    # Skip download if detection confidence already disqualifies
                    # the asset — avoids fetching a large image we'll discard.
                    conf = asset.get("face_confidence")
                    if conf is not None and conf < Config.MIN_CONFIDENCE:
                        progress.console.print(
                            f"[yellow]Skipped {asset['id']}"
                            f" (detection confidence {conf:.2f} < {Config.MIN_CONFIDENCE})[/yellow]"
                        )
                        mark_rejected(asset["id"], person_name=name)
                        progress.advance(job_task)
                        progress.advance(overall_task)
                        continue

                    # Use full-resolution for final output when configured
                    if use_full_res:
                        img = fetch_full_image(asset["id"])
                    else:
                        resp = requests.get(
                            f"{Config.IMMICH_URL}/api/assets/{asset['id']}/thumbnail?size=preview&format=JPEG",
                            headers=get_headers(),
                            timeout=30,
                        )
                        if resp.ok:
                            try:
                                img = Image.open(BytesIO(resp.content))
                            except Exception:
                                logger.warning("Invalid image data for asset %s", asset["id"])
                                img = None
                        else:
                            img = None

                    if img is None:
                        progress.console.print(f"[red]Failed download {asset['id']}[/red]")
                    else:
                        saved = process_face_mode(
                            img, asset, person, person_dir, count, insightface_app=insightface_app
                        )
                        if saved:
                            filename = f"{count}.jpg"
                            asset_map[filename] = asset["id"]
                            score_map[filename] = asset.get("quality_score")
                            if isinstance(saved, tuple):
                                dims_map[filename] = saved
                            # Time-spread path: compute blur score from the downloaded
                            # image. Cap at 1440px so the scale matches the preview
                            # thumbnails the embedding path uses for scoring — Laplacian
                            # variance grows with resolution, making full-res and
                            # thumbnail scores incomparable if left uncapped.
                            if score_map[filename] is None:
                                try:
                                    score_img = img.convert("RGB") if img.mode != "RGB" else img
                                    if score_img.width > 1440 or score_img.height > 1440:
                                        score_img = score_img.copy()
                                        score_img.thumbnail((1440, 1440), Image.LANCZOS)
                                    score_map[filename] = assess_quality(score_img).blur_score
                                except Exception as exc:
                                    logger.debug("Quality score fallback for %s: %s", asset["id"], exc)
                                    score_map[filename] = 0.0  # unknown quality — treat as lowest

                            count += 1
                        else:
                            progress.console.print(
                                f"[yellow]Skipped {asset['id']} (no usable face data)[/yellow]"
                            )
                except Exception as e:
                    logger.error("Failed to process asset %s: %s", asset["id"], e)

                progress.advance(job_task)
                progress.advance(overall_task)

            # Store maps on the job so upload_to_frigate can use them
            job["asset_map"] = asset_map
            job["score_map"] = score_map
            job["dims_map"] = dims_map

            progress.remove_task(job_task)

            # Log how many images were actually saved vs selected
            if count < len(assets):
                logger.info("%s: saved %s/%s selected images", name, count, len(assets))


def upload_to_frigate(jobs: list[dict]) -> None:
    """Upload processed face crops to Frigate via API with detailed logging.

    After each successful upload, records the Immich asset ID in the
    upload tracker so it is skipped on future runs.
    """
    if not jobs:
        rprint("[dim]No jobs to upload.[/dim]")
        return

    frigate_url = os.environ.get("FRIGATE_URL", "")
    if not frigate_url:
        rprint("[yellow]⚠️  FRIGATE_URL not set, skipping upload.[/yellow]")
        return

    _frigate_version = get_frigate_version()
    if _frigate_version is not None:
        try:
            parts = [int(x) for x in _frigate_version.lstrip("v").split("-")[0].split(".") if x.isdigit()]
            if len(parts) >= 2 and (parts[0], parts[1]) < (0, 16):
                rprint(
                    f"  [yellow]⚠  Frigate {_frigate_version} detected — "
                    "face training API requires v0.16+. Uploads may fail.[/yellow]"
                )
        except Exception:
            pass

    rprint("\n[bold cyan]📤 Uploading to Frigate[/bold cyan]")
    rprint(f"  Target: [dim]{frigate_url}[/dim]")

    # Build a mapping of output filenames → Immich asset IDs
    # from the asset_map stored on each job during execute_jobs()
    filename_to_asset_id: dict[str, dict[str, str]] = {}
    total_files = 0
    for job in jobs:
        name = job["person"]["name"]
        asset_map = job.get("asset_map", {})
        filename_to_asset_id[name] = asset_map
        total_files += len(asset_map)

    if total_files == 0:
        rprint("  [yellow]No images found to upload.[/yellow]")
        return

    rprint(f"  People: [bold]{len(jobs)}[/bold], Total images: [bold]{total_files}[/bold]")

    uploaded, failed = 0, 0
    max_retries = 2

    # Fetch all Frigate training files once — avoids one GET /api/faces per person.
    # Falls back to per-person calls inside the loop if this fetch fails.
    all_frigate_files = get_all_frigate_person_files()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        upload_task = progress.add_task("[green]Uploading to Frigate", total=total_files)

        for job in jobs:
            name = job["person"]["name"]
            # URL-encode the name for the API (handles spaces, special chars)
            encoded_name = quote(name, safe="")
            if " " in name:
                progress.console.print(f"  ℹ️  URL-encoded name for Frigate API: '{name}' → '{encoded_name}'")

            try:
                person_dir = _safe_person_dir(Config.OUTPUT_DIR, name)
            except ValueError as e:
                logger.error(str(e))
                continue
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
            # Replacement targets also come exclusively from the tracker, so manually
            # added files are never selected for deletion — only winnow-uploaded ones.
            # LIMITATION — manual files are invisible to diversity decisions: winnow
            # can observe their effect on the Frigate score (indirectly, via recognize)
            # but cannot measure their embedding distribution directly. If a user has
            # 20 manually-added frontals and winnow has room for 20 more, winnow may
            # add more frontals because it can't see that frontals are already covered.
            # TODO(frigate-api): if Frigate exposes per-file embeddings, compute
            # diversity against the full training set (tracked + manual) rather than
            # relying solely on the Frigate score as a proxy signal.
            _snapshot = (
                all_frigate_files.get(name, []) if all_frigate_files is not None
                else get_frigate_person_files(name)
            )
            if _snapshot is None:
                # Frigate GET is down. The tracker only knows files winnow mapped
                # previously — it is blind to manually-added Frigate files. Using
                # the tracker as the baseline would make those unmapped files look
                # like new uploads in reconcile, triggering the >target guard and
                # silently dropping all mappings. Skip reconciliation entirely when
                # we can't get a reliable live snapshot.
                logger.warning(
                    "%s: Frigate API unreachable at upload start"
                    " — file mapping will be skipped for this batch", name
                )
                known_frigate_files_at_start: set[str] = set()
                _skip_reconcile = True
            else:
                known_frigate_files_at_start: set[str] = set(_snapshot)
                _skip_reconcile = False
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
            if Config.ENABLE_FRIGATE_SCORES and pre_run_count == 0:
                progress.console.print(
                    f"  [dim]{name}: first run — Frigate diversity scoring will apply from the next run[/dim]"
                )
            actually_uploaded: list[tuple[str, str | None]] = []
            failed_deletes: set[str] = set()
            min_quality_score_for_slot: float | None = None
            person_has_fscores: bool = has_frigate_scores(name)

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

                # Pre-upload Frigate score — clean measurement (image not yet in training set).
                # Called for all below-cap uploads (seeds frigate_scores for future at-cap
                # replacement) and for at-cap uploads when scores already exist. Skipped on
                # the first run (pre_run_count == 0) since Frigate has no model yet.
                # recognize_face returns (face_name, score); we only use the score when the
                # best match is for the correct person. Mismatches (or "unknown") are treated
                # as None so a wrong-person score never drives a ceiling skip or replacement.
                # Frigate rebuilds its model asynchronously after any delete (clear + background
                # thread), so the first recognize call after a deletion returns None — our code
                # handles this conservatively by skipping that candidate until the next run.
                # LIMITATION — async rebuild during multi-replacement runs: each deletion in a
                # single run triggers a background model rebuild in Frigate. Subsequent recognize
                # calls in the same run may get None (rebuild in progress), causing later
                # candidates to fall back to blur-score replacement or be skipped entirely.
                # The more replacements that happen in one run, the worse the scoring gets.
                # TODO(frigate-api): if Frigate exposes a model generation counter or a
                # rebuild-complete signal, poll it between recognize calls during replacement
                # sequences rather than accepting stale/None scores.
                pre_fscore: float | None = None
                if Config.ENABLE_FRIGATE_SCORES and pre_run_count > 0:
                    if not at_cap or person_has_fscores:
                        _result = recognize_face(fpath)
                        if _result is not None and (_result[0] or "").casefold() == name.casefold():
                            pre_fscore = _result[1]

                # Below-cap novelty gate: skip candidates already covered by the Frigate model,
                # including conditions learned from manually-added images winnow can't track.
                # pre_fscore is None on the first run (pre_run_count == 0 skips recognize_face
                # above), so this block never fires on the first run without an extra guard.
                if not at_cap and pre_fscore is not None:
                    _ceiling = Config.FRIGATE_SCORE_CEILING
                    if _ceiling is None:
                        # Dynamic default: bar = most-redundant tracked file's Frigate score.
                        # Falls back to uploading freely when no tracked scores exist yet.
                        _bar = get_most_redundant_mapped_file(name)
                        _skip = _bar is not None and pre_fscore > _bar[2]
                        _bar_str = f"most redundant tracked {_bar[2]:.2f}" if _bar else ""
                    elif _ceiling == 0.0:
                        _skip = False  # explicitly disabled
                        _bar_str = ""
                    else:
                        _skip = pre_fscore > _ceiling
                        _bar_str = f"ceiling {_ceiling:.2f}"
                    if _skip:
                        progress.console.print(
                            f"    [dim]⏭  {fname}: Frigate score {pre_fscore:.2f}"
                            f" > {_bar_str}, already covered[/dim]"
                        )
                        progress.advance(upload_task)
                        continue

                if at_cap:
                    if not quality_replacement:
                        progress.console.print(f"    [dim]⏭  {fname}: at cap, quality replacement disabled[/dim]")
                        progress.advance(upload_task)
                        continue

                    using_fscore = person_has_fscores and Config.ENABLE_FRIGATE_SCORES
                    if using_fscore:
                        candidate_score = pre_fscore
                        get_target = get_most_redundant_mapped_file
                        score_label, better_note = "frigate", " (more novel)"
                        no_score_msg = "Frigate recognize unavailable, skipping replacement"
                    else:
                        candidate_score = score_map.get(fname)
                        get_target = get_lowest_quality_mapped_file
                        score_label, better_note = "blur", ""
                        no_score_msg = "no quality score, skipping replacement"

                    if candidate_score is None:
                        progress.console.print(f"    [dim]⏭  {fname}: {no_score_msg}[/dim]")
                        progress.advance(upload_task)
                        continue

                    target = get_target(name, exclude=failed_deletes)
                    not_better = target is None or (
                        candidate_score >= target[2] if using_fscore else candidate_score <= target[2]
                    )
                    if not_better:
                        target_str = f"{target[2]:.3f}" if target is not None else "N/A"
                        op = "<" if using_fscore else ">"
                        progress.console.print(
                            f"    [dim]⏭  {fname}: {score_label} {candidate_score:.3f}"
                            f" not {op} {target_str}, skipping[/dim]"
                        )
                        progress.advance(upload_task)
                        continue

                    target_frigate_file, _target_asset_id, target_score = target
                    op = "<" if using_fscore else ">"
                    progress.console.print(
                        f"    🔄 {fname}: {score_label} {candidate_score:.3f} {op} {target_score:.3f},"
                        f" replacing {target_frigate_file}{better_note}"
                    )
                    if delete_frigate_person_files(name, [target_frigate_file]):
                        remove_frigate_file(name, target_frigate_file)
                        person_has_fscores = has_frigate_scores(name)
                        effective_count -= 1
                        min_quality_score_for_slot = None if using_fscore else candidate_score
                    else:
                        logger.warning("Failed to delete %s for %s, skipping replacement", target_frigate_file, name)
                        failed_deletes.add(target_frigate_file)
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
                                mark_uploaded(
                                    asset_id,
                                    person_name=name,
                                    score=score_map.get(fname),
                                    crop_dims=dims_map.get(fname),
                                    frigate_score=pre_fscore,
                                )
                                if pre_fscore is not None:
                                    person_has_fscores = True
                                actually_uploaded.append((fname, asset_id))

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
                            full_body = resp.text
                            try:
                                error_detail = resp.json().get("message", full_body[:100])
                            except Exception:
                                error_detail = full_body[:100]
                            if resp.status_code == 400:
                                progress.console.print(f"      [dim]{error_detail}[/dim]")
                            else:
                                logger.debug("%s HTTP %s: %s", fname, resp.status_code, error_detail)
                            if resp.status_code == 400 and "face" in full_body.lower():
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
            if actually_uploaded and not _skip_reconcile:
                reconcile_frigate_mappings(name, known_frigate_files_at_start, actually_uploaded)

            # Per-person summary
            if person_failed == 0:
                progress.console.print(
                    f"  ✅ {name}: {person_uploaded}/{person_uploaded} uploaded"
                )
            else:
                progress.console.print(
                    f"  ⚠️  {name}: {person_uploaded} succeeded, {person_failed} failed"
                )

    # Grand summary
    rprint("\n  [bold]Frigate Upload Summary:[/bold]")
    rprint(f"    ✅ Succeeded: [green]{uploaded}[/green]")
    if failed:
        rprint(f"    ❌ Failed:    [red]{failed}[/red]")
    else:
        rprint("    ❌ Failed:    0")

    if failed > 0:
        rprint("  [yellow]Check logs above for per-file error details.[/yellow]")

    if failed == total_files and total_files > 0:
        rprint("  [bold red]All uploads failed. Verify FRIGATE_URL is reachable and API is enabled.[/bold red]")
