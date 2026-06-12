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
from .image_processing import process_face_mode, process_full_mode, process_object_mode
from .immich_api import fetch_face_data, fetch_full_image
from .logging import console
from .upload_tracker import mark_rejected, mark_uploaded

logger = logging.getLogger(__name__)


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
            if os.path.isdir(person_dir):
                shutil.rmtree(person_dir)
            os.makedirs(person_dir)

            # Track filename → asset_id mapping for upload dedup
            asset_map: dict[str, str] = {}

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
                            # Also record object-mode variant filenames
                            if mode == "object":
                                for f in sorted(os.listdir(person_dir)):
                                    if f.startswith(f"{count}_") and f not in asset_map:
                                        asset_map[f] = asset["id"]

                            count += 1
                        else:
                            progress.console.print(
                                f"[yellow]Skipped {asset['id']} (no usable face data)[/yellow]"
                            )
                except Exception as e:
                    logger.error(f"Failed to process asset {asset['id']}: {e}")

                progress.advance(job_task)
                progress.advance(overall_task)

            # Store asset_map on the job so upload_to_frigate can use it
            job["asset_map"] = asset_map

            progress.remove_task(job_task)

            # Log how many images were actually saved vs selected
            if count < len(assets):
                logger.info(f"{name}: saved {count}/{len(assets)} selected images")


def upload_to_frigate(jobs: list[dict]) -> None:
    """Upload processed face crops to Frigate via API with detailed logging.

    After each successful upload, records the Immich asset ID in the
    upload tracker so it is skipped on future runs.
    """
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

            person_dir = os.path.join(Config.OUTPUT_DIR, name)
            if not os.path.isdir(person_dir):
                progress.console.print(f"  [dim]⏭️  {name}: no output directory, skipping[/dim]")
                continue

            asset_map = filename_to_asset_id.get(name, {})
            person_files = sorted(asset_map.keys())

            if not person_files:
                progress.console.print(f"  [dim]⏭️  {name}: no images found[/dim]")
                continue

            progress.console.print(f"  📁 {name}: uploading {len(person_files)} image(s)...")
            person_uploaded = 0
            person_failed = 0

            for fname in person_files:
                fpath = os.path.join(person_dir, fname)
                success = False

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
                            success = True

                            # Mark this asset as uploaded so it's skipped on future runs
                            asset_id = asset_map.get(fname)
                            if asset_id:
                                mark_uploaded(asset_id, person_name=name)

                            break
                        else:
                            if attempt < max_retries:
                                logger.warning(f"Upload attempt {attempt}/{max_retries} for {fname}: HTTP {resp.status_code}, retrying...")
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
                            logger.warning(f"Upload attempt {attempt}/{max_retries} for {fname}: {type(exc).__name__}, retrying...")
                            continue
                        failed += 1
                        person_failed += 1
                        label = "Connection refused" if isinstance(exc, requests.exceptions.ConnectionError) else "Request timed out (30s)"
                        progress.console.print(
                            f"    [red]✗ {fname}: {label} (after {max_retries} attempts)[/red]"
                        )
                    except Exception as e:
                        if attempt < max_retries:
                            logger.warning(f"Upload attempt {attempt}/{max_retries} for {fname}: {type(e).__name__}, retrying...")
                            continue
                        failed += 1
                        person_failed += 1
                        progress.console.print(
                            f"    [red]✗ {fname}: {type(e).__name__} - {e} (after {max_retries} attempts)[/red]"
                        )

                progress.advance(upload_task)

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
