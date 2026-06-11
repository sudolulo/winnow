"""Interactive CLI for if-curator."""

import logging
import os
from io import BytesIO
from urllib.parse import quote

import requests
from PIL import Image
from rich import print as rprint
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from .config import Config, ConfigManager
from .diversity import select_diverse_assets
from .embeddings import is_embedding_available, load_embedding_model
from .image_processing import process_face_mode, process_full_mode, process_object_mode
from .immich_api import fetch_all_assets, fetch_face_data, fetch_full_image, filter_recent_assets, get_people
from .logging import console, setup_logging
from .upload_tracker import filter_already_uploaded, mark_uploaded

logger = logging.getLogger(__name__)

# Strategy presets: (limit, mode_name)
STRATEGY_PRESETS = {
    "1": ("auto", "Auto Diversity"),
    "2": (30, "Standard (30)"),
    "3": (100, "Broad (100)"),
}


def _get_strategy_choice(has_embedding: bool, entity_type: str) -> tuple[int | str, str]:
    """Prompt user for training strategy and return (limit, selection_mode)."""
    model_name = "InsightFace" if entity_type == "face" else "SigLIP"

    if has_embedding:
        rprint("  [bold]1.[/bold] Auto (Objective Diversity) [green][Recommended][/green]")
        rprint("     [dim]• Dynamically selects images until redundancy starts[/dim]")
        rprint("  [bold]2.[/bold] Standard (30 images)")
        rprint("  [bold]3.[/bold] Broad (100 images)")
        rprint("  [bold]4.[/bold] Custom Count")
        rprint("  [bold]5.[/bold] Skip")

        choice = Prompt.ask("Choice", choices=["1", "2", "3", "4", "5"], default="1")

        if choice == "5":
            return 0, "skip"
        if choice == "4":
            limit = IntPrompt.ask("Enter number of images", default=30)
            mode = "smart" if Confirm.ask("Use Smart Diversity?", default=True) else "time"
            return limit, mode
        if choice in STRATEGY_PRESETS:
            return STRATEGY_PRESETS[choice][0], "smart"
        return 30, "smart"

    # Fallback when embedding model not available
    rprint(f"  [yellow]Note: {model_name} not available. Using Time Spread.[/yellow]")
    rprint("  [bold]1.[/bold] Standard (30 images) [green][Recommended][/green]")
    rprint("  [bold]2.[/bold] Broad (100 images)")
    rprint("  [bold]3.[/bold] Custom Count")
    rprint("  [bold]4.[/bold] Skip")

    choice = Prompt.ask("Choice", choices=["1", "2", "3", "4"], default="1")
    limits = {"1": 30, "2": 100, "3": IntPrompt.ask("Enter number of images", default=30)}
    return limits.get(choice, 0), "time" if choice != "4" else "skip"


def _configure_person(person: dict, people: list[dict]) -> dict | None:
    """Configure training for a single person. Returns job dict or None."""
    name = person["name"]
    console.print(f"\nSelected: [bold green]{name}[/bold green]")

    # Select training mode
    rprint("\n[bold cyan]Training Mode:[/bold cyan]")
    rprint("  [bold]1.[/bold] Face (Frigate Face Recognition)")
    rprint("  [bold]2.[/bold] Object (Frigate Object Classification)")

    mode_choice = Prompt.ask("Choice", choices=["1", "2"], default="1")
    entity_type = "face" if mode_choice == "1" else "object"

    config = {"name": name, "mode": entity_type}
    if entity_type == "object":
        config["object_class"] = Prompt.ask("Enter Object Class (e.g. dog, cat, car)", default="dog")

    # Fetch and filter assets
    years = IntPrompt.ask("Filter images older than (years)", default=Config.YEARS_FILTER)

    console.print(f"Scanning for {name} ({entity_type})...")
    with console.status("[bold green]Fetching assets...[/bold green]"):
        all_assets = fetch_all_assets(person)
        recent_assets = filter_recent_assets(all_assets, years=years)

    rprint(f"  Found [bold]{len(all_assets)}[/bold] total, [bold]{len(recent_assets)}[/bold] in range ({years} years).")

    # Filter out assets already uploaded to Frigate
    before_dedup = len(recent_assets)
    new_asset_ids = set(filter_already_uploaded([a["id"] for a in recent_assets]))
    recent_assets = [a for a in recent_assets if a["id"] in new_asset_ids]
    skipped = before_dedup - len(recent_assets)
    if skipped:
        rprint(f"  [dim]Skipped {skipped} assets already uploaded to Frigate.[/dim]")

    if not recent_assets:
        rprint("  [dim]Skipping (0 new images after dedup).[/dim]")
        return None

    # Strategy selection
    has_embedding = is_embedding_available(entity_type)
    rprint(f"\n[bold cyan]Select Training Strategy for {name}:[/bold cyan]")

    limit, selection_mode = _get_strategy_choice(has_embedding, entity_type)
    if selection_mode == "skip":
        return None

    # Perform selection
    selected_assets = _perform_selection(recent_assets, limit, name, selection_mode, entity_type)

    rprint(f"  [green]Queued {len(selected_assets)} images for {name}.[/green]")
    return {"person": person, "assets": selected_assets, "limit": len(selected_assets), "config": config}


def interactive_configure(people: list[dict]) -> list[dict]:
    """Interactive phase: select person(s), mode, and configure training strategy.

    Supports multi-person batch mode — after configuring one person,
    prompts to add another.
    """
    valid_people = sorted([p for p in people if p.get("name")], key=lambda x: x["name"])

    if not valid_people:
        rprint("[red]No people found with names in Immich.[/red]")
        return []

    jobs = []

    while True:
        # Select person
        console.print("\n[bold cyan]Select Person to Train:[/bold cyan]")
        for idx, p in enumerate(valid_people, 1):
            # Mark already-queued people
            marker = " [dim](queued)[/dim]" if any(j["person"]["id"] == p["id"] for j in jobs) else ""
            console.print(f"  [bold]{idx}.[/bold] {p['name']}{marker}")

        p_choice = IntPrompt.ask("Enter Number", choices=[str(i) for i in range(1, len(valid_people) + 1)])
        person = valid_people[p_choice - 1]

        job = _configure_person(person, valid_people)
        if job:
            jobs.append(job)

        # Multi-person: ask to add another
        if not Confirm.ask("\nAdd another person?", default=False):
            break

    return jobs


def auto_configure(people: list[dict]) -> list[dict]:
    """Non-interactive: configure jobs for all named people automatically."""
    valid_people = sorted([p for p in people if p.get("name")], key=lambda x: x["name"])

    if not valid_people:
        rprint("[red]No people found with names in Immich.[/red]")
        return []

    mode = os.environ.get("TRAINING_MODE", "face")
    strategy = os.environ.get("STRATEGY", "auto")
    skip = os.environ.get("SKIP_PEOPLE", "").split(",") if os.environ.get("SKIP_PEOPLE") else []
    only = os.environ.get("ONLY_PEOPLE", "").split(",") if os.environ.get("ONLY_PEOPLE") else []

    if only:
        valid_people = [p for p in valid_people if p["name"] in only]
    if skip:
        valid_people = [p for p in valid_people if p["name"] not in skip]

    # Filter by minimum face count (Issue #6: previously unimplemented)
    min_face_count = Config.MIN_FACE_COUNT
    if min_face_count > 0:
        valid_people = [p for p in valid_people if p.get("assetCount", 0) >= min_face_count]
        if valid_people:
            rprint(f"  Filtered to {len(valid_people)} people with ≥{min_face_count} assets (MIN_FACE_COUNT={min_face_count})")

    jobs = []
    for person in valid_people:
        name = person["name"]
        entity_type = mode

        config = {"name": name, "mode": entity_type}
        if entity_type == "object":
            config["object_class"] = os.environ.get("OBJECT_CLASS", "dog")

        all_assets = fetch_all_assets(person)
        recent_assets = filter_recent_assets(all_assets, years=Config.YEARS_FILTER)

        rprint(f"  {name}: {len(all_assets)} total, {len(recent_assets)} recent")

        # Filter out assets already uploaded to Frigate
        before_dedup = len(recent_assets)
        new_asset_ids = set(filter_already_uploaded([a["id"] for a in recent_assets]))
        recent_assets = [a for a in recent_assets if a["id"] in new_asset_ids]
        skipped = before_dedup - len(recent_assets)
        if skipped:
            rprint(f"  [dim]Skipped {skipped} assets already uploaded to Frigate.[/dim]")

        if not recent_assets:
            rprint(f"  [dim]Skipping {name} (0 new images after dedup).[/dim]")
            continue

        has_embedding = is_embedding_available(entity_type)
        limit, selection_mode = _resolve_strategy(strategy, has_embedding)

        if selection_mode == "skip":
            continue

        selected_assets = _perform_selection(recent_assets, limit, name, selection_mode, entity_type)

        if selected_assets:
            rprint(f"  [green]Queued {len(selected_assets)} images for {name}.[/green]")
            jobs.append({"person": person, "assets": selected_assets, "limit": len(selected_assets), "config": config})

    return jobs


def _resolve_strategy(strategy: str, has_embedding: bool) -> tuple[int | str, str]:
    """Resolve env var strategy to (limit, selection_mode) without prompts."""
    if not has_embedding:
        return 30, "time"

    strategy_map = {
        "auto": ("auto", "smart"),
        "standard": (30, "smart"),
        "broad": (100, "smart"),
    }
    return strategy_map.get(strategy, ("auto", "smart"))


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
        person_dir = os.path.join(Config.OUTPUT_DIR, name)
        if not os.path.isdir(person_dir):
            continue
        person_files = sorted(
            f for f in os.listdir(person_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        )
        total_files += len(person_files)

        # Recover asset IDs from the job's asset_map
        asset_map = job.get("asset_map", {})
        filename_to_asset_id[name] = asset_map

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

            person_files = sorted(
                f for f in os.listdir(person_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            )

            if not person_files:
                progress.console.print(f"  [dim]⏭️  {name}: no images found[/dim]")
                continue

            # Get the asset map for this person
            asset_map = filename_to_asset_id.get(name, {})

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
                                mark_uploaded(asset_id)

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
                                progress.console.print(f"      [dim]{resp.text[:100]}[/dim]")
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


def _perform_selection(assets: list, limit: int | str, name: str, selection_mode: str, entity_type: str) -> list:
    """Run diversity selection with progress display."""
    if selection_mode == "smart":
        model_display = "InsightFace (face embeddings)" if entity_type == "face" else "SigLIP (visual embeddings)"
        rprint(f"\n[cyan]Using {model_display} for diversity analysis...[/cyan]")

        # Pre-load model explicitly (separate from availability check)
        load_embedding_model(entity_type)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"[cyan]Computing embeddings for {len(assets)} images...", total=None)
            selected = select_diverse_assets(
                assets,
                limit,
                name,
                selection_mode=selection_mode,
                entity_type=entity_type,
                progress_callback=lambda c, t: progress.update(task, completed=c, total=t),
            )

        label = f"Auto-diversity selected {len(selected)}" if limit == "auto" else f"Selected {len(selected)}"
        rprint(f"  [green]{label} diverse images.[/green]")
        return selected

    rprint(f"\n[cyan]Using time-spread selection for {limit} images...[/cyan]")
    with console.status(f"[bold]Selecting {limit} images evenly distributed over time...[/bold]"):
        selected = select_diverse_assets(assets, limit, name, selection_mode="time", entity_type=entity_type)
    rprint(f"  [green]Selected {len(selected)} images using time spread.[/green]")
    return selected


def _show_preview(jobs: list[dict]) -> None:
    """Show a summary table of all queued jobs before execution."""
    table = Table(title="📋 Training Job Preview", show_header=True, header_style="bold cyan")
    table.add_column("Person", style="bold")
    table.add_column("Mode", style="dim")
    table.add_column("Images", justify="right")
    table.add_column("Date Range", style="dim")

    for job in jobs:
        name = job["person"]["name"]
        mode = job["config"].get("mode", "face")
        count = str(job["limit"])

        # Date range
        dates = sorted(a.get("fileCreatedAt", "")[:10] for a in job["assets"] if a.get("fileCreatedAt"))
        date_range = f"{dates[0]} → {dates[-1]}" if len(dates) >= 2 else (dates[0] if dates else "—")

        table.add_row(name, mode, count, date_range)

    console.print()
    console.print(table)
    console.print()


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
        return asset

    # Skip zero-area bounding boxes (face detection failed or no face found)
    if face_data.bbox == (0, 0, 0, 0):
        logger.debug(f"Zero-area bounding box for {person.get('name')} in asset {asset.get('id')}")
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
            os.makedirs(person_dir, exist_ok=True)

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
                            headers={"x-api-key": Config.API_KEY, "Accept": "application/json"},
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


def main() -> None:
    """Entry point for if-curator CLI."""
    try:
        setup_logging(verbose=False)

        console.print(r"""
    [bold blue]if-curator[/bold blue]
    [dim]Immich -> Frigate Training Data Curator[/dim]
        """)

        ConfigManager.get().interactive_setup()

        try:
            Config.validate()
        except ValueError as e:
            rprint(f"[bold red]Configuration Error:[/bold red] {e}")
            return

        rprint(f"Server: [dim]{Config.IMMICH_URL}[/dim]")
        rprint(f"Output: [dim]{Config.OUTPUT_DIR}[/dim]")

        people = get_people()
        if not people:
            rprint("[bold red]Could not fetch people from Immich. Check URL/Key.[/bold red]")
            return

        # Check for non-interactive mode
        auto_mode = os.environ.get("AUTO_MODE", "false").lower() == "true"

        if auto_mode:
            rprint("[bold cyan]Running in AUTO mode (non-interactive)[/bold cyan]")
            jobs = auto_configure(people)
        else:
            jobs = interactive_configure(people)

        if jobs:
            _show_preview(jobs)
            if auto_mode or Confirm.ask(f"Ready to process {sum(j['limit'] for j in jobs)} images?"):
                execute_jobs(jobs)
                upload_to_frigate(jobs)
                rprint("\n[bold green]Done! Happy Training.[/bold green]")
        else:
            rprint("[yellow]No jobs configured.[/yellow]")

    except KeyboardInterrupt:
        rprint("\n[bold red]Aborted by user.[/bold red]")


if __name__ == "__main__":
    main()

