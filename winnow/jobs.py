"""Configuration phase: strategy selection and job building."""

import logging
import os

from rich import print as rprint
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from .config import Config, _getenv_bool, _getenv_int, _getenv_optional_int
from .diversity import select_diverse_assets
from .embeddings import is_embedding_available, load_embedding_model
from .frigate_api import get_frigate_face_counts
from .immich_api import fetch_all_assets, filter_recent_assets
from .log_config import console
from .upload_tracker import filter_already_uploaded, get_person_summary, update_frigate_count

logger = logging.getLogger(__name__)

# Strategy presets: (limit, mode_name)
STRATEGY_PRESETS = {
    "1": ("auto", "Adaptive Diversity"),
    "2": (30, "Standard (30)"),
    "3": (100, "Broad (100)"),
}


def _get_strategy_choice(has_embedding: bool) -> tuple[int | str, str]:
    """Prompt user for training strategy and return (limit, selection_mode)."""
    if has_embedding:
        rprint("  [bold]1.[/bold] Adaptive Diversity [green][Recommended][/green]")
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
    rprint("  [yellow]Note: InsightFace not available. Using Time Spread.[/yellow]")
    rprint("  [bold]1.[/bold] Standard (30 images) [green][Recommended][/green]")
    rprint("  [bold]2.[/bold] Broad (100 images)")
    rprint("  [bold]3.[/bold] Custom Count")
    rprint("  [bold]4.[/bold] Skip")

    choice = Prompt.ask("Choice", choices=["1", "2", "3", "4"], default="1")
    if choice == "4":
        return 0, "skip"
    if choice == "3":
        return IntPrompt.ask("Enter number of images", default=30), "time"
    return {"1": 30, "2": 100}.get(choice, 30), "time"


def _resolve_strategy(strategy: str, has_embedding: bool) -> tuple[int | str, str]:
    """Resolve env var strategy to (limit, selection_mode) without prompts."""
    if not has_embedding:
        return _getenv_int("LIMIT", 30), "time"

    custom_limit = _getenv_optional_int("LIMIT")
    if custom_limit is not None and custom_limit > 0:
        return custom_limit, "smart"
    if custom_limit == 0:
        logger.warning("LIMIT=0 is invalid — ignoring and using auto strategy")

    strategy_map = {
        "adaptive": ("auto", "smart"),
        "auto": ("auto", "smart"),  # legacy alias for adaptive
        "standard": (30, "smart"),
        "broad": (100, "smart"),
    }
    return strategy_map.get(strategy, ("auto", "smart"))


def _perform_selection(
    assets: list, limit: int | str, name: str, selection_mode: str, person_id: str | None = None
) -> list:
    """Run diversity selection with progress display."""
    if selection_mode == "smart":
        rprint("\n[cyan]Using InsightFace (face embeddings) for diversity analysis...[/cyan]")

        load_embedding_model()

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
                person_id=person_id,
                progress_callback=lambda c, t: progress.update(task, completed=c, total=t),
            )

        label = f"Auto-diversity selected {len(selected)}" if limit == "auto" else f"Selected {len(selected)}"
        rprint(f"  [green]{label} diverse images.[/green]")
        return selected

    rprint(f"\n[cyan]Using time-spread selection for {limit} images...[/cyan]")
    with console.status(f"[bold]Selecting {limit} images evenly distributed over time...[/bold]"):
        selected = select_diverse_assets(assets, limit, name, selection_mode="time", person_id=person_id)
    rprint(f"  [green]Selected {len(selected)} images using time spread.[/green]")
    return selected


def _build_job(
    person: dict,
    assets: list,
    limit: int | str,
    selection_mode: str,
    quality_replacement: bool = False,
) -> dict | None:
    """Select from pre-filtered assets and build a job dict. No terminal I/O."""
    if not assets:
        return None
    name = person["name"]
    selected = _perform_selection(assets, limit, name, selection_mode, person_id=person["id"])
    if not selected:
        return None
    return {
        "person": person,
        "assets": selected,
        "limit": len(selected),
        "config": {"name": name, "quality_replacement": quality_replacement},
    }


def _configure_person(person: dict, people: list[dict]) -> dict | None:
    """Configure training for a single person. Returns job dict or None."""
    name = person["name"]
    console.print(f"\nSelected: [bold green]{name}[/bold green]")

    years = IntPrompt.ask("Filter images older than (years)", default=Config.YEARS_FILTER)

    console.print(f"Scanning for {name}...")
    with console.status("[bold green]Fetching assets...[/bold green]"):
        all_assets, total_raw = fetch_all_assets(person)
        recent_assets = filter_recent_assets(all_assets, years=years)

    rprint(f"  Found [bold]{total_raw}[/bold] total, [bold]{len(recent_assets)}[/bold] in range ({years} years).")

    # Ask before strategy so the post-dedup count can inform the choice
    retry_env = _getenv_bool("RETRY_REJECTED", False)
    retry_rejected = Confirm.ask("Include previously rejected images?", default=retry_env)

    before_dedup = len(recent_assets)
    new_asset_ids = set(filter_already_uploaded([a["id"] for a in recent_assets], retry_rejected=retry_rejected))
    recent_assets = [a for a in recent_assets if a["id"] in new_asset_ids]
    skipped = before_dedup - len(recent_assets)
    if skipped:
        rprint(f"  [dim]Skipped {skipped} assets already uploaded to Frigate.[/dim]")

    if not recent_assets:
        rprint("  [dim]Skipping (0 new images after dedup).[/dim]")
        return None

    has_embedding = is_embedding_available()
    rprint(f"\n[bold cyan]Select Training Strategy for {name}:[/bold cyan]")
    limit, selection_mode = _get_strategy_choice(has_embedding)
    if selection_mode == "skip":
        return None

    job = _build_job(person, recent_assets, limit, selection_mode, quality_replacement=Config.QUALITY_REPLACEMENT)
    if job is None:
        rprint("  [dim]Skipping (0 images selected).[/dim]")
        return None

    rprint(f"  [green]Queued {job['limit']} images for {name}.[/green]")
    return job


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

    strategy = os.environ.get("STRATEGY", "auto")
    skip = [s.strip() for s in os.environ.get("SKIP_PEOPLE", "").split(",") if s.strip()]
    only = [s.strip() for s in os.environ.get("ONLY_PEOPLE", "").split(",") if s.strip()]

    if only:
        valid_people = [p for p in valid_people if p["name"] in only]
    if skip:
        valid_people = [p for p in valid_people if p["name"] not in skip]

    min_face_count = Config.MIN_FACE_COUNT

    frigate_counts = get_frigate_face_counts()
    # Persist each count to tracker so the last known value survives Frigate downtime
    if frigate_counts is not None:
        for pname, count in frigate_counts.items():
            update_frigate_count(pname, count)
    upload_summary = get_person_summary()
    jobs = []
    for person in valid_people:
        name = person["name"]

        all_assets, total_raw = fetch_all_assets(person)
        recent_assets = filter_recent_assets(all_assets, years=Config.YEARS_FILTER)

        rprint(f"  {name}: {total_raw} total, {len(recent_assets)} recent")

        # MIN_FACE_COUNT guard: skip people with too few Immich assets.
        # Uses total_raw so that non-dict items from a transient Immich schema
        # issue on a mixed page don't shrink the count below the threshold.
        # Done here (after fetch) rather than upfront because Immich v2.7.5+
        # dropped assetCount from the /api/people response.
        if min_face_count > 0 and total_raw < min_face_count:
            rprint(f"  [dim]Skipping {name} ({total_raw} assets < MIN_FACE_COUNT={min_face_count}).[/dim]")
            continue

        # Enforce MAX_AUTO_IMAGES against the tracked file count only.
        # Manually-added Frigate files are invisible to this cap so users can
        # curate their own files without shrinking winnow's managed quota.
        person_summary = upload_summary.get(name, {})
        already_uploaded = len(person_summary.get("frigate_files", {}))
        capacity = Config.MAX_AUTO_IMAGES - already_uploaded
        if capacity <= 0:
            if not Config.QUALITY_REPLACEMENT:
                rprint(
                    f"  [dim]Skipping {name} (at cap:"
                    f" {already_uploaded}/{Config.MAX_AUTO_IMAGES}, quality replacement disabled).[/dim]"
                )
                continue
            rprint(
                f"  [cyan]{name}: at cap ({already_uploaded}/{Config.MAX_AUTO_IMAGES}),"
                f" checking for quality improvements...[/cyan]"
            )
            quality_replacement_only = True
        else:
            quality_replacement_only = False

        quality_replacement = quality_replacement_only or Config.QUALITY_REPLACEMENT

        has_embedding = is_embedding_available()
        limit, selection_mode = _resolve_strategy(strategy, has_embedding)

        # Cap selection to remaining capacity (no cap when replacement-only — executor
        # decides per-image whether to swap; any candidate could be an improvement).
        if not quality_replacement_only:
            if limit == "auto":
                # Switch from open-ended auto to a fixed budget at remaining capacity
                # so the diversity selector itself stops at the right count instead of
                # selecting MAX_AUTO_IMAGES and then discarding the excess by position.
                if already_uploaded > 0:
                    limit = capacity
            else:
                limit = min(limit, capacity)

        if selection_mode == "skip":
            continue

        retry_rejected = _getenv_bool("RETRY_REJECTED", False)
        before_dedup = len(recent_assets)
        new_asset_ids = set(filter_already_uploaded([a["id"] for a in recent_assets], retry_rejected=retry_rejected))
        recent_assets = [a for a in recent_assets if a["id"] in new_asset_ids]
        skipped = before_dedup - len(recent_assets)
        if skipped:
            rprint(f"  [dim]Skipped {skipped} assets already uploaded to Frigate.[/dim]")

        if not recent_assets:
            rprint(f"  [dim]Skipping {name} (0 new images after dedup).[/dim]")
            continue

        job = _build_job(person, recent_assets, limit, selection_mode, quality_replacement=quality_replacement)
        if job is None:
            rprint(f"  [dim]Skipping {name} (0 images selected).[/dim]")
            continue

        rprint(f"  [green]Queued {job['limit']} images for {name}.[/green]")
        jobs.append(job)

    return jobs


def _show_preview(jobs: list[dict]) -> None:
    """Show a summary table of all queued jobs before execution."""
    table = Table(title="📋 Training Job Preview", show_header=True, header_style="bold cyan")
    table.add_column("Person", style="bold")
    table.add_column("Images", justify="right")
    table.add_column("Date Range", style="dim")

    for job in jobs:
        name = job["person"]["name"]
        count = str(job["limit"])

        dates = sorted(a.get("fileCreatedAt", "")[:10] for a in job["assets"] if a.get("fileCreatedAt"))
        date_range = f"{dates[0]} → {dates[-1]}" if len(dates) >= 2 else (dates[0] if dates else "—")

        table.add_row(name, count, date_range)

    console.print()
    console.print(table)
    console.print()
