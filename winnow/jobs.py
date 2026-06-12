"""Configuration phase: strategy selection and job building."""

import logging
import os

from rich import print as rprint
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from .config import Config
from .diversity import select_diverse_assets
from .embeddings import is_embedding_available, load_embedding_model
from .frigate_api import get_frigate_face_counts
from .immich_api import fetch_all_assets, filter_recent_assets
from .log_config import console
from .upload_tracker import filter_already_uploaded, get_person_summary, update_frigate_count

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


def _resolve_strategy(strategy: str, has_embedding: bool) -> tuple[int | str, str]:
    """Resolve env var strategy to (limit, selection_mode) without prompts."""
    custom_limit = os.environ.get("LIMIT", "").strip()

    if not has_embedding:
        limit = int(custom_limit) if custom_limit else 30
        return limit, "time"

    if custom_limit:
        return int(custom_limit), "smart"

    strategy_map = {
        "auto": ("auto", "smart"),
        "standard": (30, "smart"),
        "broad": (100, "smart"),
    }
    return strategy_map.get(strategy, ("auto", "smart"))


def _perform_selection(
    assets: list, limit: int | str, name: str, selection_mode: str, entity_type: str, person_id: str | None = None
) -> list:
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
                person_id=person_id,
                progress_callback=lambda c, t: progress.update(task, completed=c, total=t),
            )

        label = f"Auto-diversity selected {len(selected)}" if limit == "auto" else f"Selected {len(selected)}"
        rprint(f"  [green]{label} diverse images.[/green]")
        return selected

    rprint(f"\n[cyan]Using time-spread selection for {limit} images...[/cyan]")
    with console.status(f"[bold]Selecting {limit} images evenly distributed over time...[/bold]"):
        selected = select_diverse_assets(
            assets, limit, name, selection_mode="time", entity_type=entity_type, person_id=person_id
        )
    rprint(f"  [green]Selected {len(selected)} images using time spread.[/green]")
    return selected


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
    retry_rejected = os.environ.get("RETRY_REJECTED", "false").lower() in ("true", "1", "yes")
    before_dedup = len(recent_assets)
    new_asset_ids = set(filter_already_uploaded([a["id"] for a in recent_assets], retry_rejected=retry_rejected))
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
    selected_assets = _perform_selection(
        recent_assets, limit, name, selection_mode, entity_type, person_id=person["id"]
    )

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
            rprint(
                f"  Filtered to {len(valid_people)} people with"
                f" ≥{min_face_count} assets (MIN_FACE_COUNT={min_face_count})"
            )

    frigate_counts = get_frigate_face_counts()
    # Persist each count to tracker so the last known value survives Frigate downtime
    if frigate_counts is not None:
        for pname, count in frigate_counts.items():
            update_frigate_count(pname, count)
    upload_summary = get_person_summary()
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
        retry_rejected = os.environ.get("RETRY_REJECTED", "false").lower() in ("true", "1", "yes")
        before_dedup = len(recent_assets)
        new_asset_ids = set(filter_already_uploaded([a["id"] for a in recent_assets], retry_rejected=retry_rejected))
        recent_assets = [a for a in recent_assets if a["id"] in new_asset_ids]
        skipped = before_dedup - len(recent_assets)
        if skipped:
            rprint(f"  [dim]Skipped {skipped} assets already uploaded to Frigate.[/dim]")

        if not recent_assets:
            rprint(f"  [dim]Skipping {name} (0 new images after dedup).[/dim]")
            continue

        # Enforce MAX_AUTO_IMAGES as a lifetime cap per person.
        # Priority: live Frigate count → last cached Frigate count → local uploaded count.
        person_summary = upload_summary.get(name, {})
        if frigate_counts is not None:
            already_uploaded = frigate_counts.get(name, 0)
        else:
            fc = person_summary.get("frigate_count")
            already_uploaded = fc if fc is not None else person_summary.get("uploaded", 0)
        capacity = Config.MAX_AUTO_IMAGES - already_uploaded
        if capacity <= 0:
            rprint(
                f"  [dim]Skipping {name} (at lifetime cap:"
                f" {already_uploaded}/{Config.MAX_AUTO_IMAGES} trained).[/dim]"
            )
            continue

        has_embedding = is_embedding_available(entity_type)
        limit, selection_mode = _resolve_strategy(strategy, has_embedding)

        # Cap selection to remaining capacity
        if limit == "auto":
            if already_uploaded > 0:
                limit = capacity  # partially filled — select exactly what remains
        else:
            limit = min(limit, capacity)

        if selection_mode == "skip":
            continue

        selected_assets = _perform_selection(
            recent_assets, limit, name, selection_mode, entity_type, person_id=person["id"]
        )

        if selected_assets:
            rprint(f"  [green]Queued {len(selected_assets)} images for {name}.[/green]")
            jobs.append({"person": person, "assets": selected_assets, "limit": len(selected_assets), "config": config})

    return jobs


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
