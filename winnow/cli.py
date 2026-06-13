"""Interactive CLI for winnow."""

import logging
import os
import sys

from rich import print as rprint
from rich.prompt import Confirm

from .config import Config, ConfigManager
from .executor import execute_jobs, upload_to_frigate
from .immich_api import get_people, merge_people
from .jobs import _show_preview, auto_configure, interactive_configure
from .log_config import console, setup_logging
from .upload_tracker import find_by_crop_dimension, get_person_summary, reset_person

logger = logging.getLogger(__name__)


def _handle_trace_crop(size_str: str) -> None:
    """Print tracker records whose crop dimension matches the given pixel size and exit."""
    try:
        size = int(size_str)
    except ValueError:
        rprint(f"[bold red]TRACE_CROP_SIZE must be an integer, got: {size_str!r}[/bold red]")
        sys.exit(1)

    immich_url = os.environ.get("IMMICH_URL", "").rstrip("/")
    matches = find_by_crop_dimension(size)
    if not matches:
        rprint(f"[yellow]No crops with dimension {size}px found in tracker.[/yellow]")
        rprint("[dim]Note: crop dimensions are only recorded for uploads made after this feature was added.[/dim]")
        sys.exit(0)

    rprint(f"\n[bold]Crops matching dimension {size}px:[/bold] ({len(matches)} found)\n")
    for m in matches:
        rprint(f"  [bold cyan]{m['person']}[/bold cyan]")
        rprint(f"    Dimensions:   {m['width']}×{m['height']}px")
        rprint(f"    Asset ID:     {m['asset_id']}")
        if immich_url:
            rprint(f"    Immich URL:   {immich_url}/photos/{m['asset_id']}")
        blur = m.get("blur_score")
        rprint(f"    Blur score:   {blur:.1f}" if blur is not None else "    Blur score:   unknown")
        fscore = m.get("frigate_score")
        rprint(f"    Frigate score: {fscore:.2f}" if fscore is not None else "    Frigate score: unknown")
        if m.get("frigate_filename"):
            rprint(f"    Frigate file: {m['frigate_filename']}")
        else:
            rprint("    Frigate file: [dim]unmapped (reconciliation race)[/dim]")
        rprint()
    sys.exit(0)


def _handle_duplicate_people(people: list[dict]) -> list[dict]:
    """Warn about or merge Immich people that share the same name.

    Duplicates arise when Immich creates separate person records for the same
    individual (e.g. unmerged face clusters). Without handling, winnow would
    run multiple jobs for the same Frigate folder and overwrite its own output,
    leaving far fewer training images than expected.

    With MERGE_DUPLICATE_PEOPLE=false (default): prints a warning, skips the
    smaller duplicates so only the person with the most assets is processed,
    and returns a deduplicated people list.
    With MERGE_DUPLICATE_PEOPLE=true: merges each duplicate group inside
    Immich via its API (permanently combines the face records), then
    re-fetches the people list so the rest of the run sees the merged state.
    """
    from collections import defaultdict

    by_name: dict[str, list[dict]] = defaultdict(list)
    for p in people:
        name = (p.get("name") or "").strip()
        if name:
            by_name[name].append(p)

    duplicates = {name: ps for name, ps in by_name.items() if len(ps) > 1}
    if not duplicates:
        return people

    if not Config.MERGE_DUPLICATE_PEOPLE:
        rprint("\n[bold yellow]⚠  Duplicate person names detected in Immich:[/bold yellow]")
        for name, ps in sorted(duplicates.items()):
            ordered = sorted(ps, key=lambda x: x.get("assetCount", 0), reverse=True)
            entries = ", ".join(
                f"[dim]{p['id'][:8]}…[/dim] ({p.get('assetCount', 0)} assets)"
                for p in ordered
            )
            rprint(f"  [yellow]{name}[/yellow] → {len(ps)} people: {entries}")
            skipped = ordered[1:]
            rprint(
                f"  [dim]  Processing largest only "
                f"({ordered[0].get('assetCount', 0)} assets). "
                f"Skipping {len(skipped)} smaller duplicate(s) to avoid overwriting output.[/dim]"
            )
        rprint(
            "  [dim]Set MERGE_DUPLICATE_PEOPLE=true to permanently merge duplicates "
            "inside Immich (keeps the person with the most assets).[/dim]\n"
        )
        # Return deduplicated list — keep only the largest per name so that
        # downstream job creation never runs two jobs for the same Frigate folder.
        skip_ids = {
            p["id"]
            for ps in duplicates.values()
            for p in sorted(ps, key=lambda x: x.get("assetCount", 0), reverse=True)[1:]
        }
        return [p for p in people if p["id"] not in skip_ids]

    # Auto-merge: survivor = largest asset count, rest merge into it inside Immich
    merged_any = False
    for name, ps in sorted(duplicates.items()):
        ordered = sorted(ps, key=lambda x: x.get("assetCount", 0), reverse=True)
        survivor = ordered[0]
        merge_ids = [p["id"] for p in ordered[1:]]
        rprint(
            f"  [cyan]Merging {name!r} inside Immich:[/cyan] keeping "
            f"[dim]{survivor['id'][:8]}…[/dim] ({survivor.get('assetCount', 0)} assets), "
            f"absorbing {len(merge_ids)} smaller duplicate(s)..."
        )
        if merge_people(survivor["id"], merge_ids):
            rprint(f"  [green]✓ Merged {name!r}[/green]")
            merged_any = True
        else:
            rprint(f"  [red]✗ Failed to merge {name!r}[/red]")

    if merged_any:
        rprint("  [dim]Re-fetching people after merge...[/dim]")
        return get_people()

    return people


def main() -> None:
    """Entry point for winnow CLI."""
    try:
        verbose = os.environ.get("VERBOSE", "").lower() in ("true", "1", "yes")
        setup_logging(verbose=verbose)

        trace_size = os.environ.get("TRACE_CROP_SIZE", "").strip()
        if trace_size:
            _handle_trace_crop(trace_size)

        console.print(r"""
    [bold blue]winnow[/bold blue]
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

        # Handle RESET_PERSON before anything else
        reset_person_name = os.environ.get("RESET_PERSON", "").strip()
        if reset_person_name:
            reset_person(reset_person_name)
            rprint(f"[bold yellow]Reset tracking data for: {reset_person_name}[/bold yellow]")

        # Show per-person tracker summary if data exists
        summary = get_person_summary()
        if summary:
            rprint("\n[dim]Tracker summary:[/dim]")
            for person_name, counts in summary.items():
                frigate_part = (
                    f", {counts['frigate_count']} in Frigate"
                    if counts.get("frigate_count") is not None
                    else ""
                )
                rprint(
                    f"  [dim]{person_name}: {counts['uploaded']} uploaded,"
                    f" {counts['rejected']} rejected{frigate_part}[/dim]"
                )

        people = get_people()
        if not people:
            rprint("[bold red]Could not fetch people from Immich. Check URL/Key.[/bold red]")
            return

        people = _handle_duplicate_people(people)

        # Auto mode when no TTY (Docker, cron, pipes) — the primary use case.
        # A TTY means local interactive use; AUTO_MODE=true overrides that for scripting.
        auto_mode = not sys.stdin.isatty() or os.environ.get("AUTO_MODE", "").lower() in ("true", "1", "yes")
        dry_run = os.environ.get("DRY_RUN", "false").lower() in ("true", "1", "yes")

        if dry_run:
            rprint("[bold yellow]DRY RUN — no images will be downloaded or uploaded[/bold yellow]")

        if auto_mode:
            jobs = auto_configure(people)
        else:
            rprint("[bold cyan]Interactive mode — set AUTO_MODE=true to skip prompts[/bold cyan]")
            jobs = interactive_configure(people)

        if jobs:
            _show_preview(jobs)
            if dry_run:
                rprint("\n[bold yellow]Dry run complete — skipping execute and upload.[/bold yellow]")
            elif auto_mode or Confirm.ask(f"Ready to process {sum(j['limit'] for j in jobs)} images?"):
                execute_jobs(jobs)
                upload_to_frigate(jobs)
                rprint("\n[bold green]Done! Happy Training.[/bold green]")
        else:
            rprint("[yellow]No jobs configured.[/yellow]")

    except KeyboardInterrupt:
        rprint("\n[bold red]Aborted by user.[/bold red]")


if __name__ == "__main__":
    main()
