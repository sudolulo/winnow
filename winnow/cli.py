"""Interactive CLI for winnow."""

import logging
import os

from rich import print as rprint
from rich.prompt import Confirm

from .config import Config, ConfigManager
from .executor import execute_jobs, upload_to_frigate
from .immich_api import get_people
from .jobs import _show_preview, auto_configure, interactive_configure
from .log_config import console, setup_logging
from .upload_tracker import get_person_summary, reset_person

logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for winnow CLI."""
    try:
        setup_logging(verbose=False)

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

        # Check for non-interactive mode
        auto_mode = os.environ.get("AUTO_MODE", "false").lower() == "true"
        dry_run = os.environ.get("DRY_RUN", "false").lower() in ("true", "1", "yes")

        if dry_run:
            rprint("[bold yellow]DRY RUN — no images will be downloaded or uploaded[/bold yellow]")

        if auto_mode:
            rprint("[bold cyan]Running in AUTO mode (non-interactive)[/bold cyan]")
            jobs = auto_configure(people)
        else:
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
