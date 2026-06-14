#!/usr/bin/env python3
import logging
import os
import sys
import time
from pathlib import Path

try:
    from croniter import croniter
except ImportError:
    print("croniter not installed. Run: uv add croniter")
    sys.exit(1)

# Imported at module level so models loaded during the first run stay
# resident in memory across all subsequent scheduled runs.
from winnow.cli import main

logger = logging.getLogger(__name__)


def _check_models() -> None:
    insightface_home = os.environ.get("INSIGHTFACE_HOME", "/models/.insightface")
    buffalo = Path(insightface_home) / "models" / "buffalo_l"
    if not buffalo.exists():
        print("  InsightFace Buffalo_L not found — will download on first run", flush=True)


def _run_scheduler() -> None:
    schedule = os.environ.get("CRON_SCHEDULE")
    if not schedule:
        print("Error: CRON_SCHEDULE environment variable is required.", flush=True)
        sys.exit(1)

    try:
        Path("/tmp/winnow.pid").write_text(str(os.getpid()))
    except OSError as e:
        print(f"Warning: could not write PID file: {e}", flush=True)

    now = time.time()
    cron = croniter(schedule, now)
    next_run = cron.get_next(float)
    print(f"Next run: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(next_run))}", flush=True)

    while True:
        now = time.time()
        if now >= next_run:
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting winnow run...", flush=True)
            _check_models()
            try:
                main()
                print("winnow run complete", flush=True)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error("winnow run failed: %s", e, exc_info=True)
                print(f"winnow run failed: {e}", flush=True)
            next_run = cron.get_next(float)
            print(f"Next run: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(next_run))}", flush=True)
        time.sleep(min(60, max(1, next_run - time.time())))


if __name__ == "__main__":
    _run_scheduler()
