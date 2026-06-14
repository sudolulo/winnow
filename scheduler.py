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

SCHEDULE = os.environ["CRON_SCHEDULE"]
INSIGHTFACE_HOME = os.environ.get("INSIGHTFACE_HOME", "/models/.insightface")

logger = logging.getLogger(__name__)


def check_models() -> None:
    buffalo = Path(INSIGHTFACE_HOME) / "models" / "buffalo_l"
    if not buffalo.exists():
        print("  InsightFace Buffalo_L not found — will download on first run", flush=True)


NOW = time.time()
cron = croniter(SCHEDULE, NOW)
next_run = cron.get_next(float)

while True:
    now = time.time()
    if now >= next_run:
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting winnow run...", flush=True)
        check_models()
        try:
            main()
            print("winnow run complete", flush=True)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(f"winnow run failed: {e}", exc_info=True)
            print(f"winnow run failed: {e}", flush=True)
        next_run = cron.get_next(float)
    time.sleep(max(1, next_run - time.time()))
