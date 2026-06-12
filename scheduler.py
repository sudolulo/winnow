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

SCHEDULE = os.environ["CRON_SCHEDULE"]
MODELS_DIR = os.environ.get("HF_HOME", "/models/huggingface")
INSIGHTFACE_BASE = os.environ.get("INSIGHTFACE_HOME", "/models")

logger = logging.getLogger(__name__)


def check_models() -> None:
    buffalo = Path(INSIGHTFACE_BASE) / ".insightface" / "models" / "buffalo_l"
    hf_hub = Path(MODELS_DIR) / "hub"
    buffalo_ok = buffalo.exists()
    hf_ok = hf_hub.exists() and any(hf_hub.iterdir())
    if not buffalo_ok:
        print("  InsightFace Buffalo_L not found — will download on first run", flush=True)
    if not hf_ok:
        print("  HuggingFace models not found — will download on first run", flush=True)


# Import once — models loaded during the first run stay resident in memory
# for all subsequent scheduled runs, avoiding repeated multi-GB load times.
from winnow.cli import main  # noqa: E402

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
        except Exception as e:
            logger.error(f"winnow run failed: {e}", exc_info=True)
            print(f"winnow run failed: {e}", flush=True)
        next_run = cron.get_next(float)
    time.sleep(60)
