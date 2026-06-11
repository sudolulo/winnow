#!/usr/bin/env python3
import os
import sys
import subprocess
import time
import logging
from pathlib import Path

try:
    from croniter import croniter
except ImportError:
    print("❌ croniter not installed. Run: uv add croniter")
    sys.exit(1)

SCHEDULE = os.environ["CRON_SCHEDULE"]
MODELS_DIR = os.environ.get("HF_HOME", "/models/huggingface")
INSIGHTFACE_BASE = os.environ.get("INSIGHTFACE_HOME", "/models")

RUN_ENV = {**os.environ, "PYTHONUNBUFFERED": "1"}

logger = logging.getLogger(__name__)


def check_models():
    """Log model status before each run."""
    print("📦 Checking models...", flush=True)
    buffalo = Path(INSIGHTFACE_BASE) / ".insightface" / "models" / "buffalo_l"
    if buffalo.exists():
        print("  ✅ InsightFace Buffalo_L: present", flush=True)
    else:
        print("  ⬇️  InsightFace Buffalo_L: not found — will download", flush=True)

    hf_hub = Path(MODELS_DIR) / "hub"
    if hf_hub.exists() and any(hf_hub.iterdir()):
        print("  ✅ HuggingFace models: present", flush=True)
    else:
        print("  ⬇️  HuggingFace models: not found — will download", flush=True)
    print("🚀 Starting if-curator...", flush=True)


NOW = time.time()
cron = croniter(SCHEDULE, NOW)
next_run = cron.get_next(float)

while True:
    now = time.time()
    if now >= next_run:
        print(f"\n▶ [{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting if-curator...", flush=True)
        check_models()
        result = subprocess.run(["uv", "run", "if-curator"], env=RUN_ENV)
        if result.returncode != 0:
            logger.error(f"if-curator exited with code {result.returncode}")
            print(f"❌ if-curator failed with exit code {result.returncode}", flush=True)
        else:
            print("✅ if-curator completed successfully", flush=True)
        next_run = cron.get_next(float)
    time.sleep(60)

