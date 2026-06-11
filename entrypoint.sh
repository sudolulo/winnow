#!/bin/bash
set -e
export PYTHONUNBUFFERED=1

# 1. Run the job immediately on startup
echo "▶ Running on startup..."
uv run python -m if_curator.cli

# 2. If a schedule exists, start the scheduler
if [ -n "${CRON_SCHEDULE:-}" ]; then
    echo "▶ CRON_SCHEDULE set to: $CRON_SCHEDULE"
    echo "▶ Switching to scheduled mode..."
    exec uv run python3 /app/scheduler.py
else
    echo "▶ No schedule set, exiting."
fi

