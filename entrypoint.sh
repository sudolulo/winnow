#!/bin/bash
set -e
export PYTHONUNBUFFERED=1

# 1. Run the job immediately on startup
echo "▶ Running on startup..."
/app/.venv/bin/winnow

# 2. If a schedule exists, start the scheduler
if [ -n "${CRON_SCHEDULE:-}" ]; then
    echo "▶ CRON_SCHEDULE set to: $CRON_SCHEDULE"
    echo "▶ Switching to scheduled mode..."
    exec /app/.venv/bin/python /app/scheduler.py
else
    echo "▶ No schedule set, exiting."
fi

