#!/bin/bash
set -e

SCHEDULE="${CRON_SCHEDULE:-}"

if [ -z "$SCHEDULE" ]; then
    echo "▶ No CRON_SCHEDULE set — running once"
    uv run if-curator
    exit 0
fi

echo "▶ CRON_SCHEDULE set to: $SCHEDULE"
echo "▶ Running if-curator on schedule..."

# Hand off to the Python scheduler
exec uv run python3 /app/scheduler.py

