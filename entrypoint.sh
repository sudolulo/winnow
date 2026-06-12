#!/bin/bash
set -e
export PYTHONUNBUFFERED=1

# CRON_SCHEDULE controls container lifetime:
#   unset          — run once and exit
#   empty string   — stay alive, run nothing (use: docker exec -it winnow winnow)
#   cron expression — run immediately, then on schedule

if [ "${CRON_SCHEDULE+isset}" = "isset" ] && [ -z "$CRON_SCHEDULE" ]; then
    echo "▶ CRON_SCHEDULE is empty — manual mode. Use 'docker exec -it winnow winnow' to run."
    exec sleep infinity
fi

echo "▶ Running on startup..."
/app/.venv/bin/winnow

if [ -n "${CRON_SCHEDULE:-}" ]; then
    echo "▶ CRON_SCHEDULE set to: $CRON_SCHEDULE"
    echo "▶ Switching to scheduled mode..."
    exec /app/.venv/bin/python /app/scheduler.py
else
    echo "▶ No schedule set, exiting."
fi
