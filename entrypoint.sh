#!/bin/bash
set -e

SCHEDULE="${CRON_SCHEDULE:-}"

if [ -z "$SCHEDULE" ]; then
    echo "▶ No CRON_SCHEDULE set — running once"
    uv run if-curator
    exit 0
fi

# Validate cron expression
if ! echo "$SCHEDULE" | cron-validate 2>/dev/null; then
    echo "⚠️  Could not validate cron expression, attempting anyway: $SCHEDULE"
fi

# Write cron job (run as appuser)
CRON_CMD="cd /app && /root/.local/bin/uv run if-curator >> /proc/1/fd/1 2>&1"
echo "$SCHEDULE appuser $CRON_CMD" > /etc/crontab

echo "▶ Scheduled with cron: $SCHEDULE"
echo "▶ Next run: $(crontab -l 2>/dev/null || echo 'see /etc/crontab')"

# Start cron daemon in foreground
echo "▶ Cron daemon started. Waiting for scheduled runs..."
cron -f
