#!/bin/bash
set -e

export PYTHONUNBUFFERED=1

# Function to log model status before each run
check_models() {
    echo "📦 Checking models..."

    if [ -d "/models/.insightface/models/buffalo_l" ]; then
        echo "  ✅ InsightFace Buffalo_L: present"
    else
        echo "  ⬇️  InsightFace Buffalo_L: not found — will download on first run (~300MB)"
    fi

    if [ -d "/models/huggingface/hub" ] && [ "$(find /models/huggingface/hub -maxdepth 1 -type d 2>/dev/null | wc -l)" -gt 1 ]; then
        echo "  ✅ HuggingFace models: present"
    else
        echo "  ⬇️  HuggingFace models: not found — will download on first run (SigLIP ~1GB, YOLOv9c ~500MB)"
    fi

    echo "🚀 Starting if-curator..."
}

SCHEDULE="${CRON_SCHEDULE:-}"
AUTO="${AUTO_MODE:-false}"

# Check if interactive mode is viable
if [ "$AUTO" != "true" ] && [ ! -t 0 ]; then
    echo "❌ AUTO_MODE is not enabled and no TTY is attached."
    echo "❌ Set AUTO_MODE=true for headless/automated runs."
    exit 1
fi

# Always run once on startup
echo "▶ Running on startup..."
check_models
uv run if-curator

if [ -z "$SCHEDULE" ]; then
    echo "▶ No CRON_SCHEDULE set — exiting"
    exit 0
fi

echo "▶ CRON_SCHEDULE set to: $SCHEDULE"
echo "▶ Switching to scheduled mode..."
exec uv run python3 /app/scheduler.py

