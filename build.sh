#!/bin/bash
set -e

# --- Configuration ---
IMAGE_NAME="ghcr.io/sudolulo/if-curator-headless"
TAG="latest"
COMMIT_MSG="automated update: $(date '+%Y-%m-%d %H:%M:%S')"

# --- 1. Git Commit & Push ---
echo "💾 Committing changes to Git..."
git add .
if ! git diff-index --quiet HEAD --; then
    git commit -m "$COMMIT_MSG"
    git push
    echo "✅ Changes pushed to GitHub."
else
    echo "ℹ️  No changes to commit."
fi

# --- 2. Build Docker ---
echo "🔧 Building image (no cache)..."
docker build --no-cache -t ${IMAGE_NAME}:${TAG} .

# --- 3. Push to GHCR ---
echo "📤 Pushing to GHCR..."
docker push ${IMAGE_NAME}:${TAG}

# --- 4. Success UI ---
echo ""
echo "  ╔══════════════════════════════════════════════════╗"
echo "  ║                                                  ║"
echo "  ║     ✅  GIT & DOCKER DEPLOY COMPLETE             ║"
echo "  ║                                                  ║"
echo "  ╚══════════════════════════════════════════════════╝"
echo ""

