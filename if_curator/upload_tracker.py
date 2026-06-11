"""Persistent tracker for Immich asset IDs already uploaded to Frigate.

Prevents duplicate uploads across runs by recording each successfully
uploaded asset ID in a JSON file within the configured CACHE_DIR.

To re-train from scratch, simply delete the tracker file
(frigate_uploaded_ids.json) from your cache directory.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

UPLOAD_TRACKER_FILE = "frigate_uploaded_ids.json"


def _tracker_path() -> Path:
    """Return path to the tracker file, using Config.CACHE_DIR if available."""
    try:
        from .config import Config

        return Path(Config.CACHE_DIR) / UPLOAD_TRACKER_FILE
    except (ImportError, AttributeError):
        return Path(UPLOAD_TRACKER_FILE)


def load_uploaded_ids() -> set[str]:
    """Load the set of Immich asset IDs already uploaded to Frigate."""
    path = _tracker_path()
    if not path.exists():
        return set()
    try:
        with open(path) as f:
            data = json.load(f)
            return set(data.get("uploaded_asset_ids", []))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not load upload tracker: {e}")
        return set()


def save_uploaded_ids(uploaded_ids: set[str]) -> None:
    """Persist the set of uploaded Immich asset IDs to disk."""
    path = _tracker_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"uploaded_asset_ids": sorted(uploaded_ids)}, f, indent=2)


def mark_uploaded(asset_id: str) -> None:
    """Mark a single Immich asset ID as uploaded to Frigate."""
    ids = load_uploaded_ids()
    ids.add(asset_id)
    save_uploaded_ids(ids)
    logger.debug(f"Marked asset {asset_id} as uploaded to Frigate")


def is_uploaded(asset_id: str) -> bool:
    """Check if an Immich asset ID has already been uploaded to Frigate."""
    return asset_id in load_uploaded_ids()


def filter_already_uploaded(asset_ids: list[str]) -> list[str]:
    """Return only asset IDs that have NOT yet been uploaded to Frigate.

    Logs how many were skipped so the user knows dedup is working.
    """
    uploaded = load_uploaded_ids()
    new_ids = [aid for aid in asset_ids if aid not in uploaded]
    skipped = len(asset_ids) - len(new_ids)
    if skipped:
        logger.info(f"Skipping {skipped} assets already uploaded to Frigate")
    return new_ids

