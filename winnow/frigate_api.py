"""Frigate API helpers for querying face training state."""

import logging
import os

import requests

logger = logging.getLogger(__name__)


def get_frigate_face_counts() -> dict[str, int] | None:
    """Return {person_name: training_image_count} from Frigate's train directory.

    Returns None if FRIGATE_URL is not set or the API is unreachable, so callers
    can distinguish "API unavailable" from "person has 0 images."
    """
    frigate_url = os.environ.get("FRIGATE_URL", "").rstrip("/")
    if not frigate_url:
        return None
    try:
        resp = requests.get(f"{frigate_url}/api/faces", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # Response: {person_name: [file, ...], "train": [...], ...}
        # "train" is a flat pending list, not a person — skip it.
        return {
            name: len(files)
            for name, files in data.items()
            if name != "train" and isinstance(files, list)
        }
    except Exception as e:
        logger.warning(f"Could not query Frigate face counts: {e}")
        return None
