"""Frigate API helpers for querying face training state."""

import logging
import os

import requests

logger = logging.getLogger(__name__)


def _get_faces_data() -> dict | None:
    """Fetch raw GET /api/faces response. Returns None if unavailable."""
    frigate_url = os.environ.get("FRIGATE_URL", "").rstrip("/")
    if not frigate_url:
        return None
    try:
        resp = requests.get(f"{frigate_url}/api/faces", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"Could not query Frigate faces API: {e}")
        return None


def get_frigate_face_counts() -> dict[str, int] | None:
    """Return {person_name: training_image_count} from Frigate's train directory.

    Returns None if FRIGATE_URL is not set or the API is unreachable, so callers
    can distinguish "API unavailable" from "person has 0 images."
    """
    data = _get_faces_data()
    if data is None:
        return None
    # Response: {person_name: [file, ...], "train": [...], ...}
    # "train" is a flat pending list, not a person — skip it.
    return {
        name: len(files)
        for name, files in data.items()
        if name != "train" and isinstance(files, list)
    }


def get_frigate_person_files(person_name: str) -> list[str] | None:
    """Return the list of training filenames for a person in Frigate.

    Returns None if the API is unreachable. Returns an empty list if the
    person exists but has no training images yet.
    """
    data = _get_faces_data()
    if data is None:
        return None
    files = data.get(person_name)
    return files if isinstance(files, list) else []


def delete_frigate_person_files(person_name: str, filenames: list[str]) -> bool:
    """Delete specific training files for a person from Frigate.

    Uses POST /api/faces/{name}/delete with body {"ids": [filename, ...]}.
    Returns True on success, False if unreachable or the request fails.
    """
    frigate_url = os.environ.get("FRIGATE_URL", "").rstrip("/")
    if not frigate_url or not filenames:
        return False
    from urllib.parse import quote
    encoded = quote(person_name, safe="")
    try:
        resp = requests.post(
            f"{frigate_url}/api/faces/{encoded}/delete",
            json={"ids": filenames},
            timeout=10,
        )
        if resp.ok:
            logger.debug(f"Deleted {len(filenames)} Frigate file(s) for {person_name}")
            return True
        logger.warning(f"Frigate delete returned {resp.status_code} for {person_name}")
        return False
    except Exception as e:
        logger.warning(f"Failed to delete Frigate files for {person_name}: {e}")
        return False
