"""Frigate API helpers for querying face training state."""

import logging
import os

import requests

logger = logging.getLogger(__name__)


def get_frigate_version() -> str | None:
    """Fetch Frigate's version string from GET /api/version.

    Returns the version string (e.g. "0.16.0-beta4") or None if FRIGATE_URL
    is unset, the endpoint is unreachable, or the response is not parseable.
    """
    frigate_url = os.environ.get("FRIGATE_URL", "").rstrip("/")
    if not frigate_url:
        return None
    try:
        resp = requests.get(f"{frigate_url}/api/version", timeout=5)
        if resp.ok:
            return resp.text.strip().strip('"')
        return None
    except Exception:
        return None


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
        logger.warning("Could not query Frigate faces API: %s", e)
        return None


def get_all_frigate_person_files() -> dict[str, list[str]] | None:
    """Return {person_name: [filename, ...]} for every person in Frigate.

    Single call used to build per-person snapshots before the upload loop,
    avoiding one GET /api/faces per person.  Returns None if unavailable.
    """
    data = _get_faces_data()
    if data is None:
        return None
    # Response: {person_name: [file, ...], "train": [...], ...}
    # "train" is a flat pending list, not a person — skip it.
    # TODO(frigate-api): "train" is the only known special key as of Frigate v0.16.
    # Log unexpected non-list values so future Frigate schema additions are visible.
    result = {}
    for name, files in data.items():
        if name == "train":
            continue
        if isinstance(files, list):
            result[name] = files
        else:
            logger.debug("Frigate API: skipping unexpected key %r (got %s, not list)", name, type(files).__name__)
    return result


def get_frigate_face_counts() -> dict[str, int] | None:
    """Return {person_name: training_image_count} from Frigate's train directory.

    Returns None if FRIGATE_URL is not set or the API is unreachable, so callers
    can distinguish "API unavailable" from "person has 0 images."
    """
    all_files = get_all_frigate_person_files()
    if all_files is None:
        return None
    return {name: len(files) for name, files in all_files.items()}


def get_frigate_person_files(person_name: str) -> list[str] | None:
    """Return the list of training filenames for a person in Frigate.

    Returns None if the API is unreachable. Returns an empty list if the
    person exists but has no training images yet.
    """
    data = _get_faces_data()
    if data is None:
        return None
    files = data.get(person_name)
    if files is not None and not isinstance(files, list):
        logger.debug("Frigate API: unexpected type for %r — got %s, not list", person_name, type(files).__name__)
        return []
    return files if files is not None else []


def recognize_face(file_path: str) -> tuple[str | None, float] | None:
    """Submit an image to Frigate's recognize endpoint.

    Returns (face_name, score) where face_name is the best-matching person
    (may be "unknown" if below Frigate's confidence threshold) and score is
    the sigmoid-mapped cosine similarity (0-1) against that person's mean
    embedding.

    Returns None if FRIGATE_URL is unset, the API is unreachable, no face is
    detected, or face recognition is not enabled in Frigate.

    LIMITATION — mean embedding comparison: the score reflects similarity to
    the arithmetic mean of all training embeddings, not to individual ones.
    A bimodal training set (e.g. frontals + profiles) has a mean that sits
    between both clusters, making candidates from either cluster look more
    novel than they are. Winnow could add redundant frontals while the score
    suggests novelty, because the mean is pulled toward profiles.
    TODO(frigate-api): if Frigate exposes per-file embeddings via the API,
    replace mean-comparison with nearest-neighbour distance across individual
    training embeddings for accurate coverage detection.
    """
    frigate_url = os.environ.get("FRIGATE_URL", "").rstrip("/")
    if not frigate_url:
        return None
    try:
        with open(file_path, "rb") as f:
            resp = requests.post(
                f"{frigate_url}/api/faces/recognize",
                files={"file": (os.path.basename(file_path), f, "image/jpeg")},
                timeout=15,
            )
        if not resp.ok:
            return None
        data = resp.json()
        if data.get("success") and "score" in data:
            return (data.get("face_name"), round(float(data["score"]), 4))
        return None
    except Exception as e:
        logger.debug("Frigate recognize failed for %s: %s", file_path, e)
        return None


def delete_frigate_person_files(person_name: str, filenames: list[str]) -> bool:
    """Delete specific training files for a person from Frigate.

    Uses POST /api/faces/{name}/delete with body {"ids": [filename, ...]}.
    Returns True on success, False if unreachable or the request fails.
    """
    frigate_url = os.environ.get("FRIGATE_URL", "").rstrip("/")
    if not frigate_url or not filenames:
        return False
    from urllib.parse import quote
    encoded_name = quote(person_name, safe="")
    try:
        resp = requests.post(
            f"{frigate_url}/api/faces/{encoded_name}/delete",
            json={"ids": filenames},
            timeout=10,
        )
        if resp.ok:
            logger.debug("Deleted %s Frigate file(s) for %s", len(filenames), person_name)
            return True
        if resp.status_code == 404:
            # File already absent — stale tracker entry. Return True so the caller
            # removes it from the tracker and frees the slot cleanly.
            logger.warning("Frigate file(s) not found for %s (stale tracker entry?): %s", person_name, filenames)
            return True
        logger.warning("Frigate delete returned %s for %s", resp.status_code, person_name)
        return False
    except Exception as e:
        logger.warning("Failed to delete Frigate files for %s: %s", person_name, e)
        return False
