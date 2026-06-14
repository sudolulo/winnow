"""Frigate upload post-processing: reconciliation and asset enrichment."""

import logging
import time

from .frigate_api import get_frigate_person_files
from .immich_api import fetch_face_data
from .upload_tracker import record_frigate_files_batch

logger = logging.getLogger(__name__)


def reconcile_frigate_mappings(
    person_name: str,
    known_files_before: set[str],
    uploaded: list[tuple[str, str | None]],
) -> None:
    """Map Frigate filenames to asset IDs after a batch of uploads.

    Polls until all expected new files appear in the Frigate API, then maps
    them to asset IDs by filename timestamp order (Frigate processes the
    upload queue in FIFO order, so earlier uploads get earlier timestamps).

    KNOWN LIMITATION — race condition with external uploads:
    If another client uploads a face file for this person concurrently, the
    count of new files will exceed `len(uploaded)` and we bail out entirely
    (the "> target" branch). That's safe — we never record a wrong mapping —
    but those uploads become permanently unmapped (they won't be eligible for
    quality replacement). The right fix is a Frigate API that returns the
    filename in the upload response, removing the need for any post-upload
    diffing. Until then, the external-upload guard keeps mappings correct at
    the cost of occasionally missing them when another client is active.
    """
    target = len(uploaded)
    current_files: set[str] = set()

    for delay in (1, 2, 4, 8):
        time.sleep(delay)
        fresh = get_frigate_person_files(person_name)
        if fresh is None:
            logger.warning(
                "%s: Frigate API unreachable during mapping reconciliation"
                " — quality replacement won't target these files",
                person_name,
            )
            return
        current_files = set(fresh)
        new_count = len(current_files - known_files_before)
        if new_count == target:
            break
        if new_count > target:
            break  # external upload already visible — no point polling further

    new_files = current_files - known_files_before

    if len(new_files) == target:
        def _ts(fname: str) -> float:
            try:
                return float(fname.rsplit("_", 1)[-1].replace(".webp", ""))
            except (ValueError, IndexError):
                return 0.0

        mappings = {
            frigate_file: asset_id
            for (_, asset_id), frigate_file in zip(uploaded, sorted(new_files, key=_ts))
            if asset_id
        }
        record_frigate_files_batch(person_name, mappings)
    elif len(new_files) > target:
        logger.info(
            "%s: %s new Frigate files for %s uploads"
            " (external upload detected) — skipping file mapping",
            person_name,
            len(new_files),
            target,
        )
    else:
        logger.warning(
            "%s: only %s of %s expected Frigate files"
            " appeared after reconciliation — mapping skipped",
            person_name,
            len(new_files),
            target,
        )


def enrich_asset_with_face_data(asset: dict, person: dict) -> dict:
    """Enrich an asset dict with face bounding box data from the Immich faces API.

    The search/metadata endpoint does not include face bounding box data,
    so we fetch it from GET /api/faces?id={asset_id} and inject it into
    the asset's "people" field so process_face_mode can find it.

    Returns the enriched asset dict (modifies in place and returns it).
    """
    person_id = person["id"]
    face_data = fetch_face_data(asset["id"], person_id=person_id)

    if face_data is None:
        logger.debug("No face data returned for %s in asset %s", person.get("name"), asset.get("id"))
        # Clean any None entries from the people list (can come from Immich API)
        if "people" in asset:
            asset["people"] = [p for p in asset["people"] if p is not None]
        return asset

    # Skip zero-area bounding boxes (face detection failed or no face found)
    if face_data.bbox == (0, 0, 0, 0):
        logger.debug("Zero-area bounding box for %s in asset %s", person.get("name"), asset.get("id"))
        # Clean any None entries from the people list (can come from Immich API)
        if "people" in asset:
            asset["people"] = [p for p in asset["people"] if p is not None]
        return asset

    face_info = {
        "boundingBoxX1": face_data.bbox[0],
        "boundingBoxY1": face_data.bbox[1],
        "boundingBoxX2": face_data.bbox[2],
        "boundingBoxY2": face_data.bbox[3],
        "imageWidth": face_data.image_width,
        "imageHeight": face_data.image_height,
    }

    # Inject into asset so process_face_mode can find it via asset["people"]
    asset["people"] = [{"id": person_id, "faces": [face_info]}]
    asset["face_confidence"] = face_data.confidence
    return asset
