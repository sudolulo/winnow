"""Persistent tracker for Immich asset IDs already uploaded/rejected by Frigate.

Two separate JSON files in CACHE_DIR:
  frigate_uploaded_ids.json  — successfully uploaded assets
  frigate_rejected_ids.json  — assets Frigate rejected (e.g. no face detected)

Both are excluded from future candidate pools. To reset:
  - All:           delete both files
  - One person:    call reset_person("Name") or set RESET_PERSON=Name
  - Rejects only:  delete frigate_rejected_ids.json, or set RETRY_REJECTED=true
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

UPLOAD_TRACKER_FILE = "frigate_uploaded_ids.json"
REJECT_TRACKER_FILE = "frigate_rejected_ids.json"


def _tracker_path(filename: str) -> Path:
    try:
        from .config import Config
        return Path(Config.CACHE_DIR) / filename
    except (ImportError, AttributeError):
        return Path(filename)


def _load(filename: str) -> dict:
    path = _tracker_path(filename)
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not load tracker {filename}: {e}")
        return {}


def _save(filename: str, data: dict) -> None:
    path = _tracker_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _flat_key(filename: str) -> str:
    return "uploaded_asset_ids" if "uploaded" in filename else "rejected_asset_ids"


def _load_flat(filename: str) -> set[str]:
    return set(_load(filename).get(_flat_key(filename), []))


def _mark(filename: str, asset_id: str, person_name: str | None) -> None:
    data = _load(filename)
    flat_key = _flat_key(filename)
    flat = set(data.get(flat_key, []))
    flat.add(asset_id)
    data[flat_key] = sorted(flat)
    if person_name:
        by_person = data.setdefault("by_person", {})
        person_ids = set(by_person.get(person_name, []))
        person_ids.add(asset_id)
        by_person[person_name] = sorted(person_ids)
    _save(filename, data)


# ── Public API ────────────────────────────────────────────────────────────────

def load_uploaded_ids() -> set[str]:
    return _load_flat(UPLOAD_TRACKER_FILE)


def load_rejected_ids() -> set[str]:
    return _load_flat(REJECT_TRACKER_FILE)


def mark_uploaded(asset_id: str, person_name: str | None = None) -> None:
    _mark(UPLOAD_TRACKER_FILE, asset_id, person_name)
    logger.debug(f"Marked {asset_id} as uploaded ({person_name})")


def mark_rejected(asset_id: str, person_name: str | None = None) -> None:
    _mark(REJECT_TRACKER_FILE, asset_id, person_name)
    logger.debug(f"Marked {asset_id} as rejected ({person_name})")


def reset_person(person_name: str) -> None:
    """Remove all uploaded and rejected records for a given person."""
    for filename in (UPLOAD_TRACKER_FILE, REJECT_TRACKER_FILE):
        data = _load(filename)
        flat_key = _flat_key(filename)
        by_person = data.get("by_person", {})
        person_ids = set(by_person.pop(person_name, []))
        if person_ids:
            flat = set(data.get(flat_key, [])) - person_ids
            data[flat_key] = sorted(flat)
            data["by_person"] = by_person
            _save(filename, data)
    logger.info(f"Reset tracking data for {person_name}")


def get_person_summary() -> dict[str, dict[str, int]]:
    """Return {person_name: {uploaded: N, rejected: N}} for display."""
    uploaded_by = _load(UPLOAD_TRACKER_FILE).get("by_person", {})
    rejected_by = _load(REJECT_TRACKER_FILE).get("by_person", {})
    names = set(uploaded_by) | set(rejected_by)
    return {
        name: {
            "uploaded": len(uploaded_by.get(name, [])),
            "rejected": len(rejected_by.get(name, [])),
        }
        for name in sorted(names)
    }


def filter_already_uploaded(
    asset_ids: list[str],
    retry_rejected: bool = False,
) -> list[str]:
    """Return asset IDs not yet uploaded (and not rejected, unless retry_rejected)."""
    exclude = load_uploaded_ids()
    if not retry_rejected:
        exclude |= load_rejected_ids()
    new_ids = [aid for aid in asset_ids if aid not in exclude]
    skipped = len(asset_ids) - len(new_ids)
    if skipped:
        logger.info(f"Skipping {skipped} assets already uploaded or rejected by Frigate")
    return new_ids
