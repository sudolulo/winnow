"""Persistent tracker for Immich asset IDs already uploaded/rejected by Frigate.

Two separate JSON files in CACHE_DIR:
  frigate_uploaded_ids.json  — successfully uploaded assets
  frigate_rejected_ids.json  — assets Frigate rejected (e.g. no face detected)

Both are excluded from future candidate pools. To reset:
  - All:           delete both files
  - One person:    call reset_person("Name") or set RESET_PERSON=Name
  - Rejects only:  delete frigate_rejected_ids.json, or set RETRY_REJECTED=true

by_person schema (frigate_uploaded_ids.json):
  {
    "asset_ids":    ["immich-id-1", ...],   # all assets we attempted to upload
    "scores":       {"immich-id-1": 0.953}, # Immich face confidence at upload time
    "frigate_count": 42                     # last known Frigate training image count
  }
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


def _get_ids(entry: list | dict) -> list[str]:
    """Extract asset_ids from either the old list format or the new dict format."""
    if isinstance(entry, list):
        return entry
    return entry.get("asset_ids", [])


def _migrate_entry(entry: list | dict) -> dict:
    """Ensure by_person entry is in the current dict format."""
    if isinstance(entry, list):
        return {"asset_ids": sorted(entry), "scores": {}}
    entry.setdefault("asset_ids", [])
    entry.setdefault("scores", {})
    return entry


def _mark(filename: str, asset_id: str, person_name: str | None, score: float | None = None) -> None:
    data = _load(filename)
    flat_key = _flat_key(filename)
    flat = set(data.get(flat_key, []))
    flat.add(asset_id)
    data[flat_key] = sorted(flat)
    if person_name:
        by_person = data.setdefault("by_person", {})
        entry = _migrate_entry(by_person.get(person_name, {}))
        ids = set(entry["asset_ids"])
        ids.add(asset_id)
        entry["asset_ids"] = sorted(ids)
        if score is not None:
            entry["scores"][asset_id] = round(score, 4)
        by_person[person_name] = entry
    _save(filename, data)


# ── Public API ────────────────────────────────────────────────────────────────

def load_uploaded_ids() -> set[str]:
    return _load_flat(UPLOAD_TRACKER_FILE)


def load_rejected_ids() -> set[str]:
    return _load_flat(REJECT_TRACKER_FILE)


def mark_uploaded(asset_id: str, person_name: str | None = None, score: float | None = None) -> None:
    _mark(UPLOAD_TRACKER_FILE, asset_id, person_name, score=score)
    logger.debug(f"Marked {asset_id} as uploaded ({person_name})")


def mark_rejected(asset_id: str, person_name: str | None = None) -> None:
    _mark(REJECT_TRACKER_FILE, asset_id, person_name)
    logger.debug(f"Marked {asset_id} as rejected ({person_name})")


def update_frigate_count(person_name: str, count: int) -> None:
    """Record Frigate's authoritative training image count for a person."""
    data = _load(UPLOAD_TRACKER_FILE)
    by_person = data.setdefault("by_person", {})
    entry = _migrate_entry(by_person.get(person_name, {}))
    entry["frigate_count"] = count
    by_person[person_name] = entry
    _save(UPLOAD_TRACKER_FILE, data)


def reset_person(person_name: str) -> None:
    """Remove all uploaded and rejected records for a given person."""
    for filename in (UPLOAD_TRACKER_FILE, REJECT_TRACKER_FILE):
        data = _load(filename)
        flat_key = _flat_key(filename)
        by_person = data.get("by_person", {})
        entry = by_person.pop(person_name, None)
        if entry is not None:
            person_ids = set(_get_ids(entry))
            flat = set(data.get(flat_key, [])) - person_ids
            data[flat_key] = sorted(flat)
            data["by_person"] = by_person
            _save(filename, data)
    logger.info(f"Reset tracking data for {person_name}")


def get_person_summary() -> dict[str, dict]:
    """Return {person_name: {uploaded, rejected, frigate_count, scores}} for display/capacity."""
    uploaded_data = _load(UPLOAD_TRACKER_FILE).get("by_person", {})
    rejected_data = _load(REJECT_TRACKER_FILE).get("by_person", {})
    names = set(uploaded_data) | set(rejected_data)
    result = {}
    for name in sorted(names):
        u_entry = uploaded_data.get(name, {})
        r_entry = rejected_data.get(name, {})
        result[name] = {
            "uploaded": len(_get_ids(u_entry)),
            "rejected": len(_get_ids(r_entry)),
            "frigate_count": u_entry.get("frigate_count") if isinstance(u_entry, dict) else None,
            "scores": u_entry.get("scores", {}) if isinstance(u_entry, dict) else {},
        }
    return result


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
