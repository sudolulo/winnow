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
    "asset_ids":      ["immich-id-1", ...],                    # all assets we attempted to upload
    "scores":         {"immich-id-1": 450.3},                  # Laplacian blur variance at upload time
    "frigate_scores": {"immich-id-1": 0.87},                   # Frigate recognition confidence (0-1) post-upload
    "frigate_files":  {"PersonName-123.webp": "immich-id-1"},  # Frigate filename → asset ID
    "crop_dims":      {"immich-id-1": [640, 480]},             # crop pixel dimensions at upload time
    "frigate_count":  42                                       # last known Frigate training image count
  }

frigate_scores uses the same 0-1 sigmoid-mapped cosine similarity that Frigate
displays in its UI. When available, quality replacement uses frigate_scores in
preference to blur scores — an image Frigate cannot recognize is a poor training
image regardless of sharpness.

frigate_files only contains files winnow uploaded — files added manually through
Frigate's UI are never mapped here and are never touched by quality replacement.
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
        return {"asset_ids": sorted(entry), "scores": {}, "frigate_scores": {}, "frigate_files": {}, "crop_dims": {}}
    entry.setdefault("asset_ids", [])
    entry.setdefault("scores", {})
    entry.setdefault("frigate_scores", {})
    entry.setdefault("frigate_files", {})
    entry.setdefault("crop_dims", {})
    return entry


def _mark(
    filename: str,
    asset_id: str,
    person_name: str | None,
    score: float | None = None,
    crop_dims: tuple[int, int] | None = None,
    frigate_score: float | None = None,
) -> None:
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
        if crop_dims is not None:
            entry["crop_dims"][asset_id] = [crop_dims[0], crop_dims[1]]
        if frigate_score is not None:
            entry["frigate_scores"][asset_id] = round(frigate_score, 4)
        by_person[person_name] = entry
    _save(filename, data)


# ── Public API ────────────────────────────────────────────────────────────────

def load_uploaded_ids() -> set[str]:
    return _load_flat(UPLOAD_TRACKER_FILE)


def load_rejected_ids() -> set[str]:
    return _load_flat(REJECT_TRACKER_FILE)


def mark_uploaded(
    asset_id: str,
    person_name: str | None = None,
    score: float | None = None,
    crop_dims: tuple[int, int] | None = None,
    frigate_score: float | None = None,
) -> None:
    _mark(UPLOAD_TRACKER_FILE, asset_id, person_name, score=score, crop_dims=crop_dims, frigate_score=frigate_score)
    logger.debug(f"Marked {asset_id} as uploaded ({person_name})")


def mark_rejected(asset_id: str, person_name: str | None = None) -> None:
    _mark(REJECT_TRACKER_FILE, asset_id, person_name)
    logger.debug(f"Marked {asset_id} as rejected ({person_name})")


def record_frigate_file(person_name: str, frigate_filename: str, asset_id: str) -> None:
    """Record the mapping from a Frigate training filename to an Immich asset ID."""
    data = _load(UPLOAD_TRACKER_FILE)
    by_person = data.setdefault("by_person", {})
    entry = _migrate_entry(by_person.get(person_name, {}))
    entry["frigate_files"][frigate_filename] = asset_id
    by_person[person_name] = entry
    _save(UPLOAD_TRACKER_FILE, data)
    logger.debug(f"Mapped Frigate file {frigate_filename} → {asset_id} ({person_name})")


def remove_frigate_file(person_name: str, frigate_filename: str) -> None:
    """Remove a Frigate filename from the mapping after it has been deleted.

    Does NOT unmark the source asset_id — the deletion was deliberate and
    we don't want to re-upload the inferior image on the next run.
    """
    data = _load(UPLOAD_TRACKER_FILE)
    by_person = data.get("by_person", {})
    entry = _migrate_entry(by_person.get(person_name, {}))
    entry["frigate_files"].pop(frigate_filename, None)
    by_person[person_name] = entry
    _save(UPLOAD_TRACKER_FILE, data)
    logger.debug(f"Removed Frigate file mapping {frigate_filename} ({person_name})")


def get_tracked_frigate_file_count(person_name: str) -> int:
    """Return the number of Frigate training files winnow has mapped for this person.

    Used as the cap baseline so that manually-added Frigate files do not
    consume slots from winnow's managed quota.
    """
    data = _load(UPLOAD_TRACKER_FILE)
    entry = _migrate_entry(data.get("by_person", {}).get(person_name, {}))
    return len(entry["frigate_files"])


def get_tracked_frigate_filenames(person_name: str) -> set[str]:
    """Return the set of Frigate filenames currently mapped in the tracker for a person.

    Used as a pre-upload baseline when the Frigate GET API is unreachable at
    upload start, so reconciliation can still identify newly uploaded files.
    """
    data = _load(UPLOAD_TRACKER_FILE)
    entry = _migrate_entry(data.get("by_person", {}).get(person_name, {}))
    return set(entry["frigate_files"].keys())


def has_frigate_scores(person_name: str) -> bool:
    """Return True if any mapped file for this person has a stored Frigate recognition score."""
    data = _load(UPLOAD_TRACKER_FILE)
    entry = _migrate_entry(data.get("by_person", {}).get(person_name, {}))
    frigate_files = entry.get("frigate_files", {})
    frigate_scores = entry.get("frigate_scores", {})
    return any(asset_id in frigate_scores for asset_id in frigate_files.values())


def get_lowest_quality_mapped_file(
    person_name: str, exclude: set[str] | None = None
) -> tuple[str, str, float] | None:
    """Return (frigate_filename, asset_id, score) for the mapped file with the lowest
    quality score, or None if no mapped files with known scores exist.

    Uses Frigate recognition scores (0-1) when any are present for this person,
    treating files without a Frigate score as 0.0. Falls back to Laplacian blur
    scores when no Frigate scores exist yet.

    Pass `exclude` to skip files that failed to delete this run without removing
    them from the tracker — they remain candidates on the next run.
    """
    data = _load(UPLOAD_TRACKER_FILE)
    entry = _migrate_entry(data.get("by_person", {}).get(person_name, {}))
    frigate_files = entry.get("frigate_files", {})
    blur_scores = entry.get("scores", {})
    frigate_scores = entry.get("frigate_scores", {})

    mapped = [
        (ff, asset_id)
        for ff, asset_id in frigate_files.items()
        if exclude is None or ff not in exclude
    ]
    if not mapped:
        return None

    use_frigate = any(asset_id in frigate_scores for _, asset_id in mapped)

    if use_frigate:
        candidates = [
            (ff, asset_id, frigate_scores.get(asset_id, 0.0))
            for ff, asset_id in mapped
        ]
    else:
        candidates = [
            (ff, asset_id, blur_scores[asset_id])
            for ff, asset_id in mapped
            if asset_id in blur_scores
        ]

    if not candidates:
        return None
    return min(candidates, key=lambda x: x[2])


def get_frigate_filename_for_asset(person_name: str, asset_id: str) -> str | None:
    """Return the Frigate training filename mapped to this asset ID, or None."""
    data = _load(UPLOAD_TRACKER_FILE)
    entry = _migrate_entry(data.get("by_person", {}).get(person_name, {}))
    for frigate_filename, aid in entry["frigate_files"].items():
        if aid == asset_id:
            return frigate_filename
    return None


def find_by_crop_dimension(size: int) -> list[dict]:
    """Return all tracked crops whose width or height matches `size` pixels.

    Returns a list of dicts: {person, asset_id, width, height, blur_score, frigate_filename}.
    frigate_filename is None when the Frigate mapping was lost to a reconciliation race.
    """
    data = _load(UPLOAD_TRACKER_FILE)
    results = []
    for person_name, raw_entry in data.get("by_person", {}).items():
        entry = _migrate_entry(raw_entry)
        scores = entry.get("scores", {})
        frigate_files = entry.get("frigate_files", {})
        asset_to_frigate = {v: k for k, v in frigate_files.items()}
        frigate_scores = entry.get("frigate_scores", {})
        for asset_id, dims in entry.get("crop_dims", {}).items():
            w, h = dims[0], dims[1]
            if w == size or h == size:
                results.append({
                    "person": person_name,
                    "asset_id": asset_id,
                    "width": w,
                    "height": h,
                    "blur_score": scores.get(asset_id),
                    "frigate_score": frigate_scores.get(asset_id),
                    "frigate_filename": asset_to_frigate.get(asset_id),
                })
    return results


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
    """Return {person_name: {uploaded, rejected, frigate_count, scores, frigate_files}} for display/capacity."""
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
            "frigate_files": u_entry.get("frigate_files", {}) if isinstance(u_entry, dict) else {},
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
