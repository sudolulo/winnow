"""Persistent tracker for Immich asset IDs already uploaded/rejected by Frigate.

Two separate JSON files in DATA_DIR:
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
    "frigate_scores": {"immich-id-1": 0.87},                   # Frigate recognition confidence (0-1) pre-upload
    "frigate_files":  {"PersonName-123.webp": "immich-id-1"},  # Frigate filename → asset ID
    "crop_dims":      {"immich-id-1": [640, 480]},             # crop pixel dimensions at upload time
    "frigate_count":  42                                       # last known Frigate training image count
  }

frigate_scores stores pre-upload recognize scores (0-1 sigmoid-mapped cosine
similarity). High score = the existing training set already covers this face
condition well. Low score = a gap — novel/diverse for the training set.

frigate_files only contains files winnow uploaded — files added manually through
Frigate's UI are never mapped here and are never touched by quality replacement.
"""

import fcntl
import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path

from .frigate_api import _get_frigate_url, delete_frigate_person_files

logger = logging.getLogger(__name__)

UPLOAD_TRACKER_FILE = "frigate_uploaded_ids.json"
REJECT_TRACKER_FILE = "frigate_rejected_ids.json"
LOCK_FILE = ".tracker.lock"

# Number of deferred _save calls a batch accumulates before it is flushed to disk
# early. Bounds how many marks a crash mid-batch (SIGKILL/OOM/host crash) can lose —
# without this, begin_batch()/flush_batch() defer every write for an entire
# per-person upload loop, so a crash could lose every mark from that person's batch
# even though the files are already live in Frigate.
_BATCH_FLUSH_EVERY = 10

# Write-through in-memory cache keyed by the resolved file path.
# Reduces per-call JSON reads from O(calls) to O(1) after the first load.
# Keyed by full path so tests with isolated tmp dirs never share entries.
_cache: dict[str, dict] = {}
_deferred: set[str] = set()  # paths whose disk writes are batched until flush_batch()
_dirty: set[str] = set()    # deferred paths that received at least one _save during the batch
_batch_writes: dict[str, int] = {}  # deferred _save calls since the last disk write, per path

# Cross-process lock guarding the load-mutate-save cycle below. Two winnow
# invocations against the same DATA_DIR (e.g. a scheduled run overlapping a
# manual `docker exec`, which the docs explicitly instruct) can otherwise race
# a read-modify-write and silently lose whichever one saves first. Reentrant
# within a single process (depth-counted) so begin_batch()/flush_batch() pairs
# and nested tracker calls made while a batch is open don't self-deadlock.
_lock_fd: int | None = None
_lock_depth = 0


def _tracker_path(filename: str) -> Path:
    try:
        from .config import Config
        return Path(Config.DATA_DIR) / filename
    except (ImportError, AttributeError):
        return Path(filename)


def _lock_path() -> Path:
    return _tracker_path(LOCK_FILE)


def _acquire_lock() -> None:
    """Acquire the cross-process tracker lock. Reentrant within this process."""
    global _lock_fd, _lock_depth
    if _lock_depth == 0:
        path = _lock_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX)  # blocks until any other process's lock is released
        _lock_fd = fd
        # Cache entries not part of an in-progress deferred batch may be stale —
        # another process could have written to disk since we last read them.
        # Drop them so the critical section that follows re-reads from disk.
        for key in list(_cache):
            if key not in _deferred:
                del _cache[key]
    _lock_depth += 1


def _release_lock() -> None:
    global _lock_fd, _lock_depth
    if _lock_depth <= 0:
        return
    _lock_depth -= 1
    if _lock_depth == 0 and _lock_fd is not None:
        fcntl.flock(_lock_fd, fcntl.LOCK_UN)
        os.close(_lock_fd)
        _lock_fd = None


@contextmanager
def _locked():
    """Context manager wrapping a single load-mutate-save cycle in the tracker lock."""
    _acquire_lock()
    try:
        yield
    finally:
        _release_lock()


def _load(filename: str) -> dict:
    path = _tracker_path(filename)
    key = str(path)
    if key in _cache:
        return _cache[key]
    data: dict = {}
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not load tracker {filename}: {e}")
    _cache[key] = data
    return data


def _write_to_disk(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _save(filename: str, data: dict) -> None:
    path = _tracker_path(filename)
    key = str(path)
    if key in _deferred:
        _cache[key] = data  # accumulate in cache; disk write deferred until flush_batch()
        _dirty.add(key)
        # Flush early every _BATCH_FLUSH_EVERY marks so a crash mid-batch (SIGKILL/OOM/
        # host crash) loses at most a bounded number of marks instead of the whole batch.
        # Safe without re-acquiring the lock: begin_batch() already holds it for the
        # duration of the batch.
        _batch_writes[key] = _batch_writes.get(key, 0) + 1
        if _batch_writes[key] >= _BATCH_FLUSH_EVERY:
            _write_to_disk(path, data)
            _dirty.discard(key)
            _batch_writes[key] = 0
        return
    _write_to_disk(path, data)
    _cache[key] = data  # update cache only after successful write


def begin_batch(filename: str) -> None:
    """Defer tracker disk writes for filename. All _save calls accumulate in the
    in-memory cache until flush_batch() is called (with periodic early flushes —
    see _BATCH_FLUSH_EVERY). Use around per-person upload loops to reduce N writes
    towards 1.

    Acquires the cross-process tracker lock, held until the matching flush_batch()
    releases it, so a concurrent winnow invocation against the same DATA_DIR can't
    interleave a read-modify-write with this batch.

    If a previous batch for this file was interrupted before flush_batch() was called
    (e.g. an exception escaped the upload loop), the leftover cache state is flushed
    to disk here before starting fresh so that partial progress is not silently lost.
    """
    _acquire_lock()
    path = _tracker_path(filename)
    key = str(path)
    if key in _deferred and key in _dirty:
        try:
            _write_to_disk(path, _cache[key])
        except Exception:
            logger.warning(
                "begin_batch: could not flush leftover deferred state for %s"
                " — partial progress may be lost",
                path,
            )
        _deferred.discard(key)
        _dirty.discard(key)
    _deferred.add(key)
    _batch_writes[key] = 0


def flush_batch(filename: str) -> None:
    """Write the accumulated cache state for filename to disk and release the tracker lock
    acquired by the matching begin_batch().

    The lock release always runs, even if the disk write raises, so a write failure
    (e.g. a full disk) can't leave the tracker lock held for the rest of the process.
    """
    path = _tracker_path(filename)
    key = str(path)
    try:
        if key in _dirty and key in _cache:
            _write_to_disk(path, _cache[key])
    finally:
        _deferred.discard(key)
        _dirty.discard(key)
        _batch_writes.pop(key, None)
        _release_lock()


def _flat_key(filename: str) -> str:
    return "uploaded_asset_ids" if filename == UPLOAD_TRACKER_FILE else "rejected_asset_ids"


def _get_ids(entry: list | dict) -> list[str]:
    """Extract asset_ids from either the old list format or the new dict format."""
    if isinstance(entry, list):
        return entry
    return entry.get("asset_ids", [])


def _migrate_entry(entry: list | dict) -> dict:
    """Ensure by_person entry is in the current dict format."""
    if isinstance(entry, list):
        return {"asset_ids": sorted(entry), "scores": {}, "frigate_scores": {}, "frigate_files": {}, "crop_dims": {}}
    # Copy top-level and all nested dicts so callers' mutations never reach the cache.
    result = dict(entry)
    result["asset_ids"] = list(result.get("asset_ids", []))
    result["scores"] = dict(result.get("scores", {}))
    result["frigate_scores"] = dict(result.get("frigate_scores", {}))
    result["frigate_files"] = dict(result.get("frigate_files", {}))
    result["crop_dims"] = dict(result.get("crop_dims", {}))
    return result


def _mark(
    filename: str,
    asset_id: str,
    person_name: str | None,
    score: float | None = None,
    crop_dims: tuple[int, int] | None = None,
    frigate_score: float | None = None,
) -> None:
    if not person_name:
        logger.warning("_mark called with empty person_name for asset %s — asset not recorded", asset_id)
        return
    with _locked():
        data = _load(filename)
        by_person = dict(data.get("by_person", {}))
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
        new_data = dict(data)
        new_data["by_person"] = by_person
        _save(filename, new_data)
    logger.debug("Marked %s in %s (%s)", asset_id, filename, person_name)


# ── Public API ────────────────────────────────────────────────────────────────

def load_uploaded_ids() -> set[str]:
    """Return all asset IDs recorded as uploaded. Derives from by_person (primary)
    plus any legacy flat list still present in old tracker files."""
    data = _load(UPLOAD_TRACKER_FILE)
    ids = {aid for e in data.get("by_person", {}).values() for aid in _get_ids(e)}
    ids.update(data.get("uploaded_asset_ids", []))  # backward compat with pre-0.6.1 files
    return ids


def load_rejected_ids() -> set[str]:
    """Return all asset IDs recorded as rejected. Derives from by_person (primary)
    plus any legacy flat list still present in old tracker files."""
    data = _load(REJECT_TRACKER_FILE)
    ids = {aid for e in data.get("by_person", {}).values() for aid in _get_ids(e)}
    ids.update(data.get("rejected_asset_ids", []))  # backward compat with pre-0.6.1 files
    return ids


def mark_uploaded(
    asset_id: str,
    person_name: str | None = None,
    score: float | None = None,
    crop_dims: tuple[int, int] | None = None,
    frigate_score: float | None = None,
) -> None:
    _mark(UPLOAD_TRACKER_FILE, asset_id, person_name, score=score, crop_dims=crop_dims, frigate_score=frigate_score)


def mark_rejected(asset_id: str, person_name: str | None = None) -> None:
    _mark(REJECT_TRACKER_FILE, asset_id, person_name)




def record_frigate_file(person_name: str, frigate_filename: str, asset_id: str) -> None:
    """Record a single Frigate filename → asset_id mapping."""
    record_frigate_files_batch(person_name, {frigate_filename: asset_id})


def record_frigate_files_batch(person_name: str, mappings: dict[str, str]) -> None:
    """Record multiple Frigate filename → asset_id mappings in a single load/save."""
    if not mappings:
        return
    with _locked():
        src = _load(UPLOAD_TRACKER_FILE)
        by_person = dict(src.get("by_person", {}))
        entry = _migrate_entry(by_person.get(person_name, {}))
        entry["frigate_files"].update(mappings)
        by_person[person_name] = entry
        data = dict(src)
        data["by_person"] = by_person
        _save(UPLOAD_TRACKER_FILE, data)
    logger.debug(f"Batch-mapped {len(mappings)} Frigate file(s) for {person_name}")


def remove_frigate_file(person_name: str, frigate_filename: str) -> None:
    """Remove a Frigate filename from the mapping after it has been deleted.

    Does NOT unmark the source asset_id — the deletion was deliberate and
    we don't want to re-upload the inferior image on the next run.
    """
    remove_frigate_files_batch(person_name, [frigate_filename])


def remove_frigate_files_batch(person_name: str, frigate_filenames: list[str]) -> None:
    """Remove multiple Frigate filenames in a single load/save."""
    with _locked():
        src = _load(UPLOAD_TRACKER_FILE)
        raw = src.get("by_person", {}).get(person_name)
        if raw is None:
            return
        entry = _migrate_entry(raw)
        for fn in frigate_filenames:
            asset_id = entry["frigate_files"].pop(fn, None)
            if asset_id is not None and asset_id not in entry["frigate_files"].values():
                entry["frigate_scores"].pop(asset_id, None)
        by_person = dict(src.get("by_person", {}))  # copy so assignment does not mutate the cache
        by_person[person_name] = entry
        data = dict(src)
        data["by_person"] = by_person
        _save(UPLOAD_TRACKER_FILE, data)
    logger.debug(f"Removed {len(frigate_filenames)} Frigate file mapping(s) for {person_name}")


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
    raw = data.get("by_person", {}).get(person_name)
    if not raw or isinstance(raw, list):
        return False
    frigate_files = raw.get("frigate_files", {})
    frigate_scores = raw.get("frigate_scores", {})
    return any(asset_id in frigate_scores for asset_id in frigate_files.values())


def _pick_mapped_file(
    person_name: str, score_key: str, *, highest: bool, exclude: set[str] | None = None
) -> tuple[str, str, float] | None:
    data = _load(UPLOAD_TRACKER_FILE)
    entry = _migrate_entry(data.get("by_person", {}).get(person_name, {}))
    scores = entry.get(score_key, {})
    seen_assets: set[str] = set()
    candidates = []
    for ff, asset_id in entry.get("frigate_files", {}).items():
        if (exclude is None or ff not in exclude) and asset_id in scores and asset_id not in seen_assets:
            seen_assets.add(asset_id)
            candidates.append((ff, asset_id, scores[asset_id]))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[2]) if highest else min(candidates, key=lambda x: x[2])


def get_lowest_quality_mapped_file(
    person_name: str, exclude: set[str] | None = None
) -> tuple[str, str, float] | None:
    """Return (frigate_filename, asset_id, score) for the mapped file with the lowest
    blur score, or None if no mapped files with known scores exist.

    Used for quality replacement when no Frigate scores are available.
    Pass `exclude` to skip files that failed to delete this run.
    """
    return _pick_mapped_file(person_name, "scores", highest=False, exclude=exclude)


def get_most_redundant_mapped_file(
    person_name: str, exclude: set[str] | None = None
) -> tuple[str, str, float] | None:
    """Return (frigate_filename, asset_id, score) for the mapped file with the highest
    Frigate recognition score, or None if no mapped files with Frigate scores exist.

    High Frigate score = the training set already covers this face condition well
    = the most redundant file and therefore the best replacement target.
    Pass `exclude` to skip files that failed to delete this run.
    """
    return _pick_mapped_file(person_name, "frigate_scores", highest=True, exclude=exclude)


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
        asset_to_frigate: dict[str, str] = {}
        for fn, aid in frigate_files.items():
            asset_to_frigate.setdefault(aid, fn)  # first-seen wins; plain inversion silently drops duplicates
        frigate_scores = entry.get("frigate_scores", {})
        for asset_id, dims in entry.get("crop_dims", {}).items():
            if not isinstance(dims, (list, tuple)) or len(dims) < 2:
                continue
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
    with _locked():
        data = _load(UPLOAD_TRACKER_FILE)
        by_person = dict(data.get("by_person", {}))
        entry = _migrate_entry(by_person.get(person_name, {}))
        entry["frigate_count"] = count
        by_person[person_name] = entry
        new_data = dict(data)
        new_data["by_person"] = by_person
        _save(UPLOAD_TRACKER_FILE, new_data)


def reset_all_people() -> None:
    """Reset all tracking data in two writes (O(P) Frigate API calls, O(1) disk writes).

    Preferred over calling reset_person() in a loop when RESET_PERSON=* — that
    approach is O(P²) because each call rebuilds the flat list from all remaining entries.
    """
    with _locked():
        upload_data = _load(UPLOAD_TRACKER_FILE)
        frigate_url = _get_frigate_url()
        if not frigate_url:
            logger.info("FRIGATE_URL not set — skipping Frigate file deletion")
        for person_name, raw_entry in upload_data.get("by_person", {}).items():
            entry = _migrate_entry(raw_entry)
            frigate_filenames = list(entry.get("frigate_files", {}).keys())
            if not frigate_filenames:
                continue
            if frigate_url:
                if delete_frigate_person_files(person_name, frigate_filenames):
                    logger.info(f"Deleted {len(frigate_filenames)} Frigate file(s) for {person_name}")
                else:
                    logger.warning(
                        f"Could not delete Frigate files for {person_name} — tracker reset proceeding anyway"
                    )
        _save(UPLOAD_TRACKER_FILE, {})
        _save(REJECT_TRACKER_FILE, {})
    logger.info("Reset all tracking data")


def reset_person(person_name: str) -> None:
    """Remove all uploaded and rejected records for a given person.

    Also deletes winnow-managed Frigate training files so the next run starts
    clean rather than uploading on top of orphaned files. Manually-added Frigate
    files (not in frigate_files) are never touched. Proceeds with tracker reset
    even if Frigate is unreachable.
    """
    with _locked():
        upload_data = _load(UPLOAD_TRACKER_FILE)
        entry = _migrate_entry(upload_data.get("by_person", {}).get(person_name, {}))
        frigate_filenames = list(entry.get("frigate_files", {}).keys())
        if frigate_filenames:
            if not _get_frigate_url():
                logger.info(f"FRIGATE_URL not set — skipping Frigate file deletion for {person_name}")
            elif delete_frigate_person_files(person_name, frigate_filenames):
                logger.info(f"Deleted {len(frigate_filenames)} Frigate file(s) for {person_name}")
            else:
                logger.warning(f"Could not delete Frigate files for {person_name} — tracker reset proceeding anyway")

        changed = False
        for filename in (UPLOAD_TRACKER_FILE, REJECT_TRACKER_FILE):
            src = upload_data if filename == UPLOAD_TRACKER_FILE else _load(REJECT_TRACKER_FILE)
            by_person = dict(src.get("by_person", {}))  # copy so pop() does not mutate the cache
            tracker_entry = by_person.pop(person_name, None)
            if tracker_entry is not None:
                data = dict(src)
                data["by_person"] = by_person
                flat_key = _flat_key(filename)
                person_ids = set(_get_ids(tracker_entry))
                if person_ids and flat_key in data and not isinstance(data[flat_key], list):
                    logger.warning(
                        "reset_person: %s has unexpected type for %s (%s) — skipping flat-list cleanup;"
                        " all persons' legacy IDs in this field are unaffected but unreadable",
                        filename, flat_key, type(data[flat_key]).__name__,
                    )
                elif person_ids and flat_key in data:
                    data[flat_key] = sorted(set(data[flat_key]) - person_ids)
                _save(filename, data)
                changed = True
    if changed:
        logger.info(f"Reset tracking data for {person_name}")
    else:
        logger.debug(f"reset_person: no tracking data found for {person_name}")


def get_person_summary() -> dict[str, dict]:
    """Return {person_name: {uploaded, rejected, frigate_count, scores, frigate_files}} for display/capacity."""
    uploaded_data = _load(UPLOAD_TRACKER_FILE).get("by_person", {})
    rejected_data = _load(REJECT_TRACKER_FILE).get("by_person", {})
    names = set(uploaded_data) | set(rejected_data)
    result = {}
    for name in sorted(names):
        u_entry = _migrate_entry(uploaded_data.get(name, {}))
        r_entry = _migrate_entry(rejected_data.get(name, {}))
        result[name] = {
            "uploaded": len(u_entry["asset_ids"]),
            "rejected": len(r_entry["asset_ids"]),
            "frigate_count": u_entry.get("frigate_count"),
            "scores": u_entry["scores"],
            "frigate_files": u_entry["frigate_files"],
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
