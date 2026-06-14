"""Persistent tracker for Immich asset IDs uploaded/rejected by Frigate.

Uses a local SQLite database (frigate_tracker.db) in DATA_DIR.

Schema
------
tracked_assets   — one row per (asset_id, status) pair
frigate_files    — Frigate filename → Immich asset_id mapping
person_metadata  — last-known Frigate training image count per person

Migration
---------
On first open, if the old JSON files exist and the tables are empty, their
data is migrated automatically. The JSON files are then renamed to .json.bak.
"""

import json
import logging
import os
import sqlite3
from pathlib import Path

from .frigate_api import delete_frigate_person_files

logger = logging.getLogger(__name__)

# Legacy JSON filenames (for migration)
_UPLOAD_JSON = "frigate_uploaded_ids.json"
_REJECT_JSON = "frigate_rejected_ids.json"
_DB_NAME = "frigate_tracker.db"

_DDL = """
CREATE TABLE IF NOT EXISTS tracked_assets (
    asset_id      TEXT NOT NULL,
    person_name   TEXT,
    status        TEXT NOT NULL CHECK(status IN ('uploaded', 'rejected')),
    blur_score    REAL,
    crop_width    INTEGER,
    crop_height   INTEGER,
    frigate_score REAL,
    PRIMARY KEY (asset_id, status)
);

CREATE TABLE IF NOT EXISTS frigate_files (
    frigate_filename TEXT PRIMARY KEY,
    person_name      TEXT NOT NULL,
    asset_id         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS person_metadata (
    person_name   TEXT PRIMARY KEY,
    frigate_count INTEGER
);
"""

# Module-level connection state — re-opened when DATA_DIR changes (test isolation)
_conn: sqlite3.Connection | None = None
_conn_path: str | None = None


def _get_conn() -> sqlite3.Connection:
    """Return (or create) the module-level SQLite connection.

    Re-opens the connection when Config.DATA_DIR has changed — this provides
    test isolation when the isolated_cache fixture sets a new tmp directory and
    calls _Config.reset().
    """
    global _conn, _conn_path

    from .config import Config
    data_dir = Config.DATA_DIR
    db_path = str(Path(data_dir) / _DB_NAME)

    if _conn is not None and _conn_path != db_path:
        try:
            _conn.close()
        except Exception as e:
            logger.debug("Failed to close previous SQLite connection: %s", e)
        _conn = None

    if _conn is None:
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(db_path, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA foreign_keys=ON")
        _conn.executescript(_DDL)
        _conn.commit()
        _conn_path = db_path
        _maybe_migrate(data_dir, _conn)

    return _conn


# ---------------------------------------------------------------------------
# JSON → SQLite migration
# ---------------------------------------------------------------------------

def _maybe_migrate(data_dir: str, conn: sqlite3.Connection) -> None:
    """If the old JSON files exist and DB is empty, migrate and rename them."""
    base = Path(data_dir)
    upload_json = base / _UPLOAD_JSON
    reject_json = base / _REJECT_JSON

    if not upload_json.exists() and not reject_json.exists():
        return

    # No row-count guard here: INSERT OR IGNORE makes migration idempotent, so it
    # is safe to re-run if a previous attempt renamed one file but not the other
    # (e.g. a PermissionError on the second rename would have left the first file's
    # data committed but the second file un-renamed and un-migrated).

    logger.info("Migrating JSON tracker files to SQLite in %s", data_dir)

    try:
        with conn:
            if upload_json.exists():
                _migrate_json_data(conn, json.loads(upload_json.read_text()), "uploaded")
            if reject_json.exists():
                _migrate_json_data(conn, json.loads(reject_json.read_text()), "rejected")
    except Exception as exc:
        logger.warning("JSON migration failed, will retry next run: %s", exc)
        return

    # Rename each file independently so a failure on one does not prevent the
    # other from being marked complete on this run.
    for json_path in (upload_json, reject_json):
        if json_path.exists():
            try:
                json_path.rename(json_path.with_suffix(".json.bak"))
            except OSError as exc:
                logger.warning("Could not rename %s after migration: %s", json_path, exc)

    logger.info("JSON → SQLite migration complete")


def _migrate_json_data(conn: sqlite3.Connection, data: dict, status: str) -> None:
    """Insert one JSON tracker file's data into SQLite tables."""
    flat_key = "uploaded_asset_ids" if status == "uploaded" else "rejected_asset_ids"
    flat_ids: set[str] = set(data.get(flat_key, []))
    person_covered: set[str] = set()

    for person_name, raw_entry in data.get("by_person", {}).items():
        if isinstance(raw_entry, list):
            entry: dict = {"asset_ids": raw_entry, "scores": {}, "frigate_scores": {},
                           "frigate_files": {}, "crop_dims": {}}
        else:
            entry = {
                "asset_ids": raw_entry.get("asset_ids", []),
                "scores": raw_entry.get("scores", {}),
                "frigate_scores": raw_entry.get("frigate_scores", {}),
                "frigate_files": raw_entry.get("frigate_files", {}),
                "crop_dims": raw_entry.get("crop_dims", {}),
                "frigate_count": raw_entry.get("frigate_count"),
            }

        for asset_id in entry["asset_ids"]:
            person_covered.add(asset_id)
            dims = entry.get("crop_dims", {}).get(asset_id)
            conn.execute(
                """INSERT OR IGNORE INTO tracked_assets
                   (asset_id, person_name, status, blur_score,
                    crop_width, crop_height, frigate_score)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    asset_id,
                    person_name,
                    status,
                    entry.get("scores", {}).get(asset_id),
                    dims[0] if dims else None,
                    dims[1] if dims else None,
                    entry.get("frigate_scores", {}).get(asset_id) if status == "uploaded" else None,
                ),
            )

        if status == "uploaded":
            for ff, aid in entry.get("frigate_files", {}).items():
                conn.execute(
                    "INSERT OR IGNORE INTO frigate_files (frigate_filename, person_name, asset_id) VALUES (?, ?, ?)",
                    (ff, person_name, aid),
                )
            fc = entry.get("frigate_count")
            if fc is not None:
                conn.execute(
                    "INSERT OR REPLACE INTO person_metadata (person_name, frigate_count) VALUES (?, ?)",
                    (person_name, fc),
                )

    # Flat IDs not covered by any by_person entry → insert with NULL person
    for asset_id in flat_ids - person_covered:
        conn.execute(
            "INSERT OR IGNORE INTO tracked_assets (asset_id, person_name, status) VALUES (?, NULL, ?)",
            (asset_id, status),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_uploaded_ids() -> set[str]:
    conn = _get_conn()
    rows = conn.execute("SELECT asset_id FROM tracked_assets WHERE status='uploaded'").fetchall()
    return {r[0] for r in rows}


def load_rejected_ids() -> set[str]:
    conn = _get_conn()
    rows = conn.execute("SELECT asset_id FROM tracked_assets WHERE status='rejected'").fetchall()
    return {r[0] for r in rows}


def mark_uploaded(
    asset_id: str,
    person_name: str | None = None,
    score: float | None = None,
    crop_dims: tuple[int, int] | None = None,
    frigate_score: float | None = None,
) -> None:
    conn = _get_conn()
    with conn:
        conn.execute(
            """INSERT OR REPLACE INTO tracked_assets
               (asset_id, person_name, status, blur_score, crop_width, crop_height, frigate_score)
               VALUES (?, ?, 'uploaded', ?, ?, ?, ?)""",
            (
                asset_id,
                person_name,
                round(score, 4) if score is not None else None,
                crop_dims[0] if crop_dims else None,
                crop_dims[1] if crop_dims else None,
                round(frigate_score, 4) if frigate_score is not None else None,
            ),
        )
    logger.debug("Marked %s as uploaded (%s)", asset_id, person_name)


def mark_rejected(asset_id: str, person_name: str | None = None) -> None:
    conn = _get_conn()
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO tracked_assets (asset_id, person_name, status) VALUES (?, ?, 'rejected')",
            (asset_id, person_name),
        )
    logger.debug("Marked %s as rejected (%s)", asset_id, person_name)


def record_frigate_files_batch(person_name: str, mappings: dict[str, str]) -> None:
    """Record multiple Frigate filename → asset_id mappings in a single transaction."""
    if not mappings:
        return
    conn = _get_conn()
    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO frigate_files (frigate_filename, person_name, asset_id) VALUES (?, ?, ?)",
            [(ff, person_name, aid) for ff, aid in mappings.items()],
        )
    logger.debug("Batch-mapped %s Frigate file(s) for %s", len(mappings), person_name)


def remove_frigate_file(person_name: str, frigate_filename: str) -> None:
    """Remove a Frigate filename mapping and clear its asset's frigate_score.

    Does NOT unmark the source asset_id — the deletion was deliberate.
    """
    conn = _get_conn()
    with conn:
        row = conn.execute(
            "SELECT asset_id FROM frigate_files WHERE frigate_filename=? AND person_name=?",
            (frigate_filename, person_name),
        ).fetchone()
        conn.execute(
            "DELETE FROM frigate_files WHERE frigate_filename=? AND person_name=?",
            (frigate_filename, person_name),
        )
        if row:
            conn.execute(
                "UPDATE tracked_assets SET frigate_score=NULL WHERE asset_id=? AND person_name=?",
                (row["asset_id"], person_name),
            )
    logger.debug("Removed Frigate file mapping %s (%s)", frigate_filename, person_name)


def get_tracked_frigate_file_count(person_name: str) -> int:
    """Return the number of Frigate training files winnow has mapped for this person."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT COUNT(*) FROM frigate_files WHERE person_name=?", (person_name,)
    ).fetchone()
    return row[0]


def get_tracked_frigate_filenames(person_name: str) -> set[str]:
    """Return the set of Frigate filenames currently mapped for a person."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT frigate_filename FROM frigate_files WHERE person_name=?", (person_name,)
    ).fetchall()
    return {r[0] for r in rows}


def has_frigate_scores(person_name: str) -> bool:
    """Return True if any mapped file for this person has a stored Frigate recognition score."""
    conn = _get_conn()
    row = conn.execute(
        """SELECT COUNT(*) FROM frigate_files ff
           JOIN tracked_assets ta ON ta.asset_id=ff.asset_id AND ta.person_name=ff.person_name
           WHERE ff.person_name=? AND ta.frigate_score IS NOT NULL""",
        (person_name,),
    ).fetchone()
    return row[0] > 0


_VALID_SCORE_COLS = frozenset({"blur_score", "frigate_score"})


def _pick_mapped_file(
    person_name: str, score_col: str, *, highest: bool, exclude: set[str] | None = None
) -> tuple[str, str, float] | None:
    if score_col not in _VALID_SCORE_COLS:
        raise ValueError(f"Invalid score column: {score_col!r}")
    conn = _get_conn()
    order = "DESC" if highest else "ASC"
    rows = conn.execute(
        f"""SELECT ff.frigate_filename, ff.asset_id, ta.{score_col}
            FROM frigate_files ff
            JOIN tracked_assets ta ON ta.asset_id=ff.asset_id AND ta.person_name=ff.person_name
            WHERE ff.person_name=? AND ta.{score_col} IS NOT NULL
            ORDER BY ta.{score_col} {order}""",
        (person_name,),
    ).fetchall()
    for row in rows:
        if exclude is None or row[0] not in exclude:
            return (row[0], row[1], row[2])
    return None


def get_lowest_quality_mapped_file(
    person_name: str, exclude: set[str] | None = None
) -> tuple[str, str, float] | None:
    """Return (frigate_filename, asset_id, score) for the mapped file with the lowest blur score."""
    return _pick_mapped_file(person_name, "blur_score", highest=False, exclude=exclude)


def get_most_redundant_mapped_file(
    person_name: str, exclude: set[str] | None = None
) -> tuple[str, str, float] | None:
    """Return (frigate_filename, asset_id, score) for the mapped file with the highest Frigate score."""
    return _pick_mapped_file(person_name, "frigate_score", highest=True, exclude=exclude)


def get_frigate_filename_for_asset(person_name: str, asset_id: str) -> str | None:
    """Return the Frigate training filename mapped to this asset ID, or None."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT frigate_filename FROM frigate_files WHERE person_name=? AND asset_id=?",
        (person_name, asset_id),
    ).fetchone()
    return row[0] if row else None


def find_by_crop_dimension(size: int) -> list[dict]:
    """Return all tracked crops whose width or height matches `size` pixels.

    Returns a list of dicts: {person, asset_id, width, height, blur_score, frigate_score, frigate_filename}.
    """
    conn = _get_conn()
    rows = conn.execute(
        """SELECT ta.person_name, ta.asset_id, ta.crop_width, ta.crop_height,
                  ta.blur_score, ta.frigate_score, ff.frigate_filename
           FROM tracked_assets ta
           LEFT JOIN frigate_files ff ON ff.asset_id=ta.asset_id AND ff.person_name=ta.person_name
           WHERE ta.status='uploaded' AND (ta.crop_width=? OR ta.crop_height=?)""",
        (size, size),
    ).fetchall()
    return [
        {
            "person": r["person_name"],
            "asset_id": r["asset_id"],
            "width": r["crop_width"],
            "height": r["crop_height"],
            "blur_score": r["blur_score"],
            "frigate_score": r["frigate_score"],
            "frigate_filename": r["frigate_filename"],
        }
        for r in rows
    ]


def update_frigate_count(person_name: str, count: int) -> None:
    """Record Frigate's authoritative training image count for a person."""
    conn = _get_conn()
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO person_metadata (person_name, frigate_count) VALUES (?, ?)",
            (person_name, count),
        )


def reset_person(person_name: str) -> None:
    """Remove all uploaded and rejected records for a given person.

    Also deletes winnow-managed Frigate training files so the next run starts
    clean rather than uploading on top of orphaned files.
    """
    conn = _get_conn()

    # Collect Frigate filenames before deleting
    frigate_filenames = list(get_tracked_frigate_filenames(person_name))
    if frigate_filenames:
        if not os.environ.get("FRIGATE_URL", "").strip():
            logger.info("FRIGATE_URL not set — skipping Frigate file deletion for %s", person_name)
        elif delete_frigate_person_files(person_name, frigate_filenames):
            logger.info("Deleted %s Frigate file(s) for %s", len(frigate_filenames), person_name)
        else:
            logger.warning(
                "Could not delete Frigate files for %s — tracker reset proceeding anyway", person_name
            )

    with conn:
        conn.execute("DELETE FROM frigate_files WHERE person_name=?", (person_name,))
        conn.execute("DELETE FROM tracked_assets WHERE person_name=?", (person_name,))
        conn.execute("DELETE FROM person_metadata WHERE person_name=?", (person_name,))

    logger.info("Reset tracking data for %s", person_name)


def get_person_summary() -> dict[str, dict]:
    """Return {person_name: {uploaded, rejected, frigate_count, scores, frigate_files}} for display/capacity."""
    conn = _get_conn()

    # Counts per person per status
    rows = conn.execute(
        """SELECT person_name, status, COUNT(*) AS cnt
           FROM tracked_assets WHERE person_name IS NOT NULL
           GROUP BY person_name, status"""
    ).fetchall()

    summary: dict[str, dict] = {}
    for r in rows:
        name = r["person_name"]
        if name not in summary:
            summary[name] = {"uploaded": 0, "rejected": 0, "frigate_count": None,
                             "scores": {}, "frigate_files": {}}
        summary[name][r["status"]] = r["cnt"]

    # Scores for uploaded assets
    score_rows = conn.execute(
        """SELECT person_name, asset_id, blur_score
           FROM tracked_assets
           WHERE status='uploaded' AND person_name IS NOT NULL AND blur_score IS NOT NULL"""
    ).fetchall()
    for r in score_rows:
        name = r["person_name"]
        if name not in summary:
            summary[name] = {"uploaded": 0, "rejected": 0, "frigate_count": None,
                             "scores": {}, "frigate_files": {}}
        summary[name]["scores"][r["asset_id"]] = r["blur_score"]

    # Frigate file mappings
    ff_rows = conn.execute(
        "SELECT person_name, frigate_filename, asset_id FROM frigate_files"
    ).fetchall()
    for r in ff_rows:
        name = r["person_name"]
        if name not in summary:
            summary[name] = {"uploaded": 0, "rejected": 0, "frigate_count": None,
                             "scores": {}, "frigate_files": {}}
        summary[name]["frigate_files"][r["frigate_filename"]] = r["asset_id"]

    # Frigate counts
    meta_rows = conn.execute(
        "SELECT person_name, frigate_count FROM person_metadata"
    ).fetchall()
    for r in meta_rows:
        name = r["person_name"]
        if name not in summary:
            summary[name] = {"uploaded": 0, "rejected": 0, "frigate_count": None,
                             "scores": {}, "frigate_files": {}}
        summary[name]["frigate_count"] = r["frigate_count"]

    return dict(sorted(summary.items()))


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
        logger.info("Skipping %s assets already uploaded or rejected by Frigate", skipped)
    return new_ids
