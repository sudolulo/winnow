"""Tests for upload tracker — mark, filter, reset, and summary logic."""

import fcntl
import json
import os

import pytest


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    """Point tracker at a temp directory so tests don't touch real cache files."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from winnow.config import _Config
    _Config.reset()
    yield tmp_path
    _Config.reset()


def test_filter_returns_all_when_empty():
    from winnow.upload_tracker import filter_already_uploaded
    ids = ["a1", "b2", "c3"]
    assert filter_already_uploaded(ids) == ids


def test_mark_uploaded_excludes_from_filter():
    from winnow.upload_tracker import filter_already_uploaded, mark_uploaded
    mark_uploaded("a1", person_name="Alice")
    result = filter_already_uploaded(["a1", "b2"])
    assert result == ["b2"]


def test_mark_rejected_excludes_from_filter():
    from winnow.upload_tracker import filter_already_uploaded, mark_rejected
    mark_rejected("x9", person_name="Bob")
    result = filter_already_uploaded(["x9", "y8"])
    assert result == ["y8"]


def test_retry_rejected_includes_rejected():
    from winnow.upload_tracker import filter_already_uploaded, mark_rejected
    mark_rejected("x9", person_name="Bob")
    result = filter_already_uploaded(["x9", "y8"], retry_rejected=True)
    assert "x9" in result


def test_reset_person_clears_records():
    from winnow.upload_tracker import filter_already_uploaded, mark_uploaded, reset_person
    mark_uploaded("a1", person_name="Alice")
    mark_uploaded("a2", person_name="Alice")
    mark_uploaded("b1", person_name="Bob")
    reset_person("Alice")
    assert filter_already_uploaded(["a1", "a2"]) == ["a1", "a2"]
    assert filter_already_uploaded(["b1"]) == []


def test_get_person_summary():
    from winnow.upload_tracker import get_person_summary, mark_rejected, mark_uploaded
    mark_uploaded("a1", person_name="Alice")
    mark_uploaded("a2", person_name="Alice")
    mark_rejected("a3", person_name="Alice")
    mark_uploaded("b1", person_name="Bob")
    summary = get_person_summary()
    assert summary["Alice"]["uploaded"] == 2
    assert summary["Alice"]["rejected"] == 1
    assert summary["Bob"]["uploaded"] == 1
    assert summary["Bob"]["rejected"] == 0


def test_duplicate_marks_are_idempotent():
    from winnow.upload_tracker import filter_already_uploaded, mark_uploaded
    mark_uploaded("dup", person_name="Alice")
    mark_uploaded("dup", person_name="Alice")
    assert filter_already_uploaded(["dup", "new"]) == ["new"]


# ── frigate_files mapping ─────────────────────────────────────────────────────

def test_record_and_remove_frigate_file():
    from winnow.upload_tracker import get_person_summary, record_frigate_file, remove_frigate_file
    record_frigate_file("Alice", "Alice-1000.webp", "asset-a1")
    assert "Alice-1000.webp" in get_person_summary()["Alice"]["frigate_files"]
    remove_frigate_file("Alice", "Alice-1000.webp")
    assert "Alice-1000.webp" not in get_person_summary()["Alice"]["frigate_files"]


def test_remove_nonexistent_frigate_file_is_safe():
    from winnow.upload_tracker import remove_frigate_file
    # Should not raise even if the file was never recorded
    remove_frigate_file("Alice", "Alice-ghost.webp")


def test_remove_frigate_file_does_not_unmark_asset():
    """Deleting a Frigate file should not re-expose the source asset for upload."""
    from winnow.upload_tracker import (
        filter_already_uploaded,
        mark_uploaded,
        record_frigate_file,
        remove_frigate_file,
    )
    mark_uploaded("asset-a1", person_name="Alice")
    record_frigate_file("Alice", "Alice-1000.webp", "asset-a1")
    remove_frigate_file("Alice", "Alice-1000.webp")
    # Asset must still be excluded — it was deliberately replaced, not lost
    assert filter_already_uploaded(["asset-a1"]) == []


def test_get_tracked_frigate_file_count_zero_when_empty():
    from winnow.upload_tracker import get_tracked_frigate_file_count
    assert get_tracked_frigate_file_count("Alice") == 0


def test_get_tracked_frigate_file_count_counts_only_mapped():
    """Only files explicitly recorded via record_frigate_file count toward the cap."""
    from winnow.upload_tracker import get_tracked_frigate_file_count, mark_uploaded, record_frigate_file
    mark_uploaded("asset-a", person_name="Alice")
    mark_uploaded("asset-b", person_name="Alice")
    record_frigate_file("Alice", "Alice-1000.webp", "asset-a")
    # asset-b is uploaded but not yet mapped — does not count
    assert get_tracked_frigate_file_count("Alice") == 1
    record_frigate_file("Alice", "Alice-1001.webp", "asset-b")
    assert get_tracked_frigate_file_count("Alice") == 2


def test_get_lowest_quality_mapped_file_none_when_empty():
    from winnow.upload_tracker import get_lowest_quality_mapped_file
    assert get_lowest_quality_mapped_file("Alice") is None


def test_get_lowest_quality_mapped_file_returns_lowest():
    from winnow.upload_tracker import (
        get_lowest_quality_mapped_file,
        mark_uploaded,
        record_frigate_file,
    )
    mark_uploaded("asset-hi", person_name="Alice", score=0.95)
    mark_uploaded("asset-lo", person_name="Alice", score=0.71)
    record_frigate_file("Alice", "Alice-1000.webp", "asset-hi")
    record_frigate_file("Alice", "Alice-1001.webp", "asset-lo")
    result = get_lowest_quality_mapped_file("Alice")
    assert result is not None
    frigate_filename, asset_id, score = result
    assert frigate_filename == "Alice-1001.webp"
    assert asset_id == "asset-lo"
    assert score == pytest.approx(0.71, abs=0.001)


def test_get_lowest_quality_mapped_file_skips_unscored():
    """Files mapped without a score should not be returned as candidates."""
    from winnow.upload_tracker import (
        get_lowest_quality_mapped_file,
        mark_uploaded,
        record_frigate_file,
    )
    mark_uploaded("asset-scored", person_name="Alice", score=0.85)
    mark_uploaded("asset-noscr", person_name="Alice")
    record_frigate_file("Alice", "Alice-1000.webp", "asset-scored")
    record_frigate_file("Alice", "Alice-1001.webp", "asset-noscr")
    result = get_lowest_quality_mapped_file("Alice")
    assert result is not None
    assert result[1] == "asset-scored"  # only scored file is a candidate


# ── get_tracked_frigate_filenames ─────────────────────────────────────────────

def test_get_tracked_frigate_filenames_empty():
    from winnow.upload_tracker import get_tracked_frigate_filenames
    assert get_tracked_frigate_filenames("Alice") == set()


def test_get_tracked_frigate_filenames_returns_mapped():
    from winnow.upload_tracker import get_tracked_frigate_filenames, record_frigate_file
    record_frigate_file("Alice", "Alice-1000.webp", "asset-a")
    record_frigate_file("Alice", "Alice-1001.webp", "asset-b")
    assert get_tracked_frigate_filenames("Alice") == {"Alice-1000.webp", "Alice-1001.webp"}


def test_get_tracked_frigate_filenames_excludes_removed():
    from winnow.upload_tracker import (
        get_tracked_frigate_filenames,
        record_frigate_file,
        remove_frigate_file,
    )
    record_frigate_file("Alice", "Alice-1000.webp", "asset-a")
    record_frigate_file("Alice", "Alice-1001.webp", "asset-b")
    remove_frigate_file("Alice", "Alice-1000.webp")
    assert get_tracked_frigate_filenames("Alice") == {"Alice-1001.webp"}


def test_get_tracked_frigate_filenames_isolated_by_person():
    from winnow.upload_tracker import get_tracked_frigate_filenames, record_frigate_file
    record_frigate_file("Alice", "Alice-1000.webp", "asset-a")
    record_frigate_file("Bob", "Bob-2000.webp", "asset-b")
    assert get_tracked_frigate_filenames("Alice") == {"Alice-1000.webp"}
    assert get_tracked_frigate_filenames("Bob") == {"Bob-2000.webp"}


# ── get_lowest_quality_mapped_file with exclude ───────────────────────────────

def test_get_lowest_quality_exclude_skips_specified_file():
    from winnow.upload_tracker import (
        get_lowest_quality_mapped_file,
        mark_uploaded,
        record_frigate_file,
    )
    mark_uploaded("asset-lo", person_name="Alice", score=0.10)
    mark_uploaded("asset-hi", person_name="Alice", score=0.90)
    record_frigate_file("Alice", "Alice-lo.webp", "asset-lo")
    record_frigate_file("Alice", "Alice-hi.webp", "asset-hi")
    result = get_lowest_quality_mapped_file("Alice", exclude={"Alice-lo.webp"})
    assert result is not None
    assert result[1] == "asset-hi"  # lo was excluded; hi is returned


def test_get_lowest_quality_exclude_all_returns_none():
    from winnow.upload_tracker import (
        get_lowest_quality_mapped_file,
        mark_uploaded,
        record_frigate_file,
    )
    mark_uploaded("asset-a", person_name="Alice", score=0.50)
    record_frigate_file("Alice", "Alice-a.webp", "asset-a")
    assert get_lowest_quality_mapped_file("Alice", exclude={"Alice-a.webp"}) is None


# ── get_most_redundant_mapped_file ────────────────────────────────────────────

def test_get_most_redundant_none_when_no_frigate_scores():
    from winnow.upload_tracker import get_most_redundant_mapped_file, mark_uploaded, record_frigate_file
    mark_uploaded("asset-a", person_name="Alice", score=0.80)
    record_frigate_file("Alice", "Alice-a.webp", "asset-a")
    # blur score only, no frigate_score → no candidates
    assert get_most_redundant_mapped_file("Alice") is None


def test_get_most_redundant_returns_highest_frigate_score():
    from winnow.upload_tracker import get_most_redundant_mapped_file, mark_uploaded, record_frigate_file
    mark_uploaded("asset-novel", person_name="Alice", score=0.50, frigate_score=0.31)
    mark_uploaded("asset-redundant", person_name="Alice", score=0.90, frigate_score=0.88)
    record_frigate_file("Alice", "Alice-novel.webp", "asset-novel")
    record_frigate_file("Alice", "Alice-redundant.webp", "asset-redundant")
    result = get_most_redundant_mapped_file("Alice")
    assert result is not None
    frigate_filename, asset_id, score = result
    assert frigate_filename == "Alice-redundant.webp"
    assert asset_id == "asset-redundant"
    assert score == pytest.approx(0.88, abs=0.001)


def test_get_most_redundant_exclude_skips_file():
    from winnow.upload_tracker import get_most_redundant_mapped_file, mark_uploaded, record_frigate_file
    mark_uploaded("asset-hi", person_name="Alice", score=0.9, frigate_score=0.85)
    mark_uploaded("asset-lo", person_name="Alice", score=0.5, frigate_score=0.40)
    record_frigate_file("Alice", "Alice-hi.webp", "asset-hi")
    record_frigate_file("Alice", "Alice-lo.webp", "asset-lo")
    result = get_most_redundant_mapped_file("Alice", exclude={"Alice-hi.webp"})
    assert result is not None
    assert result[1] == "asset-lo"  # hi excluded; lo is next highest


def test_get_most_redundant_exclude_all_returns_none():
    from winnow.upload_tracker import get_most_redundant_mapped_file, mark_uploaded, record_frigate_file
    mark_uploaded("asset-a", person_name="Alice", score=0.5, frigate_score=0.70)
    record_frigate_file("Alice", "Alice-a.webp", "asset-a")
    assert get_most_redundant_mapped_file("Alice", exclude={"Alice-a.webp"}) is None


# ── cross-process locking ──────────────────────────────────────────────────────

def test_tracker_lock_blocks_concurrent_exclusive_access():
    """While upload_tracker holds the tracker lock, a second (non-blocking) attempt
    to exclusively lock the same file must fail — proving the lock is real and
    guards a second winnow process from racing a read-modify-write."""
    from winnow import upload_tracker as ut

    ut._acquire_lock()
    try:
        lock_path = ut._lock_path()
        assert lock_path.exists()
        fd = os.open(lock_path, os.O_RDWR)
        try:
            with pytest.raises(OSError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
    finally:
        ut._release_lock()

    # Released — a second exclusive, non-blocking lock now succeeds immediately.
    fd = os.open(ut._lock_path(), os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def test_tracker_lock_reentrant_within_process():
    """Nested acquisitions (e.g. begin_batch() for both tracker files, or a tracker
    call made while a batch is open) must not deadlock the process that holds them."""
    from winnow import upload_tracker as ut

    assert ut._lock_depth == 0
    ut._acquire_lock()
    ut._acquire_lock()
    assert ut._lock_depth == 2
    ut._release_lock()
    assert ut._lock_depth == 1
    ut._release_lock()
    assert ut._lock_depth == 0


def test_begin_flush_batch_releases_lock_for_next_caller():
    """begin_batch()/flush_batch() (including a nested mark_uploaded call inside the
    batch) must fully release the lock so it doesn't stay held for the rest of the run."""
    from winnow import upload_tracker as ut
    from winnow.upload_tracker import (
        REJECT_TRACKER_FILE,
        UPLOAD_TRACKER_FILE,
        begin_batch,
        flush_batch,
        mark_uploaded,
    )

    begin_batch(UPLOAD_TRACKER_FILE)
    begin_batch(REJECT_TRACKER_FILE)
    mark_uploaded("a1", person_name="Alice")
    flush_batch(UPLOAD_TRACKER_FILE)
    flush_batch(REJECT_TRACKER_FILE)

    assert ut._lock_depth == 0
    fd = os.open(ut._lock_path(), os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


# ── batched writes: incremental flush bounds crash loss ───────────────────────

def test_batch_defers_writes_below_flush_threshold(isolated_cache):
    """Below the flush threshold, writes stay in memory — batching still avoids
    a disk write per mark."""
    from winnow.upload_tracker import UPLOAD_TRACKER_FILE, begin_batch, flush_batch, mark_uploaded

    tracker_path = isolated_cache / UPLOAD_TRACKER_FILE
    begin_batch(UPLOAD_TRACKER_FILE)
    try:
        mark_uploaded("asset-0", person_name="Alice")
        assert not tracker_path.exists()
    finally:
        flush_batch(UPLOAD_TRACKER_FILE)
    assert tracker_path.exists()


def test_batch_flushes_incrementally_bounding_crash_loss(isolated_cache):
    """A crash mid-batch (SIGKILL/OOM/host crash) must lose at most a bounded number
    of marks, not the whole per-person batch — verified by reading the file straight
    off disk before flush_batch() is ever called."""
    from winnow.upload_tracker import _BATCH_FLUSH_EVERY, UPLOAD_TRACKER_FILE, begin_batch, flush_batch, mark_uploaded

    tracker_path = isolated_cache / UPLOAD_TRACKER_FILE
    begin_batch(UPLOAD_TRACKER_FILE)
    try:
        for i in range(_BATCH_FLUSH_EVERY):
            mark_uploaded(f"asset-{i}", person_name="Alice")
        # Threshold reached: disk already reflects all marks so far, even though
        # flush_batch() has not run yet — simulates surviving a crash here.
        on_disk = json.loads(tracker_path.read_text())
        ids = on_disk["by_person"]["Alice"]["asset_ids"]
        assert len(ids) == _BATCH_FLUSH_EVERY

        # One more mark past the threshold stays deferred again until the next
        # periodic flush or flush_batch().
        mark_uploaded("asset-extra", person_name="Alice")
        on_disk = json.loads(tracker_path.read_text())
        assert len(on_disk["by_person"]["Alice"]["asset_ids"]) == _BATCH_FLUSH_EVERY
    finally:
        flush_batch(UPLOAD_TRACKER_FILE)

    on_disk = json.loads(tracker_path.read_text())
    assert len(on_disk["by_person"]["Alice"]["asset_ids"]) == _BATCH_FLUSH_EVERY + 1
