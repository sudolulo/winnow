"""Tests for upload tracker — mark, filter, reset, and summary logic."""


import pytest


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    """Point tracker at a temp directory so tests don't touch real cache files."""
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    from winnow.config import _Config
    _Config.reset()
    # Also reset the SQLite connection so the next call opens the new path
    import winnow.upload_tracker as ut
    ut._conn = None
    ut._conn_path = None
    yield tmp_path
    # Teardown
    if ut._conn is not None:
        try:
            ut._conn.close()
        except Exception:
            pass
    ut._conn = None
    ut._conn_path = None
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

def test_record_and_remove_frigate_files_batch():
    from winnow.upload_tracker import get_person_summary, record_frigate_files_batch, remove_frigate_file
    record_frigate_files_batch("Alice", {"Alice-1000.webp": "asset-a1"})
    assert "Alice-1000.webp" in get_person_summary()["Alice"]["frigate_files"]
    remove_frigate_file("Alice", "Alice-1000.webp")
    assert "Alice-1000.webp" not in get_person_summary().get("Alice", {}).get("frigate_files", {})


def test_remove_nonexistent_frigate_file_is_safe():
    from winnow.upload_tracker import remove_frigate_file
    # Should not raise even if the file was never recorded
    remove_frigate_file("Alice", "Alice-ghost.webp")


def test_remove_frigate_file_does_not_unmark_asset():
    """Deleting a Frigate file should not re-expose the source asset for upload."""
    from winnow.upload_tracker import (
        filter_already_uploaded,
        mark_uploaded,
        record_frigate_files_batch,
        remove_frigate_file,
    )
    mark_uploaded("asset-a1", person_name="Alice")
    record_frigate_files_batch("Alice", {"Alice-1000.webp": "asset-a1"})
    remove_frigate_file("Alice", "Alice-1000.webp")
    # Asset must still be excluded — it was deliberately replaced, not lost
    assert filter_already_uploaded(["asset-a1"]) == []


def test_get_tracked_frigate_file_count_zero_when_empty():
    from winnow.upload_tracker import get_tracked_frigate_file_count
    assert get_tracked_frigate_file_count("Alice") == 0


def test_get_tracked_frigate_file_count_counts_only_mapped():
    """Only files explicitly recorded via record_frigate_files_batch count toward the cap."""
    from winnow.upload_tracker import get_tracked_frigate_file_count, mark_uploaded, record_frigate_files_batch
    mark_uploaded("asset-a", person_name="Alice")
    mark_uploaded("asset-b", person_name="Alice")
    record_frigate_files_batch("Alice", {"Alice-1000.webp": "asset-a"})
    # asset-b is uploaded but not yet mapped — does not count
    assert get_tracked_frigate_file_count("Alice") == 1
    record_frigate_files_batch("Alice", {"Alice-1001.webp": "asset-b"})
    assert get_tracked_frigate_file_count("Alice") == 2


def test_get_lowest_quality_mapped_file_none_when_empty():
    from winnow.upload_tracker import get_lowest_quality_mapped_file
    assert get_lowest_quality_mapped_file("Alice") is None


def test_get_lowest_quality_mapped_file_returns_lowest():
    from winnow.upload_tracker import (
        get_lowest_quality_mapped_file,
        mark_uploaded,
        record_frigate_files_batch,
    )
    mark_uploaded("asset-hi", person_name="Alice", score=0.95)
    mark_uploaded("asset-lo", person_name="Alice", score=0.71)
    record_frigate_files_batch("Alice", {"Alice-1000.webp": "asset-hi", "Alice-1001.webp": "asset-lo"})
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
        record_frigate_files_batch,
    )
    mark_uploaded("asset-scored", person_name="Alice", score=0.85)
    mark_uploaded("asset-noscr", person_name="Alice")
    record_frigate_files_batch("Alice", {"Alice-1000.webp": "asset-scored", "Alice-1001.webp": "asset-noscr"})
    result = get_lowest_quality_mapped_file("Alice")
    assert result is not None
    assert result[1] == "asset-scored"  # only scored file is a candidate


# ── get_tracked_frigate_filenames ─────────────────────────────────────────────

def test_get_tracked_frigate_filenames_empty():
    from winnow.upload_tracker import get_tracked_frigate_filenames
    assert get_tracked_frigate_filenames("Alice") == set()


def test_get_tracked_frigate_filenames_returns_mapped():
    from winnow.upload_tracker import get_tracked_frigate_filenames, record_frigate_files_batch
    record_frigate_files_batch("Alice", {"Alice-1000.webp": "asset-a", "Alice-1001.webp": "asset-b"})
    assert get_tracked_frigate_filenames("Alice") == {"Alice-1000.webp", "Alice-1001.webp"}


def test_get_tracked_frigate_filenames_excludes_removed():
    from winnow.upload_tracker import (
        get_tracked_frigate_filenames,
        record_frigate_files_batch,
        remove_frigate_file,
    )
    record_frigate_files_batch("Alice", {"Alice-1000.webp": "asset-a", "Alice-1001.webp": "asset-b"})
    remove_frigate_file("Alice", "Alice-1000.webp")
    assert get_tracked_frigate_filenames("Alice") == {"Alice-1001.webp"}


def test_get_tracked_frigate_filenames_isolated_by_person():
    from winnow.upload_tracker import get_tracked_frigate_filenames, record_frigate_files_batch
    record_frigate_files_batch("Alice", {"Alice-1000.webp": "asset-a"})
    record_frigate_files_batch("Bob", {"Bob-2000.webp": "asset-b"})
    assert get_tracked_frigate_filenames("Alice") == {"Alice-1000.webp"}
    assert get_tracked_frigate_filenames("Bob") == {"Bob-2000.webp"}


# ── get_lowest_quality_mapped_file with exclude ───────────────────────────────

def test_get_lowest_quality_exclude_skips_specified_file():
    from winnow.upload_tracker import (
        get_lowest_quality_mapped_file,
        mark_uploaded,
        record_frigate_files_batch,
    )
    mark_uploaded("asset-lo", person_name="Alice", score=0.10)
    mark_uploaded("asset-hi", person_name="Alice", score=0.90)
    record_frigate_files_batch("Alice", {"Alice-lo.webp": "asset-lo", "Alice-hi.webp": "asset-hi"})
    result = get_lowest_quality_mapped_file("Alice", exclude={"Alice-lo.webp"})
    assert result is not None
    assert result[1] == "asset-hi"  # lo was excluded; hi is returned


def test_get_lowest_quality_exclude_all_returns_none():
    from winnow.upload_tracker import (
        get_lowest_quality_mapped_file,
        mark_uploaded,
        record_frigate_files_batch,
    )
    mark_uploaded("asset-a", person_name="Alice", score=0.50)
    record_frigate_files_batch("Alice", {"Alice-a.webp": "asset-a"})
    assert get_lowest_quality_mapped_file("Alice", exclude={"Alice-a.webp"}) is None


# ── get_most_redundant_mapped_file ────────────────────────────────────────────

def test_get_most_redundant_none_when_no_frigate_scores():
    from winnow.upload_tracker import get_most_redundant_mapped_file, mark_uploaded, record_frigate_files_batch
    mark_uploaded("asset-a", person_name="Alice", score=0.80)
    record_frigate_files_batch("Alice", {"Alice-a.webp": "asset-a"})
    # blur score only, no frigate_score → no candidates
    assert get_most_redundant_mapped_file("Alice") is None


def test_get_most_redundant_returns_highest_frigate_score():
    from winnow.upload_tracker import get_most_redundant_mapped_file, mark_uploaded, record_frigate_files_batch
    mark_uploaded("asset-novel", person_name="Alice", score=0.50, frigate_score=0.31)
    mark_uploaded("asset-redundant", person_name="Alice", score=0.90, frigate_score=0.88)
    record_frigate_files_batch("Alice", {"Alice-novel.webp": "asset-novel", "Alice-redundant.webp": "asset-redundant"})
    result = get_most_redundant_mapped_file("Alice")
    assert result is not None
    frigate_filename, asset_id, score = result
    assert frigate_filename == "Alice-redundant.webp"
    assert asset_id == "asset-redundant"
    assert score == pytest.approx(0.88, abs=0.001)


def test_get_most_redundant_exclude_skips_file():
    from winnow.upload_tracker import get_most_redundant_mapped_file, mark_uploaded, record_frigate_files_batch
    mark_uploaded("asset-hi", person_name="Alice", score=0.9, frigate_score=0.85)
    mark_uploaded("asset-lo", person_name="Alice", score=0.5, frigate_score=0.40)
    record_frigate_files_batch("Alice", {"Alice-hi.webp": "asset-hi", "Alice-lo.webp": "asset-lo"})
    result = get_most_redundant_mapped_file("Alice", exclude={"Alice-hi.webp"})
    assert result is not None
    assert result[1] == "asset-lo"  # hi excluded; lo is next highest


def test_get_most_redundant_exclude_all_returns_none():
    from winnow.upload_tracker import get_most_redundant_mapped_file, mark_uploaded, record_frigate_files_batch
    mark_uploaded("asset-a", person_name="Alice", score=0.5, frigate_score=0.70)
    record_frigate_files_batch("Alice", {"Alice-a.webp": "asset-a"})
    assert get_most_redundant_mapped_file("Alice", exclude={"Alice-a.webp"}) is None
