"""Tests for upload tracker — mark, filter, reset, and summary logic."""


import pytest


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    """Point tracker at a temp directory so tests don't touch real cache files."""
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
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
