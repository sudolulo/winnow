"""Tests for upload tracker — mark, filter, reset, and summary logic."""

import json
import os
import tempfile

import pytest


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    """Point tracker at a temp directory so tests don't touch real cache files."""
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    from if_curator.config import _Config
    _Config.reset()
    yield tmp_path
    _Config.reset()


def test_filter_returns_all_when_empty():
    from if_curator.upload_tracker import filter_already_uploaded
    ids = ["a1", "b2", "c3"]
    assert filter_already_uploaded(ids) == ids


def test_mark_uploaded_excludes_from_filter():
    from if_curator.upload_tracker import filter_already_uploaded, mark_uploaded
    mark_uploaded("a1", person_name="Alice")
    result = filter_already_uploaded(["a1", "b2"])
    assert result == ["b2"]


def test_mark_rejected_excludes_from_filter():
    from if_curator.upload_tracker import filter_already_uploaded, mark_rejected
    mark_rejected("x9", person_name="Bob")
    result = filter_already_uploaded(["x9", "y8"])
    assert result == ["y8"]


def test_retry_rejected_includes_rejected():
    from if_curator.upload_tracker import filter_already_uploaded, mark_rejected
    mark_rejected("x9", person_name="Bob")
    result = filter_already_uploaded(["x9", "y8"], retry_rejected=True)
    assert "x9" in result


def test_reset_person_clears_records():
    from if_curator.upload_tracker import filter_already_uploaded, mark_uploaded, reset_person
    mark_uploaded("a1", person_name="Alice")
    mark_uploaded("a2", person_name="Alice")
    mark_uploaded("b1", person_name="Bob")
    reset_person("Alice")
    assert filter_already_uploaded(["a1", "a2"]) == ["a1", "a2"]
    assert filter_already_uploaded(["b1"]) == []


def test_get_person_summary():
    from if_curator.upload_tracker import get_person_summary, mark_rejected, mark_uploaded
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
    from if_curator.upload_tracker import filter_already_uploaded, mark_uploaded
    mark_uploaded("dup", person_name="Alice")
    mark_uploaded("dup", person_name="Alice")
    assert filter_already_uploaded(["dup", "new"]) == ["new"]
