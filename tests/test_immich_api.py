"""Tests for Immich API utility functions (no network required)."""

from datetime import datetime, timedelta, timezone


def _asset(days_ago: int) -> dict:
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    return {"id": f"id-{days_ago}d", "fileCreatedAt": ts}


def test_filter_recent_keeps_new_assets(monkeypatch):
    from if_curator.immich_api import filter_recent_assets
    assets = [_asset(1), _asset(30), _asset(365 * 5)]
    result = filter_recent_assets(assets, years=10)
    assert len(result) == 3


def test_filter_recent_removes_old_assets(monkeypatch):
    from if_curator.immich_api import filter_recent_assets
    old = _asset(365 * 15)
    recent = _asset(10)
    result = filter_recent_assets([old, recent], years=10)
    assert len(result) == 1
    assert result[0]["id"] == "id-10d"


def test_filter_recent_boundary(monkeypatch):
    from if_curator.immich_api import filter_recent_assets
    just_inside = _asset(365 * 10 - 1)
    just_outside = _asset(365 * 10 + 1)
    result = filter_recent_assets([just_inside, just_outside], years=10)
    assert len(result) == 1
    assert result[0] == just_inside


def test_filter_recent_skips_missing_date():
    from if_curator.immich_api import filter_recent_assets
    assets = [{"id": "no-date"}, _asset(5)]
    result = filter_recent_assets(assets, years=10)
    assert len(result) == 1
    assert result[0]["id"] == "id-5d"


def test_filter_recent_skips_bad_date():
    from if_curator.immich_api import filter_recent_assets
    assets = [{"id": "bad", "fileCreatedAt": "not-a-date"}, _asset(5)]
    result = filter_recent_assets(assets, years=10)
    assert len(result) == 1


def test_filter_recent_empty_list():
    from if_curator.immich_api import filter_recent_assets
    assert filter_recent_assets([], years=10) == []
