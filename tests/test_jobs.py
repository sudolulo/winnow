"""Tests for job strategy resolution and env var handling."""

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure LIMIT and STRATEGY are unset before each test."""
    monkeypatch.delenv("LIMIT", raising=False)
    monkeypatch.delenv("STRATEGY", raising=False)


def test_resolve_strategy_default_auto():
    from if_curator.jobs import _resolve_strategy
    limit, mode = _resolve_strategy("auto", has_embedding=True)
    assert limit == "auto"
    assert mode == "smart"


def test_resolve_strategy_standard():
    from if_curator.jobs import _resolve_strategy
    limit, mode = _resolve_strategy("standard", has_embedding=True)
    assert limit == 30
    assert mode == "smart"


def test_resolve_strategy_broad():
    from if_curator.jobs import _resolve_strategy
    limit, mode = _resolve_strategy("broad", has_embedding=True)
    assert limit == 100
    assert mode == "smart"


def test_resolve_strategy_custom_limit_env(monkeypatch):
    monkeypatch.setenv("LIMIT", "50")
    from if_curator.jobs import _resolve_strategy
    limit, mode = _resolve_strategy("auto", has_embedding=True)
    assert limit == 50
    assert mode == "smart"


def test_resolve_strategy_limit_overrides_strategy(monkeypatch):
    monkeypatch.setenv("LIMIT", "25")
    from if_curator.jobs import _resolve_strategy
    limit, mode = _resolve_strategy("broad", has_embedding=True)
    assert limit == 25


def test_resolve_strategy_no_embedding_falls_back_to_time():
    from if_curator.jobs import _resolve_strategy
    limit, mode = _resolve_strategy("auto", has_embedding=False)
    assert mode == "time"
    assert isinstance(limit, int)


def test_resolve_strategy_no_embedding_respects_limit(monkeypatch):
    monkeypatch.setenv("LIMIT", "60")
    from if_curator.jobs import _resolve_strategy
    limit, mode = _resolve_strategy("auto", has_embedding=False)
    assert limit == 60
    assert mode == "time"


def test_resolve_strategy_unknown_falls_back_to_auto():
    from if_curator.jobs import _resolve_strategy
    limit, mode = _resolve_strategy("unknown-strategy", has_embedding=True)
    assert limit == "auto"
    assert mode == "smart"
