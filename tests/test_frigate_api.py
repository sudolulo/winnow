"""Tests for Frigate API helpers (no network required)."""

import pytest


class _FakeResponse:
    def __init__(self, json_body, status_code=200):
        self._json_body = json_body
        self.status_code = status_code
        self.ok = 200 <= status_code < 300

    def raise_for_status(self):
        if not self.ok:
            raise Exception(f"HTTP {self.status_code}")

    def json(self):
        return self._json_body


@pytest.fixture(autouse=True)
def frigate_url(monkeypatch):
    monkeypatch.setenv("FRIGATE_URL", "http://frigate.test")
    yield


def test_get_all_frigate_person_files_happy_path(monkeypatch):
    from winnow import frigate_api
    body = {"Alice": ["a-1.webp", "a-2.webp"], "train": ["pending.webp"]}
    monkeypatch.setattr(frigate_api.requests, "get", lambda *a, **k: _FakeResponse(body))
    result = frigate_api.get_all_frigate_person_files()
    assert result == {"Alice": ["a-1.webp", "a-2.webp"]}


def test_get_all_frigate_person_files_rejects_non_dict_body(monkeypatch):
    """A non-dict /api/faces response must not crash data.items() — it should degrade to None."""
    from winnow import frigate_api
    monkeypatch.setattr(frigate_api.requests, "get", lambda *a, **k: _FakeResponse(["not", "a", "dict"]))
    assert frigate_api.get_all_frigate_person_files() is None


def test_get_frigate_person_files_rejects_non_dict_body(monkeypatch):
    from winnow import frigate_api
    monkeypatch.setattr(frigate_api.requests, "get", lambda *a, **k: _FakeResponse(["not", "a", "dict"]))
    assert frigate_api.get_frigate_person_files("Alice") is None


def test_get_frigate_face_counts_rejects_non_dict_body(monkeypatch):
    from winnow import frigate_api
    monkeypatch.setattr(frigate_api.requests, "get", lambda *a, **k: _FakeResponse("oops"))
    assert frigate_api.get_frigate_face_counts() is None


def test_get_all_frigate_person_files_rejects_scalar_body(monkeypatch):
    from winnow import frigate_api
    monkeypatch.setattr(frigate_api.requests, "get", lambda *a, **k: _FakeResponse(42))
    assert frigate_api.get_all_frigate_person_files() is None
