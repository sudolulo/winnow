"""Smoke tests for configuration loading."""




def test_config_loads_defaults(monkeypatch):
    """Config reads env vars and falls back to documented defaults."""
    monkeypatch.setenv("IMMICH_URL", "http://test:2283")
    monkeypatch.setenv("API_KEY", "test-key")

    from winnow.config import _Config

    _Config.reset()
    cfg = _Config()

    assert cfg.IMMICH_URL == "http://test:2283"
    assert cfg.API_KEY == "test-key"
    assert cfg.OUTPUT_DIR == "./frigate_train"
    assert cfg.YEARS_FILTER == 10
    assert cfg.MIN_FACE_WIDTH == 50
    assert cfg.MIN_FACE_COUNT == 0
    assert cfg.BLUR_THRESHOLD == 100.0
    assert cfg.MIN_CONFIDENCE == 0.7
    assert cfg.MAX_AUTO_IMAGES == 80
    assert cfg.FACE_MARGIN == 0.15
    assert cfg.USE_FULL_RESOLUTION is True
    assert cfg.ENABLE_FACE_ALIGNMENT is True
    assert cfg.ENABLE_CACHE is True

    _Config.reset()


def test_config_env_overrides(monkeypatch):
    """All quality settings are overridable via environment variables."""
    monkeypatch.setenv("IMMICH_URL", "http://test:2283")
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("YEARS_FILTER", "5")
    monkeypatch.setenv("MIN_FACE_WIDTH", "80")
    monkeypatch.setenv("BLUR_THRESHOLD", "50.0")
    monkeypatch.setenv("MIN_CONFIDENCE", "0.9")
    monkeypatch.setenv("MAX_AUTO_IMAGES", "40")
    monkeypatch.setenv("FACE_MARGIN", "0.2")
    monkeypatch.setenv("USE_FULL_RESOLUTION", "false")
    monkeypatch.setenv("ENABLE_FACE_ALIGNMENT", "false")
    monkeypatch.setenv("ENABLE_CACHE", "true")

    from winnow.config import _Config

    _Config.reset()
    cfg = _Config()

    assert cfg.YEARS_FILTER == 5
    assert cfg.MIN_FACE_WIDTH == 80
    assert cfg.BLUR_THRESHOLD == 50.0
    assert cfg.MIN_CONFIDENCE == 0.9
    assert cfg.MAX_AUTO_IMAGES == 40
    assert cfg.FACE_MARGIN == 0.2
    assert cfg.USE_FULL_RESOLUTION is False
    assert cfg.ENABLE_FACE_ALIGNMENT is False
    assert cfg.ENABLE_CACHE is True

    _Config.reset()


def test_get_headers_returns_api_key(monkeypatch):
    """get_headers() returns the correct auth header dict."""
    monkeypatch.setenv("IMMICH_URL", "http://test:2283")
    monkeypatch.setenv("API_KEY", "my-secret-key")

    from winnow.config import _Config, get_headers

    _Config.reset()
    headers = get_headers()

    assert headers["x-api-key"] == "my-secret-key"
    assert headers["Accept"] == "application/json"

    _Config.reset()
