"""Configuration management for winnow."""

import json
import logging
import os
from pathlib import Path
from typing import ClassVar

from dotenv import load_dotenv
from rich.prompt import Prompt

_LEGACY_CONFIG_FILE = Path(".immich_config.json")  # pre-v0.6: lived in process CWD, not on a volume


class _Config:
    """Singleton configuration with lazy loading via __getattr__.

    Class-level attributes are annotations only (no defaults), so attribute
    access on an un-loaded instance falls through to __getattr__, which
    triggers _load() exactly once.
    """

    _instance: ClassVar["_Config | None"] = None

    # Annotations only — no class-level defaults so __getattr__ fires on first access
    IMMICH_URL: str | None
    API_KEY: str | None
    OUTPUT_DIR: str
    YEARS_FILTER: int

    # Quality filtering
    MIN_FACE_WIDTH: int
    BLUR_THRESHOLD: float
    MIN_CONFIDENCE: float
    MAX_AUTO_IMAGES: int
    QUALITY_REPLACEMENT: bool
    FRIGATE_SCORE_CEILING: float | None
    ENABLE_FRIGATE_SCORES: bool

    # People filtering
    MIN_FACE_COUNT: int
    MERGE_DUPLICATE_PEOPLE: bool

    # Output quality
    FACE_MARGIN: float
    USE_FULL_RESOLUTION: bool
    ENABLE_FACE_ALIGNMENT: bool

    ENABLE_CACHE: bool
    DATA_DIR: str

    def __new__(cls) -> "_Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Do NOT call _load() here — keep __new__ I/O-free so that import
            # time does not trigger env/file reads.
        return cls._instance

    def __getattr__(self, name: str):
        """Called only when the attribute is not found on the instance.

        On first access to any config attribute, load all values from env/file
        and return the requested one.  Re-registers self as _instance so that
        a subsequent reset() correctly finds and clears this object's attrs.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        self._load()
        # Re-register self as the singleton so reset() can clear our __dict__.
        # This handles the case where __getattr__ is called on the module-level
        # Config object after a reset() set _instance to None.
        _Config._instance = self
        # _load() sets the attribute as an instance attr; retrieve it directly
        # to avoid infinite recursion through __getattr__.
        try:
            return self.__dict__[name]
        except KeyError:
            raise AttributeError(f"_Config has no attribute {name!r}")

    def _load(self) -> None:
        """Load configuration from environment and config file."""
        load_dotenv()
        # Load from environment (highest priority)
        self.IMMICH_URL = os.getenv("IMMICH_URL")
        self.API_KEY = os.getenv("API_KEY")
        self.OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./frigate_train")
        self.YEARS_FILTER = int(os.getenv("YEARS_FILTER", "10"))
        self.MIN_FACE_WIDTH = int(os.getenv("MIN_FACE_WIDTH", "90"))
        self.MIN_FACE_COUNT = int(os.getenv("MIN_FACE_COUNT", "3"))
        self.MERGE_DUPLICATE_PEOPLE = os.getenv("MERGE_DUPLICATE_PEOPLE", "false").lower() in ("true", "1", "yes")
        self.BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", "120.0"))
        self.MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.7"))
        self.MAX_AUTO_IMAGES = int(os.getenv("MAX_AUTO_IMAGES", "20"))
        self.QUALITY_REPLACEMENT = os.getenv("QUALITY_REPLACEMENT", "true").lower() in ("true", "1", "yes")
        _ceiling_env = os.getenv("FRIGATE_SCORE_CEILING", "").strip()
        if _ceiling_env:
            try:
                self.FRIGATE_SCORE_CEILING = float(_ceiling_env)
            except ValueError:
                logging.warning("FRIGATE_SCORE_CEILING=%r is not a valid float — ignoring", _ceiling_env)
                self.FRIGATE_SCORE_CEILING = None
        else:
            self.FRIGATE_SCORE_CEILING = None
        self.ENABLE_FRIGATE_SCORES = os.getenv("ENABLE_FRIGATE_SCORES", "true").lower() in ("true", "1", "yes")
        self.FACE_MARGIN = float(os.getenv("FACE_MARGIN", "0.15"))
        self.USE_FULL_RESOLUTION = os.getenv("USE_FULL_RESOLUTION", "true").lower() in ("true", "1", "yes")
        self.ENABLE_FACE_ALIGNMENT = os.getenv("ENABLE_FACE_ALIGNMENT", "true").lower() in ("true", "1", "yes")
        self.ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() in ("true", "1", "yes")
        _data_dir = os.getenv("DATA_DIR")
        _cache_dir_legacy = os.getenv("CACHE_DIR")
        if _data_dir:
            self.DATA_DIR = _data_dir
        elif _cache_dir_legacy:
            logging.warning(
                "CACHE_DIR is deprecated — rename it to DATA_DIR in your .env or compose.yml"
            )
            self.DATA_DIR = _cache_dir_legacy
        else:
            self.DATA_DIR = "data"

        # Fall back to config file only when the env var is genuinely absent (None).
        # An explicitly empty env var (IMMICH_URL="") takes priority over the file.
        # Prefer DATA_DIR/.immich_config.json (volume-safe in Docker) and fall back
        # to the legacy CWD path so existing installations continue to work.
        _data_cfg = Path(self.DATA_DIR) / ".immich_config.json"
        _data_cfg_exists = _data_cfg.exists()
        if _data_cfg_exists and _LEGACY_CONFIG_FILE.exists():
            logging.warning(
                "Two config files found: %s and %s — using %s. Remove the legacy file to silence this.",
                _data_cfg,
                _LEGACY_CONFIG_FILE,
                _data_cfg,
            )
        config_file = _data_cfg if _data_cfg_exists else _LEGACY_CONFIG_FILE
        # _data_cfg_exists already confirmed the primary path — avoid re-stat.
        # The short-circuit means the legacy path is stat'd at most once here.
        if _data_cfg_exists or config_file.exists():
            try:
                data = json.loads(config_file.read_text())
                if self.IMMICH_URL is None:
                    self.IMMICH_URL = data.get("IMMICH_URL")
                if os.getenv("OUTPUT_DIR") is None:
                    self.OUTPUT_DIR = data.get("OUTPUT_DIR", self.OUTPUT_DIR)
            except (json.JSONDecodeError, OSError) as e:
                logging.warning("Failed to load config file: %s", e)

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton — mainly useful for testing or delayed env setup."""
        if cls._instance is not None:
            cls._instance.__dict__.clear()
        cls._instance = None

    def save(self) -> None:
        """Persist non-sensitive configuration to file.

        API_KEY is intentionally excluded — store it in .env or as an
        environment variable instead of a plain-text config file.
        Writes to DATA_DIR/.immich_config.json so the file survives container
        restarts when DATA_DIR is a mounted volume.
        """
        config_file = Path(self.DATA_DIR) / ".immich_config.json"
        try:
            Path(self.DATA_DIR).mkdir(parents=True, exist_ok=True)
            config_file.write_text(
                json.dumps(
                    {
                        "IMMICH_URL": self.IMMICH_URL,
                        "OUTPUT_DIR": self.OUTPUT_DIR,
                    },
                    indent=2,
                )
            )
            logging.info("Configuration saved to %s", config_file)
        except OSError as e:
            logging.error("Failed to save config: %s", e)

    def interactive_setup(self) -> None:
        """Prompt user for missing configuration."""
        from rich.console import Console

        console = Console()

        if not self.IMMICH_URL:
            console.print("[yellow]Immich URL not found.[/yellow]")
            self.IMMICH_URL = Prompt.ask("Enter Immich URL (e.g. http://192.168.1.5:2283)")
            self.save()

        if not self.API_KEY:
            console.print("[yellow]Immich API Key not found.[/yellow]")
            console.print("[dim]Tip: set API_KEY in your .env file to avoid re-entering it.[/dim]")
            self.API_KEY = Prompt.ask("Enter Immich API Key", password=True)

    def validate(self) -> None:
        """Raise ValueError if required config is missing."""
        if not self.IMMICH_URL or not self.API_KEY:
            raise ValueError("Missing Immich URL or API Key.")


# Module-level singleton — lazy: no I/O until first attribute access.
Config = _Config()


def get_headers() -> dict[str, str]:
    """Return HTTP headers for Immich API requests."""
    return {"x-api-key": Config.API_KEY or "", "Accept": "application/json"}
