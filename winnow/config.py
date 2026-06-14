"""Configuration management for winnow."""

import json
import logging
import os
from pathlib import Path
from typing import ClassVar

from dotenv import load_dotenv
from rich.prompt import Prompt

load_dotenv()

CONFIG_FILE = Path(".immich_config.json")


class _Config:
    """Singleton configuration with uppercase attribute access for backward compatibility."""

    _instance: ClassVar["_Config | None"] = None

    # Configuration values
    IMMICH_URL: str | None = None
    API_KEY: str | None = None
    OUTPUT_DIR: str = "./frigate_train"
    YEARS_FILTER: int = 10

    # Quality filtering
    MIN_FACE_WIDTH: int = 90
    BLUR_THRESHOLD: float = 120.0
    MIN_CONFIDENCE: float = 0.7
    MAX_AUTO_IMAGES: int = 80
    QUALITY_REPLACEMENT: bool = True
    FRIGATE_SCORE_CEILING: float = 0.0
    ENABLE_FRIGATE_SCORES: bool = True

    # People filtering
    MIN_FACE_COUNT: int = 3
    MERGE_DUPLICATE_PEOPLE: bool = False

    # Output quality
    FACE_MARGIN: float = 0.15
    USE_FULL_RESOLUTION: bool = True
    ENABLE_FACE_ALIGNMENT: bool = True

    ENABLE_CACHE: bool = True
    CACHE_DIR: str = ".if_cache"

    def __new__(cls) -> "_Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        """Load configuration from environment and config file."""
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
        self.MAX_AUTO_IMAGES = int(os.getenv("MAX_AUTO_IMAGES", "80"))
        self.QUALITY_REPLACEMENT = os.getenv("QUALITY_REPLACEMENT", "true").lower() in ("true", "1", "yes")
        self.FRIGATE_SCORE_CEILING = float(os.getenv("FRIGATE_SCORE_CEILING", "0.0"))
        self.ENABLE_FRIGATE_SCORES = os.getenv("ENABLE_FRIGATE_SCORES", "true").lower() in ("true", "1", "yes")
        self.FACE_MARGIN = float(os.getenv("FACE_MARGIN", "0.15"))
        self.USE_FULL_RESOLUTION = os.getenv("USE_FULL_RESOLUTION", "true").lower() in ("true", "1", "yes")
        self.ENABLE_FACE_ALIGNMENT = os.getenv("ENABLE_FACE_ALIGNMENT", "true").lower() in ("true", "1", "yes")
        self.ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() in ("true", "1", "yes")
        self.CACHE_DIR = os.getenv("CACHE_DIR", ".if_cache")

        # Fall back to config file for non-sensitive values (API_KEY not stored here)
        if CONFIG_FILE.exists():
            try:
                data = json.loads(CONFIG_FILE.read_text())
                self.IMMICH_URL = self.IMMICH_URL or data.get("IMMICH_URL")
                if not os.getenv("OUTPUT_DIR"):
                    self.OUTPUT_DIR = data.get("OUTPUT_DIR", self.OUTPUT_DIR)
            except (json.JSONDecodeError, OSError) as e:
                logging.warning(f"Failed to load config file: {e}")

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton — mainly useful for testing or delayed env setup."""
        cls._instance = None

    def save(self) -> None:
        """Persist non-sensitive configuration to file.

        API_KEY is intentionally excluded — store it in .env or as an
        environment variable instead of a plain-text config file.
        """
        try:
            CONFIG_FILE.write_text(
                json.dumps(
                    {
                        "IMMICH_URL": self.IMMICH_URL,
                        "OUTPUT_DIR": self.OUTPUT_DIR,
                    },
                    indent=2,
                )
            )
            logging.info(f"Configuration saved to {CONFIG_FILE}")
        except OSError as e:
            logging.error(f"Failed to save config: {e}")

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


# Singleton instance — use a lazy property pattern to avoid import-time side effects
# when env vars aren't yet set. Call Config.instance() or just access attributes on
# the module-level `Config` (which delegates to the singleton).
class _ConfigAccessor:
    """Lazy accessor that defers singleton creation until first attribute access.

    This avoids reading .env and config files at import time, so environment
    variables set after importing the module are properly picked up.
    """

    def __getattr__(self, name: str):
        return getattr(_Config(), name)

    def __setattr__(self, name: str, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            setattr(_Config(), name, value)

    def reset(self) -> None:
        """Reset the underlying singleton."""
        _Config.reset()

    def interactive_setup(self) -> None:
        """Delegate to the singleton."""
        _Config().interactive_setup()

    def validate(self) -> None:
        """Delegate to the singleton."""
        _Config().validate()

    def save(self) -> None:
        """Delegate to the singleton."""
        _Config().save()


Config = _ConfigAccessor()


class ConfigManager:
    @staticmethod
    def get() -> _Config:
        return _Config()


def get_headers() -> dict[str, str]:
    """Return HTTP headers for Immich API requests."""
    return {"x-api-key": Config.API_KEY or "", "Accept": "application/json"}

