"""Configuration management for if-curator."""

import json
import logging
import os
from pathlib import Path
from typing import ClassVar

from dotenv import load_dotenv
from rich.prompt import Prompt

load_dotenv()

CONFIG_FILE = Path(".immich_config.json")


class Config:
    """Singleton configuration with uppercase attribute access for backward compatibility."""

    _instance: ClassVar["Config | None"] = None

    # Configuration values
    IMMICH_URL: str | None = None
    API_KEY: str | None = None
    OUTPUT_DIR: str = "./frigate_train"
    YEARS_FILTER: int = 10

    # Quality filtering
    MIN_FACE_WIDTH: int = 100
    BLUR_THRESHOLD: float = 100.0
    MIN_CONFIDENCE: float = 0.7
    MAX_AUTO_IMAGES: int = 80

    # Output quality
    FACE_MARGIN: float = 0.15
    USE_FULL_RESOLUTION: bool = True
    ENABLE_FACE_ALIGNMENT: bool = True

    # Caching (opt-in to avoid unexpected files)
    ENABLE_CACHE: bool = False
    CACHE_DIR: str = ".if_cache"

    def __new__(cls) -> "Config":
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
        self.MIN_FACE_WIDTH = int(os.getenv("MIN_FACE_WIDTH", "50"))

        # Fall back to config file for missing values
        if CONFIG_FILE.exists():
            try:
                data = json.loads(CONFIG_FILE.read_text())
                self.IMMICH_URL = self.IMMICH_URL or data.get("IMMICH_URL")
                self.API_KEY = self.API_KEY or data.get("API_KEY")
                if not os.getenv("OUTPUT_DIR"):
                    self.OUTPUT_DIR = data.get("OUTPUT_DIR", self.OUTPUT_DIR)
            except (json.JSONDecodeError, OSError) as e:
                logging.warning(f"Failed to load config file: {e}")

    def save(self) -> None:
        """Persist configuration to file."""
        try:
            CONFIG_FILE.write_text(
                json.dumps(
                    {
                        "IMMICH_URL": self.IMMICH_URL,
                        "API_KEY": self.API_KEY,
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
            self.API_KEY = Prompt.ask("Enter Immich API Key", password=True)
            self.save()

    def validate(self) -> None:
        """Raise ValueError if required config is missing."""
        if not self.IMMICH_URL or not self.API_KEY:
            raise ValueError("Missing Immich URL or API Key.")


# Singleton instance and backward-compatible aliases
Config = Config()  # type: ignore[misc]
ConfigManager = type("ConfigManager", (), {"get": staticmethod(lambda: Config)})


def get_headers() -> dict[str, str]:
    """Return HTTP headers for Immich API requests."""
    return {"x-api-key": Config.API_KEY or "", "Accept": "application/json"}
