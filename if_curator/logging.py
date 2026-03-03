"""Logging configuration for if-curator."""

import logging
import warnings

from rich.console import Console
from rich.logging import RichHandler

# Shared console instance - must be the same as used by Progress bars
console = Console()

NOISY_LOGGERS = (
    "urllib3",
    "PIL",
    "ultralytics",
    "insightface",
    "onnxruntime",
    "matplotlib",
    "transformers",
    "torch",
)


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging with Rich console and file output."""
    level = logging.DEBUG if verbose else logging.INFO

    # Configure root logger
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    # Rich console handler - uses shared console to avoid breaking progress bars
    root.addHandler(RichHandler(rich_tracebacks=True, markup=True, console=console))

    # File handler (always debug level)
    file_handler = logging.FileHandler("immich_export.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    root.addHandler(file_handler)

    # Silence noisy libraries
    for lib in NOISY_LOGGERS:
        logging.getLogger(lib).setLevel(logging.WARNING)

    # Suppress Python warnings from ML libraries
    warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

    return root
