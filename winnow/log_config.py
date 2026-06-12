"""Logging configuration for winnow."""

import logging
import os
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

    # Configure root logger; close existing handlers before replacing them
    root = logging.getLogger()
    root.setLevel(level)
    for h in root.handlers[:]:
        h.close()
        root.removeHandler(h)

    # Rich console handler - uses shared console to avoid breaking progress bars
    root.addHandler(RichHandler(rich_tracebacks=True, markup=True, console=console))

    # File handler (always debug level) — log file respects OUTPUT_DIR if set
    log_dir = os.environ.get("OUTPUT_DIR", ".")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "winnow.log")
    file_handler = logging.FileHandler(log_path)
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

