# ================================================================
# src/utils/logger.py
# ----------------------------------------------------------------
# Central logging setup for the entire project.
# Every module imports get_logger() from here.
#
# Usage in any file:
#   from src.utils.logger import get_logger
#   log = get_logger(__name__)
#   log.info("Downloading bhavcopy...")
#   log.error("HTTP 404 for 2025-06-08")
#
# Output goes to:
#   1. Terminal   → see progress while running
#   2. logs/app.log → saved file for review later
# ================================================================

import logging
import sys
from pathlib import Path
from config.settings import LOG_DIR

# ----------------------------------------------------------------
# Log format
# ----------------------------------------------------------------
# Example output:
# 2025-06-09 18:30:01 | INFO     | downloader | Downloading bhavcopy...
# 2025-06-09 18:30:04 | ERROR    | downloader | HTTP 404 — skipping date

LOG_FORMAT  = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Log file path
LOG_FILE = LOG_DIR / "app.log"


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger for the given module name.

    Parameters
    ----------
    name : str
        Pass __name__ from the calling module.
        e.g. get_logger(__name__)
        This makes log output show the module name automatically.

    Returns
    -------
    logging.Logger
        Configured logger that writes to terminal + log file.

    Example
    -------
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Starting download...")
    log.warning("MTO file missing for this date")
    log.error("Database connection failed")
    """

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already exists
    # This happens when the same module is imported multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # --------------------------------------------------------
    # Handler 1 — Terminal output
    # Shows INFO and above (hides DEBUG in terminal)
    # --------------------------------------------------------
    terminal_handler = logging.StreamHandler(sys.stdout)
    terminal_handler.setLevel(logging.INFO)
    terminal_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

    # --------------------------------------------------------
    # Handler 2 — File output
    # Saves DEBUG and above to logs/app.log
    # Append mode — does not overwrite on restart
    # --------------------------------------------------------
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

    logger.addHandler(terminal_handler)
    logger.addHandler(file_handler)

    # Prevent log messages bubbling up to root logger
    # Avoids duplicate output
    logger.propagate = False

    return logger