"""
OpenTelemetry logging integration for ATM project.

Preserves existing log formats while adding OpenTelemetry support.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from atm.utils.otel import get_config, initialize_otel


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a logger with optional OpenTelemetry integration.

    This function preserves the existing log format while optionally
    adding OpenTelemetry logging export. Existing log formats remain unchanged.

    Args:
        name: Logger name.
        level: Logging level.
        log_file: Optional log file path.
        format_string: Optional custom format string (default format preserved).

    Returns:
        Configured logger instance with same format as before.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Default format - PRESERVED from original
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)

    # Console handler - SAME AS BEFORE
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified) - SAME AS BEFORE
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # OpenTelemetry integration - OPTIONAL, doesn't change existing format
    config = get_config()
    if config.is_enabled() and config.logs_enabled:
        # Ensure OpenTelemetry is initialized
        initialize_otel()
        logger.debug("OpenTelemetry logging integration available")

    return logger
