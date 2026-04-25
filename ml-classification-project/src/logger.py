"""
Logging Configuration
=====================

This module provides a centralized logging setup for the application.

STUDY NOTE: Why Proper Logging?
-------------------------------
1. print() vs logging:
   - print() goes to stdout only
   - logging can go to files, external services, etc.
   - logging has levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
   - logging includes timestamps, module names, etc.

2. In production:
   - Logs are essential for debugging
   - Logs feed into monitoring systems
   - Logs help with auditing

3. Best practices:
   - Use appropriate log levels
   - Include context (request IDs, user IDs)
   - Don't log sensitive data (passwords, tokens)
   - Structure logs for parsing (JSON format)

Usage:
    from src.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Application started")
    logger.error("Something went wrong", exc_info=True)
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import json


# =============================================================================
# Custom Formatter for JSON Logs
# =============================================================================

class JSONFormatter(logging.Formatter):
    """
    Format logs as JSON for easy parsing by log aggregation tools.

    STUDY NOTE:
    -----------
    Structured logging (JSON) is preferred in production because:
    - Easy to parse by tools (ELK, Splunk, CloudWatch)
    - Consistent format
    - Can include arbitrary fields
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data)


# =============================================================================
# Custom Formatter for Console (Human Readable)
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """
    Format logs with colors for console output.

    STUDY NOTE:
    -----------
    Colors make it easier to spot errors in development.
    Disable colors in production or when outputting to files.
    """

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors
        self.fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        self.datefmt = "%Y-%m-%d %H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        # Save original format
        original_fmt = self._style._fmt

        # Apply color if enabled
        if self.use_colors and record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            self._style._fmt = f"{color}{self.fmt}{self.RESET}"
        else:
            self._style._fmt = self.fmt

        # Format the record
        result = super().format(record)

        # Restore original format
        self._style._fmt = original_fmt

        return result


# =============================================================================
# Logger Factory
# =============================================================================

def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    json_format: bool = False,
    use_colors: bool = True
) -> None:
    """
    Configure the root logger.

    Parameters:
    -----------
    level : str
        Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : Path, optional
        If provided, also log to this file
    json_format : bool
        If True, use JSON format (for production)
    use_colors : bool
        If True, use colors in console output

    STUDY NOTE:
    -----------
    Call this once at application startup.
    All loggers created with get_logger() will inherit this config.
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    if json_format:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(ColoredFormatter(use_colors=use_colors))

    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(JSONFormatter())  # Always JSON for files
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Parameters:
    -----------
    name : str
        Usually __name__ of the calling module

    Returns:
    --------
    logging.Logger : Configured logger instance

    Usage:
    ------
    logger = get_logger(__name__)
    logger.info("Something happened")
    logger.error("Error occurred", exc_info=True)
    """
    return logging.getLogger(name)


# =============================================================================
# Context-aware logging (for request tracing)
# =============================================================================

class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds context to all log messages.

    STUDY NOTE:
    -----------
    In web applications, it's useful to track request IDs
    across all log messages for debugging.

    Usage:
        logger = get_context_logger(__name__, request_id="abc123")
        logger.info("Processing request")  # Includes request_id
    """

    def process(self, msg, kwargs):
        # Add extra context to the message
        extra = kwargs.get("extra", {})
        extra["extra_fields"] = self.extra
        kwargs["extra"] = extra
        return msg, kwargs


def get_context_logger(name: str, **context) -> LoggerAdapter:
    """
    Get a logger with additional context.

    Parameters:
    -----------
    name : str
        Logger name (usually __name__)
    **context : dict
        Additional context to include in all logs

    Returns:
    --------
    LoggerAdapter : Logger that includes context in all messages
    """
    logger = logging.getLogger(name)
    return LoggerAdapter(logger, context)


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    # Setup logging
    setup_logging(level="DEBUG", use_colors=True)

    # Get logger
    logger = get_logger(__name__)

    print("\n" + "=" * 50)
    print("Logging Demo")
    print("=" * 50 + "\n")

    # Different log levels
    logger.debug("This is a DEBUG message - detailed info for debugging")
    logger.info("This is an INFO message - general information")
    logger.warning("This is a WARNING message - something unexpected")
    logger.error("This is an ERROR message - something went wrong")
    logger.critical("This is a CRITICAL message - system failing")

    # Log with exception
    try:
        raise ValueError("Example error")
    except Exception:
        logger.error("An exception occurred", exc_info=True)

    # Context logger
    print("\n--- Context Logger ---\n")
    ctx_logger = get_context_logger(__name__, request_id="req-123", user="john")
    ctx_logger.info("Processing user request")

    print("\n" + "=" * 50)
