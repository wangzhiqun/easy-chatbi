"""
Logging configuration for ChatBI platform.
Provides structured logging with different levels and formatters.
"""

import sys
from loguru import logger
from .config import settings


def setup_logging():
    """Configure application logging."""
    # Remove default handler
    logger.remove()

    # Add console handler with custom format
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        colorize=True,
        backtrace=True,
        diagnose=True
    )

    # Add file handler for persistent logging
    logger.add(
        "logs/chatbi.log",
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        rotation="100 MB",
        retention="7 days",
        compression="gz"
    )

    # Add error file handler
    logger.add(
        "logs/error.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        rotation="50 MB",
        retention="30 days",
        compression="gz"
    )


def get_logger(name: str = None):
    """Get logger instance with optional name."""
    if name:
        return logger.bind(name=name)
    return logger


# Initialize logging on import
setup_logging()

# Export logger for easy import
__all__ = ["logger", "get_logger", "setup_logging"]