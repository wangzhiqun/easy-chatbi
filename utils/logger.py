import sys
from pathlib import Path
from loguru import logger
from .config import get_config

config = get_config()

log_dir = Path(config.log_file).parent
log_dir.mkdir(parents=True, exist_ok=True)

logger.remove()

logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=config.log_level,
    colorize=True
)

logger.add(
    config.log_file,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level=config.log_level,
    rotation="10 MB",
    retention="7 days",
    compression="zip"
)

__all__ = ['logger']