"""Structured logging configuration.

Provides a pre-configured logger with console + file output.
Usage anywhere in the project:
    from energy_forecasting.logger import logger
    logger.info("Pipeline started")
"""

import sys

from loguru import logger

from energy_forecasting.config import PROJECT_ROOT, settings

# Directory for log files
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Remove loguru's default handler so we control the format
logger.remove()

# Console — human-readable, colorized
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    level="INFO",
    colorize=True,
)

# File — detailed, with rotation
logger.add(
    LOG_DIR / "energy_forecasting.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}",
    level="DEBUG",
    rotation="10 MB",
    retention="30 days",
    encoding="utf-8",
)

