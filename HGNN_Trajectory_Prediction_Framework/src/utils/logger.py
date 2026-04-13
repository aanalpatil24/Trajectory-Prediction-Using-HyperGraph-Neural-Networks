"""
Logging utilities for training and evaluation.
"""
import logging
import os
import sys
from datetime import datetime


def setup_logger(
    name: str,
    log_dir: str = "outputs/logs",
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logger with file and console handlers.

    Args:
        name: Logger name
        log_dir: Directory for log files
        log_file: Log file name (default: timestamp.log)
        level: Logging level

    Returns:
        Logger instance
    """
    os.makedirs(log_dir, exist_ok=True)

    if log_file is None:
        log_file = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    log_path = os.path.join(log_dir, log_file)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
