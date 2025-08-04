# FAIRshake/utils/logger.py

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Union


def setup_logging(
    log_dir: Optional[Union[str, Path]] = None,
    log_level: Union[int, str] = logging.INFO,  # Accepts either int or str
) -> Tuple[logging.Logger, Optional[Path], List[str]]:
    """
    Set up logging configuration.

    Parameters
    ----------
    log_dir : str or Path, optional
        Directory where log files will be stored. If None, logs will not be saved to a file.
    log_level : int or str, optional
        Logging level (e.g., "DEBUG", "INFO", "ERROR") or logging constant (e.g., logging.INFO).
        Defaults to logging.INFO.

    Returns
    -------
    tuple
        A tuple containing:
        - logger: Configured logger instance.
        - log_file: Path to the log file if log_dir is provided; otherwise, None.
        - log_records: List to collect log records in memory for optional use.
    """
    # Convert string log levels to logging constants, if necessary
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)

    logger: logging.Logger = logging.getLogger("FAIRshake")
    logger.setLevel(log_level)  # Set the logger to the specified level

    # Prevent adding multiple handlers if setup_logging is called multiple times
    if not logger.hasHandlers():
        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Console handler setup
        console_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        log_file: Optional[Path] = None
        if log_dir:
            log_dir_path: Path = Path(log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)
            timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir_path / f"fairshake_{timestamp}.log"

            # File handler setup
            file_handler: logging.FileHandler = logging.FileHandler(log_file, mode="a")
            file_handler.setLevel(log_level)
            file_formatter: logging.Formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
    else:
        # If handlers already exist, retrieve the file handler if any
        handlers = logger.handlers
        log_file = None
        for handler in handlers:
            if isinstance(handler, logging.FileHandler):
                log_file = Path(handler.baseFilename)
                break

    # Fabio-specific logging setup
    fabio_logger: logging.Logger = logging.getLogger("fabio")
    fabio_logger.setLevel(logging.INFO)

    # Initialize a list to store log records if needed (optional)
    log_records: List[str] = []

    # Ensure that all 'FAIRshake.*' loggers propagate to 'FAIRshake' and do not have their own FileHandlers
    for logger_name in logging.Logger.manager.loggerDict.keys():
        if logger_name.startswith("FAIRshake.") and logger_name != "FAIRshake":
            child_logger = logging.getLogger(logger_name)
            child_logger.propagate = True  # Ensure logs propagate to parent
            # Remove any FileHandlers from child loggers
            handlers_to_remove = [
                h for h in child_logger.handlers if isinstance(h, logging.FileHandler)
            ]
            for h in handlers_to_remove:
                child_logger.removeHandler(h)

    logger.debug("Logging has been configured.")
    return logger, log_file, log_records
