import logging.config
import os
from pathlib import Path

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(levelname)s:%(name)s: %(message)s"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "INFO",
            "formatter": "standard",
            "filename": "./temp.log",
            "mode": "w",
        },
        "error_file": {
            "class": "logging.FileHandler",
            "level": "ERROR",
            "formatter": "standard",
            "filename": "./error.log",
            "mode": "w",
        },
    },
    "root": {
        "handlers": ["console", "file", "error_file"],
        "level": "DEBUG",
    },
}


def setup_logging(logging_fpath: Path):
    logging_fpath.parent.mkdir(parents=True, exist_ok=True)
    try:
        LOGGING_CONFIG["handlers"]["file"]["filename"] = logging_fpath
        logging.config.dictConfig(LOGGING_CONFIG)
    except Exception:
        logging.basicConfig(
            level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s"
        )
