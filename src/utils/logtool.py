import logging
import logging.config


def setup_logger(name, log_to_console: bool = True, log_file_path: str = None):
    # Define logging configuration
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "[%(asctime)s] [%(levelname)s] %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {},
        "loggers": {"": {"level": "DEBUG", "propagate": True, "handlers": []}},
    }

    # Add console handler if logging to console
    if log_to_console:
        log_config["handlers"]["console"] = {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
            "stream": "ext://sys.stdout",
        }
        log_config["loggers"][""]["handlers"].append("console")

    # Add file handler if logging to file
    if log_file_path is not None:
        log_config["handlers"]["file"] = {
            "class": "logging.FileHandler",
            "formatter": "standard",
            "level": "INFO",
            "filename": log_file_path,
            "mode": "a",
        }
        log_config["loggers"][""]["handlers"].append("file")

    # Configure logging using the defined configuration
    logging.config.dictConfig(log_config)

    # Return the logger
    return logging.getLogger(name)
