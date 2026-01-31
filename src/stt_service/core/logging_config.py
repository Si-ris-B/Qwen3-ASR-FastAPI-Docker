import logging
import sys


def setup_logging(log_level_str: str, logger_name: str) -> logging.Logger:
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    # Professional format: [Timestamp] [Logger] [Level] - Message
    log_formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)

    app_logger = logging.getLogger(logger_name)
    if not app_logger.handlers:
        app_logger.setLevel(log_level)
        app_logger.addHandler(stream_handler)
        app_logger.propagate = False

    # Silence third-party noise but keep ours
    logging.getLogger("transformers").setLevel(logging.WARNING)
    return app_logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"qwen3_asr_api.{name}")