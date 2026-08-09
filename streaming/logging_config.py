"""
SentinelAI — Structured JSON Logging.

Configures JSON-formatted structured logging with transaction ID
correlation across the full pipeline: ingestion → scoring → sink.

Each log entry includes:
    - timestamp (ISO 8601)
    - level
    - component (producer/consumer/scorer/api)
    - worker_id (for multi-worker tracing)
    - transaction_id (for end-to-end transaction tracing)
    - message

Usage:
    from streaming.logging_config import setup_logging
    setup_logging(component="consumer", worker_id="worker-0")
    logger = logging.getLogger(__name__)
    logger.info("Scored transaction", extra={"transaction_id": 12345})
"""

import logging
import os
import sys
from typing import Optional

# Try to use python-json-logger; fall back to standard if unavailable
try:
    from pythonjsonlogger import jsonlogger

    _HAS_JSON_LOGGER = True
except ImportError:
    _HAS_JSON_LOGGER = False


class TransactionLogFilter(logging.Filter):
    """Adds default transaction_id and worker_id fields to all log records.

    If the logging call doesn't provide these via `extra={}`,
    they default to empty strings.
    """

    def __init__(self, worker_id: str = "", component: str = "") -> None:
        super().__init__()
        self.worker_id = worker_id
        self.component = component

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "transaction_id"):
            record.transaction_id = ""
        if not hasattr(record, "worker_id"):
            record.worker_id = self.worker_id
        if not hasattr(record, "component"):
            record.component = self.component
        return True


def setup_logging(
    component: str = "app",
    worker_id: str = "",
    level: str = "INFO",
    json_format: bool = True,
) -> None:
    """Configure structured logging for the application.

    Args:
        component: Component name (producer, consumer, scorer, api).
        worker_id: Worker identifier for multi-worker setups.
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        json_format: If True, output JSON-formatted logs.
            If False or python-json-logger is not installed,
            falls back to a human-readable format.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Add filter for default fields
    tx_filter = TransactionLogFilter(
        worker_id=worker_id,
        component=component,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    handler.addFilter(tx_filter)

    if json_format and _HAS_JSON_LOGGER:
        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(component)s "
                "%(worker_id)s %(transaction_id)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    else:
        formatter = logging.Formatter(
            fmt=(
                "%(asctime)s [%(levelname)s] "
                "[%(component)s/%(worker_id)s] "
                "%(name)s: %(message)s"
                "%(transaction_id_suffix)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # Patch formatter to add transaction_id suffix
        original_format = formatter.format

        def patched_format(record):
            tx_id = getattr(record, "transaction_id", "")
            record.transaction_id_suffix = (
                f" [tx={tx_id}]" if tx_id else ""
            )
            return original_format(record)

        formatter.format = patched_format

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    logging.getLogger("confluent_kafka").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
