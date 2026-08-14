"""Structured JSON logging.

JSON rather than formatted text because logs on Render's free tier are read
through a web console with no grep — searchable key/value pairs are the only way
to answer "what happened to request X" after the fact.

Every log line carries a ``request_id`` where one exists, bound once at the entry
point and inherited by everything downstream, so a single request's path through
classifier, router, provider, and verifier can be reassembled from the log alone.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO", *, json_output: bool = True) -> None:
    """Configure structlog once, at startup.

    ``json_output`` is disabled in local development, where a human is reading
    the terminal and colours beat machine-parseability.
    """
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level.upper())

    processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    processors.append(
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[level.upper()]
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
