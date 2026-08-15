"""Logging configuration shared across components.

Two surfaces matter:

* **Operational** logs — request lifecycle, errors, internal events. Get
  them via :func:`get_logger` (returns loggers under the
  ``intelloptim.*`` namespace, plus FastAPI's loggers re-routed for
  consistency).
* **Audit** logs — security-relevant events (auth attempts, token CRUD,
  study state changes). Get them via
  :func:`intelloptim._internal.audit.get_audit_logger`, which returns a
  separate ``intelloptim.audit`` logger that can be routed to its own
  sink via ``INTELLOPTIM_AUDIT_FILE``.

Configuration is driven by :class:`~intelloptim.server.config.Settings`
or the equivalent env vars and is applied exactly once per process by
:func:`configure_logging`.

Format selection:

* ``json`` — every record is a single JSON object on its own line.
  Machine-parseable; recommended for production / log shippers.
* ``console`` — coloured key=value pairs. Human-readable; recommended
  for local development.
* ``auto`` (default) — ``console`` when stderr is a TTY, ``json``
  otherwise. Matches the convention used by ``uvicorn`` / ``structlog``.

The configured root logger is ``intelloptim`` (not the global root) —
unrelated libraries' loggers are untouched.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Dict, Optional

ROOT_LOGGER_NAME = "intelloptim"
_CONFIGURED = False

# Reserved attribute names on a LogRecord that the json formatter should
# skip when scraping `extra=` keyword payload.
_RESERVED_ATTRS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "asctime",
        "message",
        "taskName",
    }
)


def _record_extras(record: logging.LogRecord) -> Dict[str, Any]:
    """Return the structured payload attached to a record via ``extra=``."""
    return {
        k: v
        for k, v in record.__dict__.items()
        if k not in _RESERVED_ATTRS and not k.startswith("_")
    }


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per record.

    Standard fields: ``ts`` (ISO-8601 UTC), ``level``, ``logger``,
    ``message``. Any keyword arguments passed via ``extra=`` are merged
    in at the top level. Exception info, if present, is emitted as a
    ``traceback`` field.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(_record_extras(record))
        if record.exc_info:
            payload["traceback"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, separators=(",", ":"))


class ConsoleFormatter(logging.Formatter):
    """Human-readable formatter with light ANSI colour when on a TTY.

    Output shape::

        13:42:11.123 INFO  intelloptim.server  message k=v k2=v2
    """

    _LEVEL_COLOURS: Dict[str, str] = {
        "DEBUG": "\x1b[36m",  # cyan
        "INFO": "\x1b[32m",  # green
        "WARNING": "\x1b[33m",  # yellow
        "ERROR": "\x1b[31m",  # red
        "CRITICAL": "\x1b[1;31m",  # bold red
    }
    _RESET = "\x1b[0m"
    _DIM = "\x1b[2m"

    def __init__(self, *, use_colour: bool) -> None:
        super().__init__()
        self._use_colour = use_colour

    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        ts_full = f"{ts}.{int(record.msecs):03d}"
        level = record.levelname.ljust(8)
        if self._use_colour:
            colour = self._LEVEL_COLOURS.get(record.levelname, "")
            level = f"{colour}{level}{self._RESET}"
            ts_full = f"{self._DIM}{ts_full}{self._RESET}"
            name = f"{self._DIM}{record.name}{self._RESET}"
        else:
            name = record.name

        extras = _record_extras(record)
        extras_str = ""
        if extras:
            kv = " ".join(f"{k}={v}" for k, v in sorted(extras.items()))
            extras_str = (
                f" {self._DIM}{kv}{self._RESET}" if self._use_colour else f" {kv}"
            )

        line = f"{ts_full} {level} {name}  {record.getMessage()}{extras_str}"
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


def _resolve_format(format_choice: str) -> str:
    """Resolve ``auto`` to a concrete ``json`` / ``console`` choice."""
    choice = format_choice.lower()
    if choice in ("json", "console"):
        return choice
    if choice == "auto":
        return "console" if sys.stderr.isatty() else "json"
    raise ValueError(
        f"INTELLOPTIM_LOG_FORMAT must be one of "
        f"'auto', 'json', 'console'; got {format_choice!r}"
    )


def _build_handler(
    stream: Any,
    format_choice: str,
    use_colour: bool,
) -> logging.Handler:
    handler = logging.StreamHandler(stream=stream)
    if format_choice == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(ConsoleFormatter(use_colour=use_colour))
    return handler


def configure_logging(
    *,
    level: str = "INFO",
    format: str = "auto",
    file: Optional[str] = None,
    audit_file: Optional[str] = None,
    force: bool = False,
) -> None:
    """Configure the ``intelloptim`` logger tree.

    Idempotent unless ``force=True``. Called automatically by
    :func:`get_logger` the first time it's used, and explicitly at
    server startup by :func:`intelloptim.server.app.create_app`.

    Args:
        level: Minimum level for both operational and audit loggers.
        format: ``auto``, ``json``, or ``console``.
        file: Optional path. When set, operational logs *also* go to
            this file (in JSON format regardless of the console choice
            so they're machine-parseable on disk).
        audit_file: Optional path for the ``intelloptim.audit`` logger.
            When unset, audit records flow through the same stderr
            handler as operational ones with ``event_category=audit``.
        force: If ``True``, tear down and rebuild the configuration even
            if it has already been applied.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    resolved_format = _resolve_format(format)
    log_level = level.upper()

    root = logging.getLogger(ROOT_LOGGER_NAME)
    # Tear down any handlers from a previous configure call.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(log_level)
    root.propagate = False

    stderr_handler = _build_handler(
        sys.stderr,
        resolved_format,
        use_colour=resolved_format == "console" and sys.stderr.isatty(),
    )
    root.addHandler(stderr_handler)

    if file:
        # File output is always JSON for ingestability.
        file_handler = logging.FileHandler(file, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        root.addHandler(file_handler)

    audit_logger = logging.getLogger(f"{ROOT_LOGGER_NAME}.audit")
    for h in list(audit_logger.handlers):
        audit_logger.removeHandler(h)
    audit_logger.setLevel(log_level)

    if audit_file:
        audit_handler = logging.FileHandler(audit_file, encoding="utf-8")
        audit_handler.setFormatter(JsonFormatter())
        audit_logger.addHandler(audit_handler)
        # Don't propagate to the operational handler so audit isn't duplicated.
        audit_logger.propagate = False
    else:
        # Inherit operational handlers; the ``event_category=audit`` field
        # on every record lets operators filter.
        audit_logger.propagate = True

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger under the ``intelloptim`` namespace.

    Args:
        name: Module name, typically ``__name__``. The leading
            ``intelloptim.`` prefix is preserved when present.

    Returns:
        A :class:`logging.Logger` instance.
    """
    if not _CONFIGURED:
        configure_logging(
            level=os.environ.get("INTELLOPTIM_LOG_LEVEL", "INFO"),
            format=os.environ.get("INTELLOPTIM_LOG_FORMAT", "auto"),
            file=os.environ.get("INTELLOPTIM_LOG_FILE") or None,
            audit_file=os.environ.get("INTELLOPTIM_AUDIT_FILE") or None,
        )
    if not name.startswith(ROOT_LOGGER_NAME):
        name = f"{ROOT_LOGGER_NAME}.{name}"
    return logging.getLogger(name)


def reset_for_tests() -> None:
    """Tear down logging configuration. Test-only."""
    global _CONFIGURED
    _CONFIGURED = False
    for name in (ROOT_LOGGER_NAME, f"{ROOT_LOGGER_NAME}.audit"):
        logger = logging.getLogger(name)
        for h in list(logger.handlers):
            logger.removeHandler(h)
        logger.setLevel(logging.NOTSET)
        logger.propagate = True


__all__ = [
    "ConsoleFormatter",
    "JsonFormatter",
    "ROOT_LOGGER_NAME",
    "configure_logging",
    "get_logger",
    "reset_for_tests",
]
