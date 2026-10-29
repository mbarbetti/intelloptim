"""Logging configuration shared by the SDK, the algorithms and the server.

Two kinds of records are emitted, both under the ``intelloptim`` logger tree:

* **Operational** logs — request lifecycle, errors, internal events. Get a
  logger with :func:`get_logger`.
* **Audit** logs — security-relevant events (authentication attempts, token
  changes, study state changes). They are written through the helpers of
  :mod:`intelloptim._internal.audit` to the ``intelloptim.audit`` logger, which
  can be routed to its own file.

:func:`configure_logging` applies the configuration once per process. Callers
that want specific settings (the server, the evaluator CLI) call it explicitly
at start-up; any other caller gets it for free, because :func:`get_logger`
configures logging on first use from the ``INTELLOPTIM_LOG_LEVEL``,
``INTELLOPTIM_LOG_FORMAT``, ``INTELLOPTIM_LOG_FILE`` and
``INTELLOPTIM_AUDIT_FILE`` environment variables.

Output formats:

* ``json`` — one JSON object per line. Machine-parseable; use it in
  production and with log shippers.
* ``console`` — ``key=value`` pairs, coloured when stderr is a terminal.
  Human-readable; use it in local development.
* ``auto`` (default) — ``console`` when stderr is a terminal, ``json``
  otherwise.

Only the ``intelloptim`` logger is configured, never the global root logger, so
the loggers of unrelated libraries are left untouched. :func:`get_logger` places
every name it is given inside the tree, including the names of sibling packages.
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
    """Return the user-supplied fields of a record (those passed via ``extra=``)."""
    return {
        k: v
        for k, v in record.__dict__.items()
        if k not in _RESERVED_ATTRS and not k.startswith("_")
    }


class JsonFormatter(logging.Formatter):
    """Format records as single-line JSON objects.

    Every object has ``ts`` (ISO-8601, UTC), ``level``, ``logger`` and
    ``message``. Fields passed via ``extra=`` are merged in at the top level,
    and exception information, if any, goes into a ``traceback`` field. Values
    that are not JSON-serialisable are written with ``str()``.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Render *record* as one JSON line.

        Args:
            record (logging.LogRecord): The record to format.

        Returns:
            str: A compact JSON object without a trailing newline.
        """
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
    """Format records as one human-readable line, optionally coloured.

    Fields passed via ``extra=`` are appended as ``key=value`` pairs sorted by
    key, and exception information follows on the next lines. Output shape::

        13:42:11.123 INFO     intelloptim.client  message k=v k2=v2

    The timestamp is in local time. Colour uses ANSI escape codes, so enable it
    only for a terminal.
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
        """Initialise the formatter.

        Args:
            use_colour (bool): Whether to emit ANSI colour codes.
        """
        super().__init__()
        self._use_colour = use_colour

    def format(self, record: logging.LogRecord) -> str:
        """Render *record* as one console line.

        Args:
            record (logging.LogRecord): The record to format.

        Returns:
            str: The line, followed by the traceback if the record has one.
        """
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
    """Resolve ``auto`` to ``json`` or ``console`` and validate the choice.

    Args:
        format_choice (str): ``auto``, ``json`` or ``console``, case-insensitive.

    Returns:
        str: ``json`` or ``console``.

    Raises:
        ValueError: If *format_choice* is none of the accepted values.
    """
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
    """Build a stream handler with the formatter matching *format_choice*.

    Args:
        stream (Any): Text stream to write to, for example ``sys.stderr``.
        format_choice (str): ``json`` or ``console`` (already resolved).
        use_colour (bool): Colour switch for the console formatter; ignored for
            ``json``.

    Returns:
        logging.Handler: The configured handler.
    """
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

    Does nothing if logging is already configured, unless *force* is set. It is
    called explicitly at start-up by the server and the evaluator CLI, and
    implicitly by :func:`get_logger` on first use.

    Operational records go to stderr and, if *file* is given, also to that file.
    Audit records follow the operational handlers unless *audit_file* is given,
    in which case they go only to that file, so they are not duplicated.

    Args:
        level (str): Minimum level for the operational and audit loggers, for
            example ``"INFO"``; case-insensitive.
        format (str): ``auto``, ``json`` or ``console`` (see the module
            docstring). Applies to stderr only.
        file (Optional[str]): Path of an additional operational log file. Always
            JSON, whatever *format* is, so it can be ingested.
        audit_file (Optional[str]): Path of a dedicated audit log file, always
            JSON.
        force (bool): Rebuild the configuration even if it was already applied.
            Handlers installed by an earlier call are removed (not closed).

    Raises:
        ValueError: If *format* or *level* is not recognised.
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
    """Return a logger in the ``intelloptim`` tree, configuring logging if needed.

    The first call in a process applies :func:`configure_logging` with settings
    read from ``INTELLOPTIM_LOG_LEVEL`` (default ``INFO``),
    ``INTELLOPTIM_LOG_FORMAT`` (default ``auto``), ``INTELLOPTIM_LOG_FILE`` and
    ``INTELLOPTIM_AUDIT_FILE``. Later calls only look the logger up.

    Args:
        name (str): Logger name, normally ``__name__``. A name that is not
            ``intelloptim`` itself or inside it (``intelloptim.<...>``) is
            placed under it as ``intelloptim.<name>``, so a sibling package
            such as ``intelloptim_server`` still logs through the configured
            handlers.

    Returns:
        logging.Logger: The logger called *name*.
    """
    if not _CONFIGURED:
        configure_logging(
            level=os.environ.get("INTELLOPTIM_LOG_LEVEL", "INFO"),
            format=os.environ.get("INTELLOPTIM_LOG_FORMAT", "auto"),
            file=os.environ.get("INTELLOPTIM_LOG_FILE") or None,
            audit_file=os.environ.get("INTELLOPTIM_AUDIT_FILE") or None,
        )
    if name != ROOT_LOGGER_NAME and not name.startswith(f"{ROOT_LOGGER_NAME}."):
        name = f"{ROOT_LOGGER_NAME}.{name}"
    return logging.getLogger(name)


def reset_for_tests() -> None:
    """Remove the handlers and levels installed by :func:`configure_logging`.

    Lets a test start from an unconfigured state, so that the next
    :func:`get_logger` configures logging again. Not for production code.
    """
    global _CONFIGURED
    _CONFIGURED = False
    for name in (ROOT_LOGGER_NAME, f"{ROOT_LOGGER_NAME}.audit"):
        logger = logging.getLogger(name)
        for h in list(logger.handlers):
            logger.removeHandler(h)
        logger.setLevel(logging.NOTSET)
        logger.propagate = True


__all__ = ["ROOT_LOGGER_NAME", "configure_logging", "get_logger"]
