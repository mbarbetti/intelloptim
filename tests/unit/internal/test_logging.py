"""Logging configuration: formatters, levels, idempotency."""

from __future__ import annotations

import json
import logging
from typing import Iterator, List

import pytest

from intelloptim._internal.logging import (
    ROOT_LOGGER_NAME,
    ConsoleFormatter,
    JsonFormatter,
    configure_logging,
    get_logger,
    reset_for_tests,
)


@pytest.fixture(autouse=True)
def _clean_logging() -> Iterator[None]:
    reset_for_tests()
    yield
    reset_for_tests()


def _make_record(
    *, name: str = "intelloptim.test", level: int = logging.INFO, msg: str = "hello"
) -> logging.LogRecord:
    return logging.LogRecord(
        name=name,
        level=level,
        pathname=__file__,
        lineno=10,
        msg=msg,
        args=(),
        exc_info=None,
    )


class TestJsonFormatter:
    def test_minimal_record_round_trips(self) -> None:
        record = _make_record()
        payload = json.loads(JsonFormatter().format(record))
        assert payload["level"] == "INFO"
        assert payload["logger"] == "intelloptim.test"
        assert payload["message"] == "hello"
        assert "ts" in payload

    def test_extras_are_merged(self) -> None:
        record = _make_record()
        record.study_id = "sha256:abc"
        record.duration_ms = 42.5
        payload = json.loads(JsonFormatter().format(record))
        assert payload["study_id"] == "sha256:abc"
        assert payload["duration_ms"] == 42.5

    def test_exception_traceback_included(self) -> None:
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            import sys

            record = logging.LogRecord(
                "intelloptim.test",
                logging.ERROR,
                __file__,
                10,
                "fail",
                (),
                sys.exc_info(),
            )
        payload = json.loads(JsonFormatter().format(record))
        assert "traceback" in payload
        assert "RuntimeError" in payload["traceback"]


class TestConsoleFormatter:
    def test_uncoloured_output(self) -> None:
        record = _make_record()
        line = ConsoleFormatter(use_colour=False).format(record)
        assert "INFO" in line
        assert "intelloptim.test" in line
        assert "hello" in line
        assert "\x1b[" not in line

    def test_coloured_output_has_ansi(self) -> None:
        record = _make_record(level=logging.ERROR, msg="oh no")
        line = ConsoleFormatter(use_colour=True).format(record)
        assert "\x1b[" in line
        assert "oh no" in line

    def test_extras_rendered(self) -> None:
        record = _make_record()
        record.study_id = "sha256:abc"
        line = ConsoleFormatter(use_colour=False).format(record)
        assert "study_id=sha256:abc" in line


class TestConfigureLogging:
    def test_idempotent_by_default(self) -> None:
        configure_logging(level="DEBUG", format="json")
        root = logging.getLogger(ROOT_LOGGER_NAME)
        n = len(root.handlers)
        configure_logging(level="INFO", format="console")
        assert len(root.handlers) == n  # second call is a no-op

    def test_force_rebuilds(self) -> None:
        configure_logging(level="DEBUG", format="json")
        configure_logging(level="WARNING", format="json", force=True)
        root = logging.getLogger(ROOT_LOGGER_NAME)
        assert root.level == logging.WARNING

    def test_format_choice_validation(self) -> None:
        with pytest.raises(ValueError, match="INTELLOPTIM_LOG_FORMAT"):
            configure_logging(format="banana", force=True)

    def test_get_logger_lazy_configures(self) -> None:
        reset_for_tests()
        log = get_logger("module.x")
        assert log.name == "intelloptim.module.x"
        # configure_logging ran with defaults; root logger has handlers.
        root = logging.getLogger(ROOT_LOGGER_NAME)
        assert root.handlers

    @pytest.mark.parametrize(
        ("requested", "expected"),
        [
            ("module.x", "intelloptim.module.x"),
            ("intelloptim", "intelloptim"),
            ("intelloptim.client.api", "intelloptim.client.api"),
            # Sibling packages sharing the prefix are not inside the tree.
            ("intelloptim_server.app", "intelloptim.intelloptim_server.app"),
            ("intelloptimx", "intelloptim.intelloptimx"),
        ],
    )
    def test_get_logger_namespacing(self, requested: str, expected: str) -> None:
        assert get_logger(requested).name == expected

    def test_sibling_package_records_reach_the_handlers(self) -> None:
        configure_logging(level="DEBUG", format="json", force=True)
        seen: List[str] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                seen.append(record.name)

        logging.getLogger(ROOT_LOGGER_NAME).addHandler(_Capture())
        get_logger("intelloptim_server.app").info("hello")
        assert seen == ["intelloptim.intelloptim_server.app"]

    def test_file_handler_emits_json(self, tmp_path) -> None:
        log_file = tmp_path / "intelloptim.log"
        configure_logging(
            level="DEBUG", format="console", file=str(log_file), force=True
        )
        get_logger("module.x").info("hi", extra={"study_id": "sha256:abc"})
        for h in logging.getLogger(ROOT_LOGGER_NAME).handlers:
            h.flush()
        lines = log_file.read_text(encoding="utf-8").splitlines()
        assert lines, "log file should contain at least one record"
        payload = json.loads(lines[-1])
        assert payload["message"] == "hi"
        assert payload["study_id"] == "sha256:abc"


class TestAuditRouting:
    def test_audit_inherits_main_handlers_by_default(self) -> None:
        configure_logging(level="DEBUG", format="json", force=True)
        audit = logging.getLogger(f"{ROOT_LOGGER_NAME}.audit")
        assert audit.propagate is True
        assert not audit.handlers

    def test_audit_file_creates_dedicated_sink(self, tmp_path) -> None:
        audit_file = tmp_path / "audit.log"
        configure_logging(
            level="DEBUG",
            format="json",
            audit_file=str(audit_file),
            force=True,
        )
        audit = logging.getLogger(f"{ROOT_LOGGER_NAME}.audit")
        assert audit.propagate is False
        assert audit.handlers
        audit.info("test event", extra={"event_category": "audit"})
        for h in audit.handlers:
            h.flush()
        payload = json.loads(audit_file.read_text(encoding="utf-8").strip())
        assert payload["event_category"] == "audit"


def test_capture_with_caplog_works(caplog) -> None:
    """A sanity check that pytest's caplog still captures records under our
    namespace — important so other tests can assert on log output without
    bypassing our config."""
    caplog.set_level(logging.INFO, logger="intelloptim.test")
    configure_logging(level="DEBUG", format="json", force=True)
    log = get_logger("test")

    # caplog inserts its own handler at the root level; our logger has
    # propagate=False so we need to attach caplog's handler explicitly.
    log.addHandler(caplog.handler)
    try:
        log.info("captured", extra={"k": "v"})
    finally:
        log.removeHandler(caplog.handler)

    messages = [r.message for r in caplog.records if r.name == "intelloptim.test"]
    assert "captured" in messages
