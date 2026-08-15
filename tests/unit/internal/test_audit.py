"""Audit-event helpers — schema and routing."""

from __future__ import annotations

import io
import json
import logging
from typing import Iterator

import pytest

from intelloptim._internal.audit import (
    emit,
    get_audit_logger,
    login_failure,
    login_success,
    study_created,
    study_stopped,
    token_authentication_failure,
    token_issued,
    token_revoked,
)
from intelloptim._internal.logging import (
    ROOT_LOGGER_NAME,
    JsonFormatter,
    configure_logging,
    reset_for_tests,
)


@pytest.fixture
def captured(tmp_path) -> Iterator[io.StringIO]:
    """Configure logging with a single in-memory stream and yield it."""
    reset_for_tests()
    buf = io.StringIO()
    configure_logging(level="DEBUG", format="json", force=True)
    handler = logging.StreamHandler(buf)
    handler.setFormatter(JsonFormatter())
    logging.getLogger(ROOT_LOGGER_NAME).addHandler(handler)
    yield buf
    reset_for_tests()


def _records(buf: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in buf.getvalue().strip().splitlines() if line]


def test_emit_attaches_event_category(captured) -> None:
    emit("test.event", outcome="success", actor="u-1")
    [r] = _records(captured)
    assert r["event_category"] == "audit"
    assert r["event"] == "test.event"
    assert r["outcome"] == "success"
    assert r["actor"] == "u-1"


def test_emit_anonymous_default(captured) -> None:
    emit("anon.event", outcome="failure")
    [r] = _records(captured)
    assert r["actor"] == "anonymous"


def test_login_helpers_produce_canonical_events(captured) -> None:
    login_success(user_id="u-1", email="a@b.c")
    login_failure(email="a@b.c", reason="bad-password")
    rs = _records(captured)
    assert {r["event"] for r in rs} == {"login.success", "login.failure"}
    failure = next(r for r in rs if r["event"] == "login.failure")
    assert failure["level"] == "WARNING"
    assert failure["reason"] == "bad-password"


def test_token_helpers(captured) -> None:
    token_issued(user_id="u-1", token_prefix="ABC", name="ci")
    token_revoked(user_id="u-1", token_prefix="ABC")
    token_authentication_failure(prefix="zzz", reason="unknown")
    rs = _records(captured)
    events = {r["event"] for r in rs}
    assert events == {
        "token.issued",
        "token.revoked",
        "token.auth_failed",
    }


def test_study_helpers(captured) -> None:
    study_created(study_id="sha256:1", user_id="u-1", name="toy")
    study_stopped(study_id="sha256:1", user_id="u-1")
    rs = _records(captured)
    assert all(r["actor"] == "u-1" for r in rs)
    assert {r["event"] for r in rs} == {"study.created", "study.stopped"}


def test_audit_logger_is_a_distinct_subtree() -> None:
    reset_for_tests()
    configure_logging(level="INFO", format="json", force=True)
    audit = get_audit_logger()
    assert audit.name == f"{ROOT_LOGGER_NAME}.audit"
    reset_for_tests()
