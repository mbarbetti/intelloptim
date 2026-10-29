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
    quota_exhausted,
    study_access_denied,
    study_created,
    study_stopped,
    token_authentication_failure,
    token_issued,
    token_revoked,
    trial_enqueued,
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


@pytest.mark.parametrize(
    "key", ["event", "outcome", "actor", "event_category", "name", "message", "args"]
)
def test_emit_rejects_reserved_extra_names(captured, key: str) -> None:
    with pytest.raises(ValueError, match=key):
        emit("test.event", outcome="success", extra={key: "x"})
    assert _records(captured) == []


def test_emit_lists_every_clashing_name(captured) -> None:
    with pytest.raises(ValueError, match="actor, name"):
        emit("test.event", outcome="success", extra={"name": 1, "actor": 2, "ok": 3})


def test_emit_accepts_other_extra_names(captured) -> None:
    emit("test.event", outcome="success", extra={"study_id": "s", "study_name": "n"})
    [r] = _records(captured)
    assert r["study_id"] == "s"
    assert r["study_name"] == "n"


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


def test_trial_enqueued_records_who_and_where_but_not_the_data(captured) -> None:
    trial_enqueued(study_id="suid:s", user_id="u-1", trial_number=3)
    [r] = _records(captured)
    assert r["event"] == "trial.enqueued"
    assert r["outcome"] == "success"
    assert r["actor"] == "u-1"
    assert r["study_id"] == "suid:s"
    assert r["trial_number"] == 3
    assert not {"value", "params"} & set(r)


def test_quota_exhausted_is_a_warning_failure(captured) -> None:
    quota_exhausted(user_id="u-1", calls_per_month=100)
    [r] = _records(captured)
    assert r["event"] == "quota.exhausted"
    assert r["outcome"] == "failure"
    assert r["level"] == "WARNING"
    assert r["actor"] == "u-1"
    assert r["calls_per_month"] == 100
    assert r["refused"] == 1


def test_quota_exhausted_reports_the_number_of_refusals(captured) -> None:
    quota_exhausted(user_id="u-1", calls_per_month=100, refused=7)
    [r] = _records(captured)
    assert r["refused"] == 7


def test_study_access_denied_names_the_action(captured) -> None:
    study_access_denied(study_id="suid:s", user_id="u-2", action="stop")
    [r] = _records(captured)
    assert r["event"] == "study.access_denied"
    assert r["outcome"] == "failure"
    assert r["level"] == "WARNING"
    assert r["actor"] == "u-2"
    assert r["study_id"] == "suid:s"
    assert r["action"] == "stop"


def test_audit_logger_is_a_distinct_subtree() -> None:
    reset_for_tests()
    configure_logging(level="INFO", format="json", force=True)
    audit = get_audit_logger()
    assert audit.name == f"{ROOT_LOGGER_NAME}.audit"
    reset_for_tests()
