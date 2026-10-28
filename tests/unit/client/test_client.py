"""``Client`` — study creation, quota snapshot and its log output."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Iterator, List

import httpx
import pytest

import intelloptim as intopt
from intelloptim.client.api import Client
from intelloptim.client.errors import (
    IntelloptimAPIError,
    MissingTokenError,
    StudyLimitExceededError,
)

_ME: Dict[str, Any] = {
    "id": "u1",
    "email": "u@example.com",
    "is_active": True,
    "role_name": "free",
    "calls_remaining": None,
    "active_studies": 0,
    "max_active_studies": None,
    "period_resets_at": None,
    "is_group_context": False,
    "group_name": None,
    "group_calls_remaining": None,
}


class _Server:
    """A programmable stand-in for the server that records the requests."""

    def __init__(self, me: Dict[str, Any]) -> None:
        self.me = me
        self.requests: List[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/auth/me":
            return httpx.Response(200, json=self.me)
        if request.url.path == "/studies/resolve":
            return httpx.Response(
                200, json={"study_id": "suid:abc", "existed": False, "n_trials": 0}
            )
        return httpx.Response(404, json={"detail": "not found"})

    def paths(self) -> List[str]:
        return [r.url.path for r in self.requests]


@pytest.fixture
def make_client(token: str) -> Iterator[Callable[..., Any]]:
    clients: List[Client] = []

    def factory(**me: Any) -> Any:
        server = _Server({**_ME, **me})
        http = httpx.Client(transport=httpx.MockTransport(server), base_url="http://m")
        client = Client(url="http://m", token=token, http_client=http)
        clients.append(client)
        return client, server

    yield factory
    for client in clients:
        client.close()


def _create(client: Client, **overrides: Any) -> Any:
    arguments: Dict[str, Any] = dict(
        study_name="demo",
        direction="minimize",
        parameters=[intopt.params.Float("x", -1.0, 1.0)],
        sampler_spec=intopt.samplers.TPESamplerSpec(seed=1),
    )
    arguments.update(overrides)
    return client.create_study(**arguments)


class TestCreateStudy:
    def test_posts_the_declared_study_and_returns_a_handle(self, make_client) -> None:
        client, server = make_client()
        study = _create(client, pruner_spec=intopt.pruners.MedianPrunerSpec())

        assert study.id == "suid:abc"
        assert study.existed is False
        body = json.loads(server.requests[-1].content)
        assert list(body) == [
            "study_name",
            "direction",
            "parameters",
            "sampler_spec",
            "pruner_spec",
        ]
        assert body["study_name"] == "demo"
        assert body["sampler_spec"]["kind"] == "tpe"
        assert body["pruner_spec"]["kind"] == "median"

    def test_pruner_defaults_to_nop(self, make_client) -> None:
        client, server = make_client()
        _create(client)
        assert json.loads(server.requests[-1].content)["pruner_spec"] == {"kind": "nop"}

    def test_the_old_argument_names_are_rejected(self, make_client) -> None:
        client, _ = make_client()
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            client.create_study(
                name="demo",
                direction="minimize",
                parameters=[],
                sampler_spec=intopt.samplers.TPESamplerSpec(),
            )

    def test_the_digest_is_computed_locally(self, make_client) -> None:
        client, _ = make_client()
        first = _create(client)
        second = _create(client)
        different = _create(client, direction="maximize")
        assert first.config_digest == second.config_digest
        assert first.config_digest != different.config_digest


class TestStudyLimit:
    def test_raises_before_sending_the_study(self, make_client) -> None:
        client, server = make_client(active_studies=3, max_active_studies=3)
        with pytest.raises(StudyLimitExceededError):
            _create(client)
        assert "/studies/resolve" not in server.paths()

    def test_below_the_cap_is_fine(self, make_client) -> None:
        client, server = make_client(active_studies=2, max_active_studies=3)
        _create(client)
        assert "/studies/resolve" in server.paths()


class TestQuotaSnapshot:
    def test_is_fetched_once_and_cached(self, make_client) -> None:
        client, server = make_client()
        _create(client)
        _create(client)
        assert server.paths().count("/auth/me") == 1

    def test_error_response_maps_to_an_api_error(self, token: str) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(401, json={"detail": "bad token"})
        )
        with httpx.Client(transport=transport, base_url="http://m") as http:
            client = Client(url="http://m", token=token, http_client=http)
            with pytest.raises(IntelloptimAPIError) as info:
                _create(client)
        assert info.value.status_code == 401
        assert info.value.detail == "bad token"


class TestQuotaWarning:
    def test_limited_role_logs_a_warning(self, make_client, log_records) -> None:
        client, _ = make_client(
            role_name="free", calls_remaining=5, period_resets_at="2026-11-01"
        )
        _create(client)
        messages = [r.getMessage() for r in log_records.records]
        assert any(
            "Role 'free': 5 ask/tell call(s) remaining" in m
            and "2026-11-01" in m
            and r.levelno == 30
            for m, r in zip(messages, log_records.records, strict=True)
        )

    def test_limited_group_logs_a_warning(self, make_client, log_records) -> None:
        client, _ = make_client(
            is_group_context=True, group_name="lab", group_calls_remaining=7
        )
        _create(client)
        messages = [r.getMessage() for r in log_records.records]
        assert any("Group 'lab': 7 ask/tell call(s) remaining" in m for m in messages)

    def test_unlimited_budget_does_not_warn(self, make_client, log_records) -> None:
        client, _ = make_client()
        _create(client)
        assert not [r for r in log_records.records if r.levelno >= 30]

    def test_no_logging_error_is_raised(self, make_client, log_records) -> None:
        """Regression: stray ``%`` arguments used to make the handler fail."""
        import logging

        errors: List[BaseException] = []
        handler = logging.Handler()
        handler.handleError = lambda record: errors.append(Exception(record.msg))  # type: ignore[assignment]
        handler.emit = lambda record: record.getMessage()  # type: ignore[assignment]
        package_logger = logging.getLogger("intelloptim")
        package_logger.addHandler(handler)
        try:
            client, _ = make_client(role_name="free", calls_remaining=5)
            _create(client)
        finally:
            package_logger.removeHandler(handler)
        assert not errors


def test_missing_token_fails_at_construction(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("INTELLOPTIM_TOKEN", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(MissingTokenError):
        Client(url="http://m")


def test_context_manager_closes_an_owned_pool(token: str) -> None:
    with Client(url="http://m", token=token) as client:
        assert not client._http.is_closed
    assert client._http.is_closed


def test_an_injected_http_client_is_left_open(token: str) -> None:
    with httpx.Client(base_url="http://m") as http:
        with Client(url="http://m", token=token, http_client=http):
            pass
        assert not http.is_closed


def test_url_and_token_properties(token: str) -> None:
    client = Client(url="http://m/", token=f"  {token}  ")
    assert client.url == "http://m"
    assert client.token == token
    client.close()
