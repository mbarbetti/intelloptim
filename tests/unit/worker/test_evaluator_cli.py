"""``worker/evaluator_cli.py`` — spec loading, health probe and the full loop.

The server is replaced by an ``httpx.MockTransport`` handler, so these are unit
tests: no network and no server. The real client and evaluator against a real
server run in the private ``intelloptim-server`` integration suite.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Iterator, List

import httpx
import pytest

from intelloptim._internal.logging import reset_for_tests
from intelloptim.core.objective import _clear_registry_for_tests, objective
from intelloptim.worker import evaluator_cli
from intelloptim.worker.evaluator_cli import _health_probe, _load_spec, run_evaluator

_SPEC: Dict[str, Any] = {
    "study_name": "demo",
    "direction": "minimize",
    "parameters": [{"kind": "float", "name": "x", "low": 0.0, "high": 1.0}],
    "sampler_spec": {"kind": "tpe", "seed": 1},
    "pruner_spec": {"kind": "nop"},
}


@pytest.fixture(autouse=True)
def _isolate() -> Iterator[None]:
    _clear_registry_for_tests()
    reset_for_tests()
    yield
    _clear_registry_for_tests()
    reset_for_tests()


class TestLoadSpec:
    def test_file_wins(self, tmp_path) -> None:
        path = tmp_path / "spec.json"
        path.write_text(json.dumps(_SPEC))

        @objective("o", study_spec={"study_name": "other"})
        def fn(trial: Any) -> float:
            return 0.0

        assert _load_spec(str(path), fn) == _SPEC

    def test_spec_attached_to_the_objective(self) -> None:
        @objective("registered-under-another-name", study_spec=_SPEC)
        def fn(trial: Any) -> float:
            return 0.0

        assert _load_spec(None, fn) == _SPEC

    def test_no_spec_anywhere(self) -> None:
        @objective("plain")
        def fn(trial: Any) -> float:
            return 0.0

        with pytest.raises(RuntimeError, match="--spec-file"):
            _load_spec(None, fn)

    def test_unregistered_objective(self) -> None:
        with pytest.raises(RuntimeError, match="No study spec"):
            _load_spec(None, lambda trial: 0.0)

    def test_malformed_json_file(self, tmp_path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        with pytest.raises(json.JSONDecodeError):
            _load_spec(str(path), lambda trial: 0.0)


class TestHealthProbe:
    def test_healthy_server(self, monkeypatch) -> None:
        seen: List[Any] = []

        def fake_get(url: str, **kwargs: Any) -> httpx.Response:
            seen.append((url, kwargs["headers"]))
            return httpx.Response(200, request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx, "get", fake_get)
        _health_probe("https://example.org/", "tok")
        assert seen == [
            ("https://example.org/healthz", {"Authorization": "Bearer tok"})
        ]

    def test_unreachable_server_exits(self, monkeypatch) -> None:
        def fake_get(url: str, **kwargs: Any) -> httpx.Response:
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(httpx, "get", fake_get)
        with pytest.raises(SystemExit) as info:
            _health_probe("https://example.org", "tok")
        assert info.value.code == 1

    def test_error_status_exits(self, monkeypatch) -> None:
        def fake_get(url: str, **kwargs: Any) -> httpx.Response:
            return httpx.Response(503, request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx, "get", fake_get)
        with pytest.raises(SystemExit) as info:
            _health_probe("https://example.org", "tok")
        assert info.value.code == 1


class _Server:
    """Stand-in for the intelloptim server, reached through a MockTransport."""

    def __init__(self) -> None:
        self.requests: List[httpx.Request] = []
        self._next = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/auth/me":
            return httpx.Response(
                200, json={"id": "u", "email": "u@example.org", "is_active": True}
            )
        if path == "/studies/resolve":
            return httpx.Response(
                200, json={"study_id": "suid:s", "existed": False, "n_trials": 0}
            )
        if path.endswith("/ask"):
            number, self._next = self._next, self._next + 1
            return httpx.Response(
                200,
                json={"trial_id": str(number), "number": number, "params": {"x": 0.5}},
            )
        if path.endswith("/tell"):
            return httpx.Response(200, json={"ack": True})
        if path.endswith("/best_trial"):
            return httpx.Response(
                200,
                json={
                    "available": True, "trial_id": "0", "number": 0,
                    "params": {"x": 0.5}, "value": 0.25,
                },
            )  # fmt: skip
        return httpx.Response(404, json={"detail": "unexpected " + path})

    def bodies(self, suffix: str) -> List[Dict[str, Any]]:
        return [
            json.loads(r.content) for r in self.requests if r.url.path.endswith(suffix)
        ]


@pytest.fixture
def server(monkeypatch) -> _Server:
    stub = _Server()
    real_client = httpx.Client

    class _StubbedClient(real_client):  # type: ignore[valid-type, misc]
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(transport=httpx.MockTransport(stub), **kwargs)

    monkeypatch.setattr(httpx, "Client", _StubbedClient)
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kw: httpx.Response(200, request=httpx.Request("GET", url)),
    )
    return stub


class TestRunEvaluator:
    def _run(self, objective_fn: Callable[..., float], **kwargs: Any) -> None:
        @objective("run-test", study_spec=_SPEC)
        def fn(trial: Any) -> float:
            return objective_fn(trial)

        import sys

        sys.modules["intopt_run_test_module"] = type(sys)("intopt_run_test_module")
        sys.modules["intopt_run_test_module"].fn = fn  # type: ignore[attr-defined]
        try:
            run_evaluator(
                "https://example.org",
                "intopt_run_test_module:fn",
                token="tok",
                loglevel="WARNING",
                **kwargs,
            )
        finally:
            del sys.modules["intopt_run_test_module"]

    def test_runs_the_requested_number_of_trials(self, server) -> None:
        values: List[float] = []

        def obj(trial: Any) -> float:
            values.append(trial.sampled_float("x"))
            return 0.25

        self._run(obj, n_trials=3, wid="w-1", site="s-1")
        assert values == [0.5, 0.5, 0.5]
        assert len(server.bodies("/ask")) == 3
        tells = server.bodies("/tell")
        assert [t["state"] for t in tells] == ["complete"] * 3

    def test_every_request_identifies_the_worker_and_site(self, server) -> None:
        self._run(lambda trial: 0.0, n_trials=1, wid="w-9", site="s-9")
        api = [r for r in server.requests if r.url.path != "/healthz"]
        assert api
        assert {r.headers["X-Worker-Id"] for r in api} == {"w-9"}
        assert {r.headers["X-Site"] for r in api} == {"s-9"}
        assert {b["worker_id"] for b in server.bodies("/ask")} == {"w-9"}
        assert {r.headers["Authorization"] for r in api} == {"Bearer tok"}

    def test_identity_defaults_come_from_slurm(self, server, monkeypatch) -> None:
        monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "77")
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "3")
        self._run(lambda trial: 0.0, n_trials=1)
        ask = next(r for r in server.requests if r.url.path.endswith("/ask"))
        assert ask.headers["X-Worker-Id"] == "eval-slurm-77-3"
        assert ask.headers["X-Site"] == "slurm-77"

    def test_study_is_resolved_from_the_attached_spec(self, server) -> None:
        self._run(lambda trial: 0.0, n_trials=1, wid="w", site="s")
        [body] = server.bodies("/studies/resolve")
        assert body["study_name"] == "demo"
        assert body["sampler_spec"]["kind"] == "tpe"

    def test_failures_are_reported_and_the_loop_continues(self, server) -> None:
        def obj(trial: Any) -> float:
            raise ValueError("bad evaluation")

        self._run(obj, n_trials=2, wid="w", site="s")
        assert [t["state"] for t in server.bodies("/tell")] == ["failed", "failed"]

    def test_no_catch_stops_at_the_first_failure(self, server) -> None:
        def obj(trial: Any) -> float:
            raise ValueError("bad evaluation")

        with pytest.raises(ValueError, match="bad evaluation"):
            self._run(obj, n_trials=2, wid="w", site="s", catch_exceptions=False)
        assert len(server.bodies("/ask")) == 1

    def test_a_dead_server_stops_before_any_study_is_created(
        self, server, monkeypatch
    ) -> None:
        def refuse(url: str, **kwargs: Any) -> httpx.Response:
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(httpx, "get", refuse)
        with pytest.raises(SystemExit):
            self._run(lambda trial: 0.0, n_trials=1, wid="w", site="s")
        assert server.requests == []

    def test_spec_file_replaces_the_attached_spec(self, server, tmp_path) -> None:
        path = tmp_path / "spec.json"
        path.write_text(json.dumps({**_SPEC, "study_name": "from-file"}))
        self._run(lambda trial: 0.0, n_trials=1, wid="w", site="s", spec_file=str(path))
        [body] = server.bodies("/studies/resolve")
        assert body["study_name"] == "from-file"


def test_module_exports_the_entry_point_only() -> None:
    assert evaluator_cli.__all__ == ["run_evaluator"]
