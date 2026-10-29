"""Pruning from the client — ``Trial.report`` / ``should_prune`` and ``optimize``."""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Callable, Dict, Iterator, List, Optional

import httpx
import pytest

import intelloptim as intopt
from intelloptim.client.api import Client
from intelloptim.client.errors import (
    ConfigDigestMismatchError,
    IntelloptimError,
    TrialNotFoundError,
    TrialPruned,
)
from intelloptim.client.study import StudyHandle
from intelloptim.client.trial import Trial


class _Server:
    """Stand-in server: hands out trials, answers ``should_prune`` from a script."""

    def __init__(self) -> None:
        self.requests: List[httpx.Request] = []
        self.verdicts: List[bool] = []
        self.status: Optional[int] = None
        self._next = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path.endswith("/ask"):
            number, self._next = self._next, self._next + 1
            return httpx.Response(
                200,
                json={"trial_id": str(number), "number": number, "params": {"x": 0.5}},
            )
        if path.endswith("/should_prune"):
            if self.status is not None:
                return httpx.Response(self.status, json={"detail": "nope"})
            verdict = self.verdicts.pop(0) if self.verdicts else False
            return httpx.Response(200, json={"should_prune": verdict})
        if path.endswith("/tell"):
            return httpx.Response(200, json={"ack": True})
        return httpx.Response(404, json={"detail": "not found"})

    def bodies(self, suffix: str) -> List[Dict[str, Any]]:
        return [
            json.loads(r.content) for r in self.requests if r.url.path.endswith(suffix)
        ]


@pytest.fixture
def setup(token: str) -> Iterator[Callable[[], Any]]:
    clients: List[Client] = []

    def factory() -> Any:
        server = _Server()
        http = httpx.Client(transport=httpx.MockTransport(server), base_url="http://m")
        client = Client(url="http://m", token=token, http_client=http)
        clients.append(client)
        study = StudyHandle(
            client=client,
            study_id="suid:abc",
            existed=False,
            n_trials=0,
            config_digest="digest-1",
        )
        return study, server

    yield factory
    for client in clients:
        client.close()


class TestReport:
    def test_stores_values_locally_without_a_request(self, setup) -> None:
        study, server = setup()
        trial = study.ask()
        before = len(server.requests)
        trial.report(1.5, 0)
        trial.report(1, 1)
        assert trial.intermediate_values == {0: 1.5, 1: 1.0}
        assert len(server.requests) == before

    def test_intermediate_values_is_a_copy(self) -> None:
        trial = Trial(trial_id="t", number=0, params={})
        trial.report(1.0, 0)
        trial.intermediate_values[5] = 9.0
        assert trial.intermediate_values == {0: 1.0}

    @pytest.mark.parametrize("step", [1.0, "1", None, True])
    def test_rejects_non_int_steps(self, step: Any) -> None:
        with pytest.raises(TypeError, match="step"):
            Trial(trial_id="t", number=0, params={}).report(1.0, step)

    @pytest.mark.parametrize("value", ["1", None, True, [1.0]])
    def test_rejects_non_numeric_values(self, value: Any) -> None:
        with pytest.raises(TypeError, match="value"):
            Trial(trial_id="t", number=0, params={}).report(value, 0)

    def test_rejects_negative_step(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            Trial(trial_id="t", number=0, params={}).report(1.0, -1)

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_values(self, value: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            Trial(trial_id="t", number=0, params={}).report(value, 0)

    def test_rejects_a_step_reported_twice(self) -> None:
        trial = Trial(trial_id="t", number=0, params={})
        trial.report(1.0, 3)
        with pytest.raises(ValueError, match="already reported"):
            trial.report(2.0, 3)
        assert trial.intermediate_values == {3: 1.0}


class TestShouldPrune:
    def test_nothing_reported_makes_no_request(self, setup) -> None:
        study, server = setup()
        trial = study.ask()
        before = len(server.requests)
        assert trial.should_prune() is False
        assert len(server.requests) == before

    def test_posts_digest_and_values_to_the_trial_endpoint(self, setup) -> None:
        study, server = setup()
        trial = study.ask()
        trial.report(0.9, 0)
        trial.report(0.8, 1)
        trial.should_prune()
        request = server.requests[-1]
        assert request.url.path == "/studies/suid:abc/trials/0/should_prune"
        assert request.headers["authorization"].startswith("Bearer ")
        assert json.loads(request.content) == {
            "config_digest": "digest-1",
            "intermediate_values": {"0": 0.9, "1": 0.8},
        }

    def test_returns_the_server_verdict(self, setup) -> None:
        study, server = setup()
        server.verdicts = [False, True]
        trial = study.ask()
        trial.report(1.0, 0)
        assert trial.should_prune() is False
        trial.report(2.0, 1)
        assert trial.should_prune() is True

    def test_sends_only_values_not_sent_before(self, setup) -> None:
        study, server = setup()
        trial = study.ask()
        trial.report(1.0, 0)
        trial.should_prune()
        trial.report(2.0, 1)
        trial.should_prune()
        sent = [b["intermediate_values"] for b in server.bodies("/should_prune")]
        assert sent == [{"0": 1.0}, {"1": 2.0}]

    def test_without_new_values_repeats_the_last_verdict(self, setup) -> None:
        study, server = setup()
        server.verdicts = [True]
        trial = study.ask()
        trial.report(1.0, 0)
        assert trial.should_prune() is True
        count = len(server.bodies("/should_prune"))
        # A second call must not turn a "prune" into "continue".
        assert trial.should_prune() is True
        assert len(server.bodies("/should_prune")) == count

    def test_failed_request_keeps_values_pending(self, setup) -> None:
        study, server = setup()
        trial = study.ask()
        trial.report(1.0, 0)
        server.status = 503
        with pytest.raises(IntelloptimError):
            trial.should_prune()
        server.status = None
        trial.should_prune()
        sent = [b["intermediate_values"] for b in server.bodies("/should_prune")]
        assert sent == [{"0": 1.0}, {"0": 1.0}]

    def test_trial_not_from_a_study_cannot_prune(self) -> None:
        trial = Trial(trial_id="t", number=0, params={})
        assert trial.should_prune() is False  # nothing to send, nothing to ask
        trial.report(1.0, 0)
        with pytest.raises(RuntimeError, match="not attached to a study"):
            trial.should_prune()

    def test_unknown_trial_maps_to_trial_not_found(self, setup) -> None:
        study, server = setup()
        trial = study.ask()
        trial.report(1.0, 0)
        server.status = 404
        with pytest.raises(TrialNotFoundError):
            trial.should_prune()

    def test_digest_mismatch_maps_to_config_digest_mismatch(self, setup) -> None:
        study, server = setup()
        trial = study.ask()
        trial.report(1.0, 0)
        server.status = 409
        with pytest.raises(ConfigDigestMismatchError):
            trial.should_prune()


class TestOptimizeWithPruning:
    def test_pruned_trial_is_told_as_pruned_with_its_last_value(self, setup) -> None:
        study, server = setup()
        server.verdicts = [False, True]

        def objective(trial: Trial) -> float:
            for step, loss in enumerate([3.0, 2.0, 1.0]):
                trial.report(loss, step)
                if trial.should_prune():
                    raise TrialPruned
            return 0.0

        study.optimize(objective, n_trials=1)
        (tell,) = server.bodies("/tell")
        assert tell["state"] == "pruned"
        assert tell["value"] == 2.0
        assert "intermediate_values" not in tell

    def test_pruned_without_a_report_has_no_value(self, setup) -> None:
        study, server = setup()

        def objective(trial: Trial) -> float:
            raise TrialPruned

        study.optimize(objective, n_trials=1)
        (tell,) = server.bodies("/tell")
        assert (tell["state"], tell["value"]) == ("pruned", None)

    def test_loop_continues_after_a_pruned_trial(self, setup) -> None:
        study, server = setup()

        def objective(trial: Trial) -> float:
            if trial.number == 0:
                raise TrialPruned
            return 4.0

        study.optimize(objective, n_trials=3)
        states = [b["state"] for b in server.bodies("/tell")]
        assert states == ["pruned", "complete", "complete"]

    def test_pruning_is_not_a_failure(self, setup, log_records) -> None:
        study, _ = setup()
        logging.getLogger("intelloptim").setLevel(logging.INFO)

        def objective(trial: Trial) -> float:
            raise TrialPruned

        study.optimize(objective, n_trials=1)
        levels = [r.levelname for r in log_records.records]
        assert "INFO" in levels  # the pruning was logged ...
        assert "WARNING" not in levels  # ... but not as a failure

    @pytest.mark.parametrize("catch", [None, ValueError, Exception])
    def test_pruned_regardless_of_catch(self, setup, catch: Any) -> None:
        study, server = setup()

        def objective(trial: Trial) -> float:
            raise TrialPruned

        study.optimize(objective, n_trials=1, catch=catch)
        assert [b["state"] for b in server.bodies("/tell")] == ["pruned"]

    def test_other_errors_still_fail_the_trial(self, setup) -> None:
        study, server = setup()

        def objective(trial: Trial) -> float:
            raise ValueError("boom")

        study.optimize(objective, n_trials=1)
        assert [b["state"] for b in server.bodies("/tell")] == ["failed"]


class TestTrialPrunedSignal:
    def test_is_not_an_sdk_error(self) -> None:
        # ``except IntelloptimError`` must not swallow the control-flow signal.
        assert not issubclass(TrialPruned, IntelloptimError)
        assert issubclass(TrialPruned, Exception)

    def test_is_reached_through_the_client_errors(self) -> None:
        assert intopt.client.errors.TrialPruned is TrialPruned
        assert "TrialPruned" not in intopt.__all__

    def test_is_listed_in_the_errors_module(self) -> None:
        from intelloptim.client import errors

        assert "TrialPruned" in errors.__all__
