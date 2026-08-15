"""Wire-protocol pydantic models — round-trip and discriminator behaviour."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from intelloptim._internal.schemas import (
    API_VERSION,
    AskRequest,
    AskResponse,
    EnqueueRequest,
    EnqueueResponse,
    ParameterSpec,
    SamplerSpec,
    StudyResolveRequest,
    StudyResolveResponse,
    TellRequest,
)
from intelloptim.core.parameters import Float, Int
from intelloptim.core.pruners import NopPruner
from intelloptim.core.samplers.classical import TPE
from intelloptim.core.samplers.quantum import QuantumQAOA
from intelloptim.core.trial import TrialState


def test_api_version_constant() -> None:
    assert API_VERSION == "v1"


def test_resolve_request_round_trip() -> None:
    req = StudyResolveRequest(
        name="demo",
        direction="minimize",
        parameters=[Float("x", 0.0, 1.0), Int("n", 1, 10)],
        sampler=TPE(seed=42),
        pruner=NopPruner(),
    )
    revived = StudyResolveRequest.model_validate(req.model_dump(mode="json"))
    assert revived == req


def test_resolve_request_with_quantum_sampler() -> None:
    req = StudyResolveRequest(
        name="quantum",
        direction="minimize",
        parameters=[Float("x", 0.0, 1.0)],
        sampler=QuantumQAOA(reps=2),
        pruner=NopPruner(),
    )
    dumped = req.model_dump(mode="json")
    assert dumped["sampler"]["kind"] == "quantum_qaoa"
    revived = StudyResolveRequest.model_validate(dumped)
    assert isinstance(revived.sampler, QuantumQAOA)


def test_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError, match="extra"):
        StudyResolveRequest.model_validate(
            {
                "name": "x",
                "direction": "minimize",
                "parameters": [],
                "sampler": {"kind": "tpe"},
                "pruner": {"kind": "nop"},
                "rogue": "field",
            }
        )


def test_resolve_response() -> None:
    r = StudyResolveResponse(study_id="sha256:abc", existed=False, n_trials=0)
    assert r.existed is False


def test_ask_request() -> None:
    req = AskRequest(config_digest="sha256:def", worker_id="w-1")
    assert req.worker_id == "w-1"


def test_ask_response() -> None:
    r = AskResponse(trial_id="t-7", number=7, params={"x": 0.42})
    assert r.params["x"] == 0.42


def test_tell_request_state_enum() -> None:
    req = TellRequest(
        config_digest="sha256:def",
        value=1.23,
        state=TrialState.COMPLETE,
    )
    assert req.state is TrialState.COMPLETE


def test_tell_request_pruned_without_value() -> None:
    # Pruned trials may not have a final value.
    req = TellRequest(config_digest="sha256:d", state=TrialState.PRUNED)
    assert req.value is None


def test_enqueue_request_completed() -> None:
    req = EnqueueRequest(
        config_digest="sha256:d",
        params={"x": 0.5},
        value=0.42,
    )
    assert req.value == 0.42


def test_enqueue_request_waiting() -> None:
    req = EnqueueRequest(config_digest="sha256:d", params={"x": 0.5})
    assert req.value is None


def test_enqueue_response() -> None:
    r = EnqueueResponse(trial_id="t-1", state=TrialState.WAITING)
    assert r.state is TrialState.WAITING


def test_parameter_spec_discriminator_round_trip() -> None:
    adapter = TypeAdapter(ParameterSpec)
    floats = adapter.validate_python(
        {"kind": "float", "name": "x", "low": 0.0, "high": 1.0}
    )
    assert isinstance(floats, Float)


def test_sampler_spec_discriminator_round_trip() -> None:
    adapter = TypeAdapter(SamplerSpec)
    s = adapter.validate_python({"kind": "tpe", "seed": 7})
    assert isinstance(s, TPE)
    assert s.seed == 7
