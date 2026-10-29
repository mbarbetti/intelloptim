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
    ShouldPruneRequest,
    ShouldPruneResponse,
    StudyResolveRequest,
    StudyResolveResponse,
    TellRequest,
    TrialState,
)
from intelloptim.core.parameters import FloatSpec, IntSpec
from intelloptim.core.pruners import NopPrunerSpec
from intelloptim.core.samplers.optuna import TPESamplerSpec
from intelloptim.core.samplers.quantum import SimQuantumGausSamplerSpec


def test_api_version_constant() -> None:
    assert API_VERSION == "v1"


def test_resolve_request_round_trip() -> None:
    req = StudyResolveRequest(
        study_name="demo",
        direction="minimize",
        parameters=[FloatSpec("x", 0.0, 1.0), IntSpec("n", 1, 10)],
        sampler_spec=TPESamplerSpec(seed=42),
        pruner_spec=NopPrunerSpec(),
    )
    revived = StudyResolveRequest.model_validate(req.model_dump(mode="json"))
    assert revived == req


def test_resolve_request_fields_match_create_study_arguments() -> None:
    """The wire body uses the same names and order as ``Client.create_study``."""
    import inspect

    from intelloptim.client.api import Client

    arguments = [p for p in inspect.signature(Client.create_study).parameters][1:]
    assert list(StudyResolveRequest.model_fields) == arguments
    assert arguments == [
        "study_name",
        "direction",
        "parameters",
        "sampler_spec",
        "pruner_spec",
    ]


def test_resolve_request_with_quantum_sampler() -> None:
    req = StudyResolveRequest(
        study_name="quantum",
        direction="minimize",
        parameters=[FloatSpec("x", 0.0, 1.0)],
        sampler_spec=SimQuantumGausSamplerSpec(n_layers=2),
        pruner_spec=NopPrunerSpec(),
    )
    dumped = req.model_dump(mode="json")
    assert dumped["sampler_spec"]["kind"] == "sim_quantum_gaus"
    revived = StudyResolveRequest.model_validate(dumped)
    assert isinstance(revived.sampler_spec, SimQuantumGausSamplerSpec)


def test_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError, match="extra"):
        StudyResolveRequest.model_validate(
            {
                "study_name": "x",
                "direction": "minimize",
                "parameters": [],
                "sampler_spec": {"kind": "tpe"},
                "pruner_spec": {"kind": "nop"},
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


def test_tell_request_has_no_intermediate_values() -> None:
    # Intermediate values travel through ``should_prune``, not ``tell``.
    with pytest.raises(ValidationError):
        TellRequest(
            config_digest="sha256:d",
            state=TrialState.PRUNED,
            intermediate_values={0: 1.0},
        )


def test_should_prune_request_roundtrip() -> None:
    req = ShouldPruneRequest(config_digest="sha256:d", intermediate_values={0: 1.5})
    wire = req.model_dump(mode="json")
    # JSON object keys are strings; the model parses them back to ints.
    assert wire == {"config_digest": "sha256:d", "intermediate_values": {"0": 1.5}}
    assert ShouldPruneRequest.model_validate(wire) == req


def test_should_prune_request_values_default_to_empty() -> None:
    assert ShouldPruneRequest(config_digest="sha256:d").intermediate_values == {}


def test_should_prune_models_forbid_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ShouldPruneRequest(config_digest="d", step=1)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        ShouldPruneResponse(should_prune=True, reason="x")  # type: ignore[call-arg]
    assert ShouldPruneResponse(should_prune=False).should_prune is False


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
    r = EnqueueResponse(trial_id="t-1", number=3, state=TrialState.WAITING)
    assert r.state is TrialState.WAITING
    assert r.number == 3


def test_enqueue_response_requires_number() -> None:
    with pytest.raises(ValidationError):
        EnqueueResponse.model_validate({"trial_id": "t-1", "state": "waiting"})


def test_parameter_spec_discriminator_round_trip() -> None:
    adapter = TypeAdapter(ParameterSpec)
    floats = adapter.validate_python(
        {"kind": "float", "name": "x", "low": 0.0, "high": 1.0}
    )
    assert isinstance(floats, FloatSpec)


def test_sampler_spec_discriminator_round_trip() -> None:
    adapter = TypeAdapter(SamplerSpec)
    s = adapter.validate_python({"kind": "tpe", "seed": 7})
    assert isinstance(s, TPESamplerSpec)
    assert s.seed == 7


class TestTrialState:
    def test_wire_values(self) -> None:
        assert {s.name: s.value for s in TrialState} == {
            "WAITING": "waiting",
            "RUNNING": "running",
            "COMPLETE": "complete",
            "PRUNED": "pruned",
            "FAILED": "failed",
        }

    @pytest.mark.parametrize("state", list(TrialState))
    def test_round_trip_from_wire_value(self, state: TrialState) -> None:
        assert TrialState(state.value) is state

    def test_is_a_string(self) -> None:
        assert TrialState.COMPLETE == "complete"

    def test_exposed_at_the_top_level(self) -> None:
        import intelloptim as intopt

        assert intopt.TrialState is TrialState
        assert intopt.TrialState.PRUNED.value == "pruned"
        assert "TrialState" in intopt.__all__
