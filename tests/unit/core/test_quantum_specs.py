"""Specs of the quantum policy samplers."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

import pytest
from pydantic import TypeAdapter, ValidationError

from intelloptim.core.samplers import (
    PluginProviderSpec,
    QuantumGausSamplerSpec,
    QuantumMixtureSamplerSpec,
    SamplerKind,
    SamplerSpec,
    SimQuantumGausSamplerSpec,
    SimQuantumMixtureSamplerSpec,
)
from intelloptim.core.samplers.base import BaseRebuildSamplerSpec

ADAPTER: TypeAdapter[Any] = TypeAdapter(SamplerSpec)
HW: Dict[str, Any] = {
    "provider": {"provider": "ibm_runtime"},
    "shots": 1000,
    "max_total_shots": 10**7,
}


def _digest(spec: Any) -> str:
    payload = json.dumps(spec.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


class TestKinds:
    def test_four_distinct_kinds_named_after_their_classes(self) -> None:
        kinds = {
            SimQuantumGausSamplerSpec().kind: "sim_quantum_gaus",
            QuantumGausSamplerSpec(**HW).kind: "quantum_gaus",
            SimQuantumMixtureSamplerSpec().kind: "sim_quantum_mixture",
            QuantumMixtureSamplerSpec(**HW).kind: "quantum_mixture",
        }
        assert {k.value for k in kinds} == set(kinds.values())
        assert not any(k.name in ("QUANTUM_QAOA", "QUANTUM_VQE") for k in SamplerKind)

    @pytest.mark.parametrize(
        "raw",
        [
            {"kind": "sim_quantum_gaus"},
            {"kind": "sim_quantum_mixture", "shots": 256},
            {"kind": "quantum_gaus", **HW},
            {"kind": "quantum_mixture", **HW},
            {
                "kind": "quantum_gaus",
                **{**HW, "provider": {"provider": "aws", "options": {"region": "eu"}}},
            },
        ],
    )
    def test_union_round_trip(self, raw: Dict[str, Any]) -> None:
        spec = ADAPTER.validate_python(raw)
        assert ADAPTER.validate_python(spec.model_dump(mode="json")) == spec

    def test_all_are_rebuildable_and_remote(self) -> None:
        for spec in (SimQuantumGausSamplerSpec(), SimQuantumMixtureSamplerSpec()):
            assert isinstance(spec, BaseRebuildSamplerSpec)
            assert (
                type(spec).runs_remote is True and type(spec).rebuild_policy is not None
            )

    def test_rebuild_every_is_the_cohort_size(self) -> None:
        assert SimQuantumGausSamplerSpec().rebuild_every == 10
        assert SimQuantumGausSamplerSpec(rebuild_every=25).rebuild_every == 25


class TestSimulatedVersusHardware:
    def test_simulated_is_exact_by_default(self) -> None:
        assert SimQuantumGausSamplerSpec().shots is None
        assert SimQuantumMixtureSamplerSpec(shots=512).shots == 512

    def test_simulated_samplers_cannot_name_a_provider(self) -> None:
        with pytest.raises(ValidationError):
            SimQuantumGausSamplerSpec(provider={"provider": "ibm_runtime"})  # type: ignore[call-arg]

    @pytest.mark.parametrize("missing", ["provider", "shots", "max_total_shots"])
    def test_hardware_requires_provider_shots_and_budget(self, missing: str) -> None:
        fields = {k: v for k, v in HW.items() if k != missing}
        with pytest.raises(ValidationError):
            QuantumGausSamplerSpec(**fields)
        with pytest.raises(ValidationError):
            QuantumMixtureSamplerSpec(**fields)

    def test_budget_must_pay_for_at_least_one_update(self) -> None:
        # 4 epochs x (2*18 + 1) + 1 = 149 circuits x 1000 shots
        with pytest.raises(ValidationError, match="cannot pay for one update"):
            QuantumGausSamplerSpec(**{**HW, "max_total_shots": 148_999})
        assert QuantumGausSamplerSpec(**{**HW, "max_total_shots": 149_000})

    def test_no_credentials_can_be_put_in_a_spec(self) -> None:
        for field in ("token", "api_key", "password", "channel", "instance"):
            with pytest.raises(ValidationError):
                QuantumGausSamplerSpec(**HW, **{field: "x"})
            with pytest.raises(ValidationError):
                QuantumGausSamplerSpec(
                    **{**HW, "provider": {"provider": "ibm_runtime", field: "x"}}
                )

    def test_simulated_and_hardware_studies_never_share_an_identity(self) -> None:
        assert _digest(SimQuantumGausSamplerSpec()) != _digest(
            QuantumGausSamplerSpec(**HW)
        )
        assert _digest(SimQuantumGausSamplerSpec(shots=1000)) != _digest(
            SimQuantumGausSamplerSpec()
        )

    def test_provider_and_encoding_are_part_of_the_identity(self) -> None:
        named = {**HW, "provider": {"provider": "ibm_runtime", "backend_name": "ibm_x"}}
        assert _digest(QuantumGausSamplerSpec(**HW)) != _digest(
            QuantumGausSamplerSpec(**named)
        )
        assert _digest(SimQuantumGausSamplerSpec()) != _digest(
            SimQuantumGausSamplerSpec(categorical_encoding="head")
        )


class TestPluginProvider:
    def test_a_plugin_provider_carries_plain_options(self) -> None:
        spec = QuantumGausSamplerSpec(
            **{
                **HW,
                "provider": {
                    "provider": "aws_braket",
                    "options": {"region": "eu-west-1"},
                },
            }
        )
        assert isinstance(spec.provider, PluginProviderSpec)
        assert spec.provider.options == {"region": "eu-west-1"}

    @pytest.mark.parametrize(
        "key",
        [
            "token",
            "API_KEY",
            "aws_secret_access_key",
            "password",
            "my-api-key",
            "credentials",
        ],
    )
    def test_secret_looking_options_are_rejected(self, key: str) -> None:
        with pytest.raises(ValidationError, match="looks like a secret"):
            PluginProviderSpec(provider="aws", options={key: "x"})

    @pytest.mark.parametrize("name", ["ibm_runtime", "", "Has-Caps", "1abc"])
    def test_provider_name_rules(self, name: str) -> None:
        with pytest.raises(ValidationError):
            PluginProviderSpec(provider=name)


class TestValidation:
    def test_defaults_describe_the_report_circuit(self) -> None:
        gaus, mix = SimQuantumGausSamplerSpec(), SimQuantumMixtureSamplerSpec()
        assert gaus.n_qubits == 6 and gaus.n_layers == 3 and gaus.n_angles == 18
        assert mix.n_mode_qubits == 2 and mix.n_geometry_qubits == 4
        assert mix.base_qubits == 6 and mix.n_angles == 18
        assert gaus.categorical_encoding == mix.categorical_encoding == "register"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_qubits": 1},
            {"n_qubits": 17},
            {"n_layers": 0},
            {"clip_eps": 0.0},
            {"clip_eps": 1.0},
            {"ppo_epochs": 0},
            {"lr_theta": 0.0},
            {"lr_phi": 2.0},
            {"kl_stop": 0.0},
            {"kl_stop": float("inf")},
            {"entropy_coef": -1.0},
            {"advantage": "zscore"},
            {"sigma_min": 0.0},
            {"init_scale": 0.01},
            {"min_cohort": 1},
            {"min_cohort": 11},
            {"min_category_weight": 0.0},
            {"min_category_weight": 0.9},
            {"categorical_encoding": "dense"},
            {"shots": 0},
            {"seed": "abc"},
        ],
    )
    def test_gaus_rejects_bad_values(self, kwargs: Dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            SimQuantumGausSamplerSpec(**kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_mode_qubits": 0},
            {"n_mode_qubits": 5},
            {"n_geometry_qubits": 1},
            {"min_mode_weight": 0.0},
            {"min_mode_weight": 0.9},
        ],
    )
    def test_mixture_rejects_bad_values(self, kwargs: Dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            SimQuantumMixtureSamplerSpec(**kwargs)

    def test_largest_mixture_is_within_the_qubit_cap(self) -> None:
        assert (
            SimQuantumMixtureSamplerSpec(
                n_mode_qubits=4, n_geometry_qubits=12
            ).base_qubits
            == 16
        )

    def test_specs_are_frozen_and_forbid_unknown_fields(self) -> None:
        spec = SimQuantumGausSamplerSpec()
        with pytest.raises(ValidationError):
            spec.seed = 1  # type: ignore[misc]
        with pytest.raises(ValidationError):
            SimQuantumGausSamplerSpec(unknown=1)  # type: ignore[call-arg]
