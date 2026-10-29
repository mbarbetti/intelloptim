"""Optional packages of Optuna-backed samplers — clear failure at resolve time."""

from __future__ import annotations

import os
import tomllib
from typing import Any, Set
from unittest.mock import patch

import pytest

from intelloptim.algorithms.errors import SamplerDependencyError
from intelloptim.algorithms.samplers import (
    _OPTIONAL_DEPENDENCIES,
    _PROVIDER_DEPENDENCIES,
    resolve_sampler,
)
from intelloptim.core.samplers import (
    GPSamplerSpec,
    QMCSamplerSpec,
    QuantumGausSamplerSpec,
    RandomSamplerSpec,
    SASamplerSpec,
    SimQuantumGausSamplerSpec,
    SimQuantumMixtureSamplerSpec,
    TPESamplerSpec,
)

_FIND_SPEC = "intelloptim.algorithms.samplers.find_spec"


def _installed(*names: str) -> Any:
    """A ``find_spec`` stand-in that only knows the given packages."""
    return lambda name: object() if name in names else None


def test_gp_without_torch_and_scipy_names_both_and_the_extra() -> None:
    with patch(_FIND_SPEC, _installed()), pytest.raises(SamplerDependencyError) as info:
        resolve_sampler(GPSamplerSpec())
    message = str(info.value)
    assert "torch" in message and "scipy" in message
    assert 'pip install "intelloptim[gp]"' in message


def test_only_the_missing_package_is_named() -> None:
    with (
        patch(_FIND_SPEC, _installed("torch")),
        pytest.raises(SamplerDependencyError) as info,
    ):
        resolve_sampler(GPSamplerSpec())
    assert "requires scipy, which is not installed" in str(info.value)


def test_qmc_without_scipy_points_to_the_qmc_extra() -> None:
    with (
        patch(_FIND_SPEC, _installed()),
        pytest.raises(SamplerDependencyError, match=r"intelloptim\[qmc\]"),
    ):
        resolve_sampler(QMCSamplerSpec())


def test_installed_dependencies_resolve_normally() -> None:
    with patch(_FIND_SPEC, _installed("torch", "scipy")):
        assert resolve_sampler(GPSamplerSpec()) is not None
        assert resolve_sampler(QMCSamplerSpec()) is not None


def _hardware(provider: dict) -> QuantumGausSamplerSpec:
    return QuantumGausSamplerSpec(
        provider=provider, shots=100, max_total_shots=10**7  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    "spec", [SimQuantumGausSamplerSpec(), SimQuantumMixtureSamplerSpec()]
)
def test_simulated_quantum_samplers_name_the_quantum_extra(spec: Any) -> None:
    with (
        patch(_FIND_SPEC, _installed()),
        pytest.raises(SamplerDependencyError, match=r"intelloptim\[quantum\]"),
    ):
        resolve_sampler(spec)


def test_the_ibm_provider_names_the_ibm_extra() -> None:
    spec = _hardware({"provider": "ibm_runtime"})
    with (
        patch(_FIND_SPEC, _installed("qiskit")),
        pytest.raises(SamplerDependencyError, match=r"intelloptim\[quantum-ibm\]"),
    ):
        resolve_sampler(spec)
    with patch(_FIND_SPEC, _installed("qiskit", "qiskit_ibm_runtime")):
        assert resolve_sampler(spec) is not None


def test_a_plugin_provider_brings_its_own_requirements() -> None:
    # No IBM SDK is needed (or demanded) to resolve a sampler for another provider.
    spec = _hardware({"provider": "aws_braket"})
    with patch(_FIND_SPEC, _installed()):
        assert resolve_sampler(spec) is not None


@pytest.mark.parametrize(
    "spec", [RandomSamplerSpec(), TPESamplerSpec(), SASamplerSpec()]
)
def test_samplers_without_optional_dependencies_are_not_checked(spec: Any) -> None:
    with patch(_FIND_SPEC, _installed()):
        assert resolve_sampler(spec) is not None


def test_error_is_an_import_error() -> None:
    assert issubclass(SamplerDependencyError, ImportError)


def _extras() -> Any:
    path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "pyproject.toml")
    with open(path, "rb") as handle:
        return tomllib.load(handle)["project"]["optional-dependencies"]


def _names(extras: Any, extra: str) -> Set[str]:
    """Package names an extra installs, following ``intelloptim[...]`` references."""
    out: Set[str] = set()
    for req in extras[extra]:
        if req.startswith("intelloptim["):
            for inner in req[len("intelloptim[") : req.index("]")].split(","):
                out |= _names(extras, inner.strip())
            continue
        for sep in ("<", ">", "=", "[", " "):
            req = req.split(sep)[0]
        out.add(req.replace("-", "_"))
    return out


_ALL_CHECKS = {
    **{k.value: v for k, v in _OPTIONAL_DEPENDENCIES.items()},
    **_PROVIDER_DEPENDENCIES,
}


@pytest.mark.parametrize("name", list(_ALL_CHECKS))
def test_each_check_is_backed_by_an_extra_in_pyproject(name: str) -> None:
    modules, extra = _ALL_CHECKS[name]
    extras = _extras()
    assert extra in extras
    assert set(modules) <= _names(extras, extra)


def test_samplers_extra_bundles_every_sampler_extra_except_hardware() -> None:
    extras = _extras()
    needed = {extra for _, extra in _ALL_CHECKS.values()}
    bundled = extras["samplers"][0]
    # Hardware access stays opt-in: it is not installed with the server image.
    assert all(extra in bundled for extra in needed - {"quantum-ibm"})
    assert "quantum-ibm" not in bundled
