"""Optional packages of Optuna-backed samplers — clear failure at resolve time."""

from __future__ import annotations

import os
from typing import Any, Set
from unittest.mock import patch

import pytest

from intelloptim.algorithms.errors import SamplerDependencyError
from intelloptim.algorithms.samplers import _OPTIONAL_DEPENDENCIES, resolve_sampler
from intelloptim.core.samplers import (
    GPSamplerSpec,
    QMCSamplerSpec,
    RandomSamplerSpec,
    SASamplerSpec,
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


@pytest.mark.parametrize(
    "spec", [RandomSamplerSpec(), TPESamplerSpec(), SASamplerSpec()]
)
def test_samplers_without_optional_dependencies_are_not_checked(spec: Any) -> None:
    with patch(_FIND_SPEC, _installed()):
        assert resolve_sampler(spec) is not None


def test_error_is_an_import_error() -> None:
    assert issubclass(SamplerDependencyError, ImportError)


def _extras() -> Any:
    tomllib = pytest.importorskip("tomllib")  # stdlib from Python 3.11
    path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "pyproject.toml")
    with open(path, "rb") as handle:
        return tomllib.load(handle)["project"]["optional-dependencies"]


def _names(requirements: Any) -> Set[str]:
    out: Set[str] = set()
    for req in requirements:
        for sep in ("<", ">", "=", "[", " "):
            req = req.split(sep)[0]
        out.add(req)
    return out


@pytest.mark.parametrize("kind", list(_OPTIONAL_DEPENDENCIES))
def test_each_check_is_backed_by_an_extra_in_pyproject(kind: Any) -> None:
    modules, extra = _OPTIONAL_DEPENDENCIES[kind]
    extras = _extras()
    assert extra in extras
    assert set(modules) <= _names(extras[extra])


def test_samplers_extra_bundles_every_sampler_extra() -> None:
    extras = _extras()
    needed = {extra for _, extra in _OPTIONAL_DEPENDENCIES.values()}
    bundled = extras["samplers"][0]
    assert all(extra in bundled for extra in needed)
