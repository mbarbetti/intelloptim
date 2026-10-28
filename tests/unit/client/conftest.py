"""Shared fixtures for client unit tests (public package).

These tests exercise the client's HTTP/wire layer *without* a live server
— either via :class:`httpx.MockTransport` (see ``test_error_mapping``) or
pure client-side logic (``test_trial``, ``test_auth``). The real
client↔server round-trip lives in the private ``intelloptim-server`` repo's
integration suite, which can import both the client and the server.
"""

from __future__ import annotations

import logging
from typing import List

import pytest

import intelloptim as intopt
from intelloptim.core.parameters import ParameterSpec
from intelloptim.core.samplers import SamplerSpec


@pytest.fixture
def token() -> str:
    """A syntactically-plausible bearer token.

    The value is irrelevant to the mock-transport tests — it only needs to
    be a non-empty string the client will place in the Authorization
    header, so no server is required to mint a real one.
    """
    return "intopt_testtok_" + "a" * 32


@pytest.fixture
def simple_parameters() -> List[ParameterSpec]:
    return [intopt.params.Float("x", -10.0, 10.0)]


@pytest.fixture
def simple_sampler() -> SamplerSpec:
    return intopt.samplers.TPESamplerSpec(seed=42)


@pytest.fixture
def log_records(caplog):
    """Capture records of the ``intelloptim`` logger.

    ``configure_logging`` turns off propagation on that logger, so pytest's
    root-level handler would not see them; attach it directly instead.
    """
    package_logger = logging.getLogger("intelloptim")
    package_logger.addHandler(caplog.handler)
    previous = package_logger.level
    package_logger.setLevel(logging.WARNING)
    try:
        yield caplog
    finally:
        package_logger.removeHandler(caplog.handler)
        package_logger.setLevel(previous)
