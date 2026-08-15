"""Shared fixtures for client unit tests (public package).

These tests exercise the client's HTTP/wire layer *without* a live server
— either via :class:`httpx.MockTransport` (see ``test_error_mapping``) or
pure client-side logic (``test_trial``, ``test_auth``). The real
client↔server round-trip lives in the private ``intelloptim-server`` repo's
integration suite, which can import both the client and the server.
"""

from __future__ import annotations

from typing import List

import pytest

import intelloptim as intopt


@pytest.fixture
def token() -> str:
    """A syntactically-plausible bearer token.

    The value is irrelevant to the mock-transport tests — it only needs to
    be a non-empty string the client will place in the Authorization
    header, so no server is required to mint a real one.
    """
    return "intopt_testtok_" + "a" * 32


@pytest.fixture
def simple_parameters() -> List[intopt.params.Parameter]:
    return [intopt.params.Float("x", -10.0, 10.0)]


@pytest.fixture
def simple_sampler() -> intopt.samplers.Sampler:
    return intopt.samplers.TPE(seed=42)
