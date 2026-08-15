"""HTTP-status → client-exception translation."""

from __future__ import annotations

from typing import Iterator

import httpx
import pytest

from intelloptim.client.api import Client
from intelloptim.client.errors import (
    ConfigDigestMismatchError,
    IntelloptimAPIError,
    IntelloptimError,
    StudyNotFoundError,
    StudyRebuildingError,
    StudyStoppedError,
    TrialNotFoundError,
)


@pytest.fixture
def mock_client(token) -> Iterator[Client]:
    """A client wired to a programmable per-test mock transport."""
    handler: dict = {}

    def dispatch(request: httpx.Request) -> httpx.Response:
        return handler["responder"](request)

    transport = httpx.MockTransport(dispatch)
    with httpx.Client(transport=transport, base_url="http://mock") as http:
        client = Client(url="http://mock", token=token, http_client=http)
        client.__test_handler = handler  # type: ignore[attr-defined]
        try:
            yield client
        finally:
            client.close()


def _set(mock_client: Client, responder) -> None:
    mock_client.__test_handler["responder"] = responder  # type: ignore[attr-defined]


def test_404_on_ask_maps_to_study_not_found(mock_client) -> None:
    _set(
        mock_client,
        lambda req: httpx.Response(404, json={"detail": "study sha256:x"}),
    )
    with pytest.raises(StudyNotFoundError):
        mock_client._post_json("/studies/sha256:x/ask", {"config_digest": "d"})


def test_404_on_tell_maps_to_trial_not_found(mock_client) -> None:
    _set(
        mock_client,
        lambda req: httpx.Response(404, json={"detail": "trial 9"}),
    )
    with pytest.raises(TrialNotFoundError):
        mock_client._post_json("/studies/sha256:x/trials/9/tell", {"x": 1})


def test_409_maps_to_config_digest_mismatch(mock_client) -> None:
    _set(
        mock_client,
        lambda req: httpx.Response(409, json={"detail": "digest mismatch"}),
    )
    with pytest.raises(ConfigDigestMismatchError):
        mock_client._post_json("/studies/sha256:x/ask", {})


def test_410_maps_to_study_stopped(mock_client) -> None:
    _set(
        mock_client,
        lambda req: httpx.Response(410, json={"detail": "stopped"}),
    )
    with pytest.raises(StudyStoppedError):
        mock_client._post_json("/studies/sha256:x/ask", {})


def test_503_maps_to_study_rebuilding_with_retry_after(mock_client) -> None:
    _set(
        mock_client,
        lambda req: httpx.Response(
            503,
            json={"detail": "rebuilding"},
            headers={"Retry-After": "5"},
        ),
    )
    with pytest.raises(StudyRebuildingError) as excinfo:
        mock_client._post_json("/studies/sha256:x/ask", {})
    assert excinfo.value.retry_after_seconds == 5.0


def test_other_4xx_falls_through_to_api_error(mock_client) -> None:
    _set(
        mock_client,
        lambda req: httpx.Response(400, json={"detail": "bad request"}),
    )
    with pytest.raises(IntelloptimAPIError) as excinfo:
        mock_client._post_json("/studies/sha256:x/ask", {})
    assert excinfo.value.status_code == 400


def test_401_unauthorized_falls_through_to_api_error(mock_client) -> None:
    _set(
        mock_client,
        lambda req: httpx.Response(401, json={"detail": "no token"}),
    )
    with pytest.raises(IntelloptimAPIError):
        mock_client._post_json("/studies/sha256:x/ask", {})


def test_error_hierarchy() -> None:
    """Every specific error inherits from :class:`IntelloptimError`."""
    for cls in (
        StudyNotFoundError,
        TrialNotFoundError,
        ConfigDigestMismatchError,
        StudyStoppedError,
        StudyRebuildingError,
        IntelloptimAPIError,
    ):
        assert issubclass(cls, IntelloptimError), cls.__name__
