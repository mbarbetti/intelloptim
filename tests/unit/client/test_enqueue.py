"""``StudyHandle.enqueue`` — the trial number comes from the response."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx

from intelloptim.client.api import Client
from intelloptim.client.study import StudyHandle


def _handle(client: Client) -> StudyHandle:
    return StudyHandle(
        client=client,
        study_id="suid:abc",
        existed=False,
        n_trials=0,
        config_digest="digest-1",
    )


def test_enqueue_uses_the_number_of_the_response(token: str) -> None:
    sent: List[Dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        # An id that is not a number: the SDK must not parse it.
        return httpx.Response(
            201, json={"trial_id": "opaque-id", "number": 4, "state": "waiting"}
        )

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://m")
    with Client(url="http://m", token=token, http_client=http) as client:
        trial = _handle(client).enqueue({"x": 0.5})
    assert trial.trial_id == "opaque-id"
    assert trial.number == 4
    assert trial.params == {"x": 0.5}
    assert sent[0]["config_digest"] == "digest-1"


def test_opaque_identifiers_are_encoded_as_single_path_segments(token: str) -> None:
    paths: List[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.raw_path)
        return httpx.Response(
            201, json={"trial_id": "opaque-id", "number": 4, "state": "waiting"}
        )

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://m")
    with Client(url="http://m", token=token, http_client=http) as client:
        handle = StudyHandle(
            client=client,
            study_id="study/id?x",
            existed=False,
            n_trials=0,
            config_digest="digest-1",
        )
        handle.enqueue({"x": 0.5})
    assert paths == [b"/studies/study%2Fid%3Fx/enqueue"]
