"""``StudyHandle.optimize(worker_id=...)`` — attribution sent with every ask."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import httpx

from intelloptim.client.api import Client
from intelloptim.client.study import StudyHandle


def _handle(client: Client) -> StudyHandle:
    return StudyHandle(
        client=client, study_id="suid:abc", existed=False, n_trials=0,
        config_digest="digest-1",
    )  # fmt: skip


def _optimize(token: str, worker_id: Optional[str], n: int = 2) -> List[Dict[str, Any]]:
    asks: List[Dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/ask"):
            asks.append(json.loads(request.content))
            number = len(asks) - 1
            return httpx.Response(
                200, json={"trial_id": str(number), "number": number, "params": {}}
            )
        return httpx.Response(200, json={"ack": True})

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://m")
    with Client(url="http://m", token=token, http_client=http) as client:
        kwargs: Dict[str, Any] = {} if worker_id is None else {"worker_id": worker_id}
        _handle(client).optimize(lambda trial: 1.0, n_trials=n, **kwargs)
    return asks


def test_worker_id_is_sent_with_every_ask(token: str) -> None:
    asks = _optimize(token, "w-1")
    assert [a["worker_id"] for a in asks] == ["w-1", "w-1"]
    assert all(a["config_digest"] == "digest-1" for a in asks)


def test_worker_id_defaults_to_none(token: str) -> None:
    assert [a["worker_id"] for a in _optimize(token, None)] == [None, None]
