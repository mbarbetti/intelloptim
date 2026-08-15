"""The :class:`Client` — entry point of the intelloptim SDK.

A :class:`Client` is a thin wrapper around :class:`httpx.Client` that
adds:

* canonical request signing (Bearer token + ``study_id`` / ``config_digest``
  bookkeeping),
* translation of server HTTP errors into the typed
  :mod:`intelloptim.client.errors` hierarchy,
* a constructor for :class:`StudyHandle` instances, which provide the
  Optuna-shaped ``ask`` / ``tell`` / ``optimize`` surface.

Tests inject a pre-built ``httpx.Client`` so they can run against an
in-process FastAPI app via ``httpx.ASGITransport`` without bringing up a
real socket.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import httpx

from intelloptim._internal.hashing import study_identity
from intelloptim._internal.schemas import (
    StudyResolveRequest,
    StudyResolveResponse,
)
from intelloptim.client.auth import resolve_token
from intelloptim.client.errors import (
    ConfigDigestMismatchError,
    IntelloptimAPIError,
    StudyNotFoundError,
    StudyRebuildingError,
    StudyStoppedError,
    TrialNotFoundError,
)
from intelloptim.core.parameters import Parameter
from intelloptim.core.pruners import NopPruner, Pruner

if TYPE_CHECKING:
    from intelloptim.client.study import StudyHandle
    from intelloptim.core.samplers.classical import TPE
    from intelloptim.core.samplers.quantum._stubs import (
        QuantumQAOA,
        QuantumVQE,
    )
    from intelloptim.core.samplers.remote import RemoteRandom

    SamplerSpecType = TPE | RemoteRandom | QuantumQAOA | QuantumVQE

DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)


class Client:
    """REST client for an intelloptim server.

    Args:
        url: Base URL of the server (e.g. ``"https://intelloptim.local"``).
            Trailing slashes are stripped.
        token: API bearer token. Falls back to ``INTELLOPTIM_TOKEN`` /
            ``~/.config/intelloptim/token`` per :func:`resolve_token`.
        timeout: Optional :class:`httpx.Timeout` override.
        verify: SSL certificate verification. Pass ``False`` to disable
            (useful against self-signed certs in local compose stacks).
            Defaults to ``True``.
        http_client: Optional pre-built :class:`httpx.Client`. Tests
            inject one configured with :class:`httpx.ASGITransport` so
            requests never leave the process.
    """

    def __init__(
        self,
        url: str,
        token: Optional[str] = None,
        *,
        timeout: Optional[httpx.Timeout] = None,
        verify: bool = True,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self._url = url.rstrip("/")
        self._token = resolve_token(token)
        if http_client is not None:
            self._http = http_client
            self._owns_http = False
        else:
            self._http = httpx.Client(
                base_url=self._url, timeout=timeout or DEFAULT_TIMEOUT, verify=verify
            )
            self._owns_http = True

    # ------------------------------------------------------------------
    # context-manager support
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the underlying HTTP connection pool, if we own it."""
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @property
    def token(self) -> str:
        return self._token

    @property
    def url(self) -> str:
        return self._url

    def create_study(
        self,
        name: str,
        direction: str,
        sampler: "SamplerSpecType",
        parameters: List[Parameter],
        pruner: Optional[Pruner] = None,
    ) -> "StudyHandle":
        """Resolve a study spec to a :class:`StudyHandle`.

        Idempotent — calling :func:`create_study` with the same args from
        any process yields the same ``study_id`` (content-addressed).
        """
        from intelloptim.client.study import StudyHandle  # late: cycles

        request = StudyResolveRequest(
            name=name,
            direction=direction,
            parameters=list(parameters),
            sampler=sampler,
            pruner=pruner or NopPruner(),
        )
        local_study_id, local_digest = study_identity(self._token, request)
        response = self._post_json("/studies/resolve", request.model_dump(mode="json"))
        result = StudyResolveResponse.model_validate(response.json())
        if result.study_id != local_study_id:
            # Server disagrees about the hash — almost certainly a
            # canonicalization bug on one side. Fail loudly rather than
            # carrying on with a mystery study.
            raise IntelloptimAPIError(
                response.status_code,
                detail=(
                    f"server returned study_id {result.study_id!r}, "
                    f"client computed {local_study_id!r}"
                ),
            )
        return StudyHandle(
            client=self,
            study_id=result.study_id,
            existed=result.existed,
            n_trials=result.n_trials,
            config_digest=local_digest,
        )

    # Optuna-flavoured alias.
    study = create_study

    # ------------------------------------------------------------------
    # request plumbing (internal; used by StudyHandle)
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}

    def _post_json(self, path: str, body: Dict[str, Any]) -> httpx.Response:
        response = self._http.post(path, json=body, headers=self._headers())
        self._raise_for_status(response, path=path)
        return response

    def _get(self, path: str) -> httpx.Response:
        response = self._http.get(path, headers=self._headers())
        self._raise_for_status(response, path=path)
        return response

    @staticmethod
    def _detail(response: httpx.Response) -> Optional[str]:
        try:
            body = response.json()
        except Exception:  # pragma: no cover - non-JSON error body
            return response.text or None
        if isinstance(body, dict):
            detail = body.get("detail")
            if isinstance(detail, str):
                return detail
        return response.text or None

    def _raise_for_status(self, response: httpx.Response, *, path: str) -> None:
        if not response.is_error:
            return
        detail = self._detail(response)
        code = response.status_code

        if code == 404:
            # ``/tell`` endpoint is the only place 404 means "trial",
            # everything else means "study".
            if "/trials/" in path:
                raise TrialNotFoundError(detail or "trial not found")
            raise StudyNotFoundError(detail or "study not found")
        if code == 409:
            raise ConfigDigestMismatchError(
                detail or "config_digest does not match stored study"
            )
        if code == 410:
            raise StudyStoppedError(detail or "study is stopped")
        if code == 503:
            retry_after_raw = response.headers.get("retry-after")
            retry_after: Optional[float]
            try:
                retry_after = (
                    float(retry_after_raw) if retry_after_raw is not None else None
                )
            except ValueError:
                retry_after = None
            raise StudyRebuildingError(
                detail or "study is rebuilding", retry_after_seconds=retry_after
            )

        raise IntelloptimAPIError(code, detail)


__all__ = ["Client", "DEFAULT_TIMEOUT"]
