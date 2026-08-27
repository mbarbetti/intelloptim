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

from intelloptim._internal.hashing import config_digest as _config_digest
from intelloptim._internal.logging import get_logger
from intelloptim._internal.schemas import (
    StudyResolveRequest,
    StudyResolveResponse,
    WhoAmIResponse,
)
from intelloptim.client.auth import resolve_token
from intelloptim.client.errors import (
    ConfigDigestMismatchError,
    InsufficientQuotaError,
    IntelloptimAPIError,
    StudyLimitExceededError,
    StudyNotFoundError,
    StudyRebuildingError,
    StudyStoppedError,
    TrialNotFoundError,
)
from intelloptim.core.parameters import Parameter
from intelloptim.core.pruners import NopPrunerSpec, PrunerSpec

_log = get_logger(__name__)

if TYPE_CHECKING:
    from intelloptim.client.study import StudyHandle
    from intelloptim.core.samplers import (
        TPESamplerSpec,
        SASamplerSpec,
        QuantumQAOA,
        QuantumVQE,
    )

    SamplerSpecType = TPESamplerSpec | SASamplerSpec | QuantumQAOA | QuantumVQE

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
        self._who_am_i: Optional[WhoAmIResponse] = None
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
        pruner: Optional[PrunerSpec] = None,
    ) -> "StudyHandle":
        """Resolve a study spec to a :class:`StudyHandle`.

        Idempotent — calling :func:`create_study` with the same args from
        any process that shares the same user or group identity yields the
        same ``study_id``. The server is the sole authority on study
        identity; no local hash agreement check is performed.

        Raises:
            StudyLimitExceededError: The caller has reached their
                active-study cap (pre-flight check; no server call made).
        """
        from intelloptim.client.study import StudyHandle  # late: cycles

        who = self._fetch_who_am_i()
        if (
            who.max_active_studies is not None
            and who.active_studies >= who.max_active_studies
        ):
            raise StudyLimitExceededError(
                f"Active study limit reached ({who.active_studies}/"
                f"{who.max_active_studies}). "
                "Delete an existing study to free a slot, or contact your "
                "admin to upgrade your role."
            )

        request = StudyResolveRequest(
            name=name,
            direction=direction,
            parameters=list(parameters),
            sampler=sampler,
            pruner=pruner or NopPrunerSpec(),
        )
        # Compute config_digest locally (no token in this digest).
        params_dicts = [p.model_dump(mode="json") for p in request.parameters]
        local_digest = _config_digest(
            parameters=params_dicts,
            sampler=request.sampler.model_dump(mode="json"),
            pruner=request.pruner.model_dump(mode="json"),
            direction=request.direction,
        )
        response = self._post_json("/studies/resolve", request.model_dump(mode="json"))
        result = StudyResolveResponse.model_validate(response.json())
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
    # identity / quota (internal)
    # ------------------------------------------------------------------

    def _fetch_who_am_i(self) -> WhoAmIResponse:
        """Fetch and cache the caller's identity and quota snapshot.

        Called lazily on the first :meth:`create_study`. Logs a WARNING
        when the caller is operating under a limited budget so they are
        aware before any trials are consumed.
        """
        if self._who_am_i is not None:
            return self._who_am_i
        response = self._get("/auth/me")
        who = WhoAmIResponse.model_validate(response.json())
        self._who_am_i = who
        if who.is_group_context:
            if who.group_calls_remaining is not None:
                _log.warning(
                    "Group '%s': %d ask/tell call(s) remaining in the shared "
                    "pool this month. Resets %s.",
                    who.group_name,
                    who.group_calls_remaining,
                    who.period_resets_at or "N/A",
                )
            else:
                _log.info(
                    "Operating in group context '%s' (unlimited budget).",
                    who.group_name,
                )
        elif who.calls_remaining is not None:
            _log.warning(
                "Role '%s': %d ask/tell call(s) remaining this month. "
                "Resets %s. Contact your admin to extend the quota.",
                who.role_name,
                who.calls_remaining,
                who.period_resets_at or "N/A",
            )
        return who

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
