"""The :class:`Client` — entry point of the intelloptim SDK.

A :class:`Client` is a thin wrapper around :class:`httpx.Client` that adds:

- authentication: every request carries the API token as a bearer token, which
  is looked up by :func:`~intelloptim.client.auth.resolve_token`;
- the translation of server error responses into the typed hierarchy of
  :mod:`intelloptim.client.errors`;
- :meth:`Client.create_study`, which declares a study and returns the
  :class:`~intelloptim.client.study.StudyHandle` used to ask for trials and
  report their results.

Typical usage::

    import intelloptim as intopt

    with intopt.Client(url="https://intelloptim.local") as client:
        study = client.create_study(...)

Tests can inject a pre-built ``httpx.Client`` so that they run against an
in-process app via ``httpx.ASGITransport`` or a mock transport, without
opening a real socket.
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
    IntelloptimAPIError,
    StudyLimitExceededError,
    StudyNotFoundError,
    StudyRebuildingError,
    StudyStoppedError,
    TrialNotFoundError,
)
from intelloptim.core.parameters import ParameterSpec
from intelloptim.core.pruners import NopPrunerSpec, PrunerSpec

_log = get_logger(__name__)

if TYPE_CHECKING:
    from intelloptim.client.study import StudyHandle
    from intelloptim.core.samplers import SamplerSpec

DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)


class Client:
    """REST client for an intelloptim server.

    The client owns an HTTP connection pool, so it is best used as a context
    manager, which closes the pool on exit::

        with Client(url="https://intelloptim.local") as client:
            ...

    Args:
        url (str): Base URL of the server, e.g. ``"https://intelloptim.local"``.
            Trailing slashes are stripped.
        token (Optional[str]): API bearer token. If ``None`` it is looked up in
            the ``INTELLOPTIM_TOKEN`` environment variable and then in
            ``~/.config/intelloptim/token``. Defaults to ``None``.
        timeout (Optional[httpx.Timeout]): Timeouts for the HTTP requests.
            Defaults to ``None``, which uses a 5 s connect and 30 s read
            timeout.
        verify (bool): Whether to verify the server's TLS certificate. Pass
            ``False`` only against self-signed certificates, for example in a
            local compose stack. Ignored when *http_client* is given. Defaults
            to ``True``.
        http_client (Optional[httpx.Client]): Pre-built HTTP client to use
            instead of creating one, typically wired to an in-process
            transport in tests. The caller keeps ownership: :meth:`close` does
            not close it. Defaults to ``None``.

    Raises:
        MissingTokenError: If no token is passed and none can be found.
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
        """Release the HTTP connection pool, unless it was passed in."""
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> "Client":
        """Return the client itself, for use in a ``with`` block."""
        return self

    def __exit__(self, *_exc: Any) -> None:
        """Close the client when the ``with`` block ends."""
        self.close()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @property
    def token(self) -> str:
        """The API token in use. Treat it as a secret: never log or print it."""
        return self._token

    @property
    def url(self) -> str:
        """The base URL of the server, without a trailing slash."""
        return self._url

    def create_study(
        self,
        study_name: str,
        direction: str,
        parameters: List[ParameterSpec],
        sampler_spec: "SamplerSpec",
        pruner_spec: Optional[PrunerSpec] = None,
    ) -> "StudyHandle":
        """Create a study, or join it if it already exists.

        The search space and the sampler and pruner settings are declared here,
        once; the objective then only reads the values the server sampled.
        The call is idempotent: the same arguments from any process that shares
        the same user or group identity yield the same ``study_id``, so several
        workers can call it independently to work on one study. The server is
        the sole authority on the identity, and no local check of it is made.

        The first call also fetches the caller's quota snapshot, which is
        cached on the client and used for the pre-flight checks of this method
        and of :meth:`~intelloptim.client.study.StudyHandle.optimize`. A
        warning is logged if the call budget is limited.

        Args:
            study_name (str): Human-readable label of the study. Two studies
                with the same name but a different configuration are different
                studies.
            direction (str): ``"minimize"`` or ``"maximize"``.
            parameters (List[ParameterSpec]): Declaration of the search space.
                The order is part of the study identity.
            sampler_spec (SamplerSpec): Declaration of the sampler the server
                runs.
            pruner_spec (Optional[PrunerSpec]): Declaration of the pruner.
                Defaults to ``None``, which means no pruning.

        Returns:
            StudyHandle: Handle to the study, whether it was just created or
            already existed.

        Raises:
            StudyLimitExceededError: If the caller has reached the cap on
                active studies. This is checked before the study is sent.
            IntelloptimAPIError: For any error response from the server, such
                as an invalid token.
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
                f"Delete an existing study to free a slot, or contact "
                f"your admin to upgrade your role."
            )

        request = StudyResolveRequest(
            study_name=study_name,
            direction=direction,
            parameters=list(parameters),
            sampler_spec=sampler_spec,
            pruner_spec=pruner_spec or NopPrunerSpec(),
        )
        # Compute config_digest locally (no token in this digest).
        params_dicts = [p.model_dump(mode="json") for p in request.parameters]
        local_digest = _config_digest(
            parameters=params_dicts,
            sampler=request.sampler_spec.model_dump(mode="json"),
            pruner=request.pruner_spec.model_dump(mode="json"),
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

    # ------------------------------------------------------------------
    # identity / quota (internal)
    # ------------------------------------------------------------------

    def _fetch_who_am_i(self) -> WhoAmIResponse:
        """Fetch the caller's identity and quota, caching the answer.

        Called lazily by the first :meth:`create_study`. A warning is logged
        when the caller works under a limited call budget, so that it is known
        before any trial is consumed.

        Returns:
            WhoAmIResponse: The identity and quota snapshot of the token.

        Raises:
            IntelloptimAPIError: For any error response from the server, such
                as an invalid token.
        """
        if self._who_am_i is not None:
            return self._who_am_i
        response = self._get("/auth/me")
        who = WhoAmIResponse.model_validate(response.json())
        self._who_am_i = who
        if who.is_group_context:
            if who.group_calls_remaining is not None:
                _log.warning(
                    f"Group '{who.group_name}': {who.group_calls_remaining} "
                    f"ask/tell call(s) remaining in the shared "
                    f"pool this month. Resets {who.period_resets_at or 'N/A'}.",
                )
            else:
                _log.info(
                    f"Operating in group context '{who.group_name}' "
                    f"(unlimited budget).",
                )
        elif who.calls_remaining is not None:
            _log.warning(
                f"Role '{who.role_name}': {who.calls_remaining} ask/tell "
                f"call(s) remaining this month. "
                f"Resets {who.period_resets_at or 'N/A'}. Contact your "
                f"admin to extend the quota.",
            )
        return who

    # ------------------------------------------------------------------
    # request plumbing (internal; used by StudyHandle)
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        """Return the headers that authenticate a request."""
        return {"Authorization": f"Bearer {self._token}"}

    def _post_json(self, path: str, body: Dict[str, Any]) -> httpx.Response:
        """POST *body* as JSON to *path* and return the successful response.

        Raises:
            IntelloptimError: The typed error for an error response; see
                :meth:`_raise_for_status`.
        """
        response = self._http.post(path, json=body, headers=self._headers())
        self._raise_for_status(response, path=path)
        return response

    def _get(self, path: str) -> httpx.Response:
        """GET *path* and return the successful response.

        Raises:
            IntelloptimError: The typed error for an error response; see
                :meth:`_raise_for_status`.
        """
        response = self._http.get(path, headers=self._headers())
        self._raise_for_status(response, path=path)
        return response

    @staticmethod
    def _detail(response: httpx.Response) -> Optional[str]:
        """Return the explanation in an error response, or its raw text."""
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
        """Raise the typed error that matches an HTTP error response.

        Does nothing for a successful response. Otherwise the status code
        selects the error: ``404`` is a trial error for a ``/tell`` request
        (its path contains ``/trials/``) and a study error otherwise, ``409``,
        ``410`` and ``503`` map to their dedicated errors, and anything else
        becomes :class:`~intelloptim.client.errors.IntelloptimAPIError`. See
        :mod:`intelloptim.client.errors` for the full mapping.

        Args:
            response (httpx.Response): The response to check.
            path (str): Path of the request, used to tell a missing trial from
                a missing study.

        Raises:
            IntelloptimError: The error that matches *response*.
        """
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
