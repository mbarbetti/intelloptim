"""Evaluator loop behind ``intelloptim-worker evaluator``.

A worker imports the user's objective, creates the study it describes or joins
the existing one, and runs ``study.optimize()`` against the server. Several
workers started with the same spec, for example the tasks of a SLURM array job,
collaborate on one study without any wrapper code.

Typical invocation::

    intelloptim-worker evaluator \\
        --server https://intelloptim.example.org \\
        --import my_pkg.objectives:my_fn \\
        --n-trials 50

The study spec comes from the ``study_spec`` given to
``@intopt.objective("name", study_spec={...})``, or from a JSON file with a
``StudyResolveRequest`` payload passed with ``--spec-file``.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Callable, Dict, Optional, cast

from intelloptim._internal.logging import get_logger

_log = get_logger(__name__)


def _load_spec(
    spec_file: Optional[str],
    objective_fn: Callable[..., Any],
) -> Dict[str, Any]:
    """Return the study spec from *spec_file*, or else from the objective.

    Args:
        spec_file (Optional[str]): Path of a JSON file with the study spec, or
            ``None`` to use the spec attached to the objective.
        objective_fn (Callable[..., Any]): The imported objective; its spec is
            the one given with ``study_spec=`` to ``@intopt.objective``.

    Returns:
        Dict[str, Any]: The raw study spec.

    Raises:
        RuntimeError: If *spec_file* is ``None`` and the objective has no spec.
    """
    if spec_file is not None:
        with open(spec_file) as fh:
            return cast(Dict[str, Any], json.load(fh))

    from intelloptim.core.objective import get_study_spec_for

    spec = get_study_spec_for(objective_fn)
    if spec is not None:
        return spec

    name = getattr(objective_fn, "__qualname__", repr(objective_fn))
    raise RuntimeError(
        f"No study spec available for objective {name!r}. "
        "Either register it with study_spec={...} in @intopt.objective, "
        "or pass --spec-file <path/to/spec.json>."
    )


def _health_probe(base_url: str, token: str) -> None:
    """Check that the server answers ``GET /healthz``; exit if it does not.

    Args:
        base_url (str): Base URL of the server.
        token (str): API bearer token sent with the request.

    Raises:
        SystemExit: With status 1 if the server is unreachable or answers with
            an error status.
    """
    import httpx

    url = base_url.rstrip("/") + "/healthz"
    try:
        resp = httpx.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=10)
        resp.raise_for_status()
    except httpx.TransportError as exc:
        _log.error(
            "server health probe failed — is the server running?",
            extra={"url": url, "error": str(exc)},
        )
        sys.exit(1)
    except httpx.HTTPStatusError as exc:
        _log.error(
            "server health probe returned an error",
            extra={"url": url, "status": exc.response.status_code},
        )
        sys.exit(1)
    _log.info("server health probe passed", extra={"url": url})


def run_evaluator(
    server: str,
    import_spec: str,
    *,
    token: Optional[str] = None,
    spec_file: Optional[str] = None,
    n_trials: int = 50,
    wid: Optional[str] = None,
    site: Optional[str] = None,
    loglevel: str = "INFO",
    catch_exceptions: bool = True,
) -> None:
    """Run one evaluator worker: import the objective and optimise it.

    Probes the server, creates or joins the study described by the spec, then
    runs ``study.optimize()`` for *n_trials* trials. Every request carries the
    worker and site ids, so each trial can be attributed to the process and the
    site that evaluated it.

    Args:
        server (str): Base URL of the intelloptim server.
        import_spec (str): ``"module.path:callable_name"`` or
            ``"/path/to/file.py:callable_name"`` identifying the objective.
        token (Optional[str]): API bearer token. Falls back to
            ``INTELLOPTIM_TOKEN`` and ``~/.config/intelloptim/token``.
        spec_file (Optional[str]): Path of a JSON file with the study spec. If
            ``None``, the spec attached to the objective with
            ``@intopt.objective(..., study_spec=...)`` is used.
        n_trials (int): Number of trials this worker runs. Defaults to 50.
        wid (Optional[str]): Worker id. Derived from the SLURM environment, or
            from the host name and process id, if ``None``.
        site (Optional[str]): Site id. Derived from the SLURM environment, or
            from the host name, if ``None``.
        loglevel (str): Logging level name. Defaults to ``"INFO"``.
        catch_exceptions (bool): If ``True``, an exception raised by the
            objective marks only that trial as failed. If ``False``, it stops
            the run. Defaults to ``True``.

    Raises:
        SystemExit: With status 1 if the server health probe fails.
        RuntimeError: If no study spec is available.
        ImportSpecError: If the objective cannot be imported.
    """
    from intelloptim._internal.logging import configure_logging
    from intelloptim._internal.schemas import StudyResolveRequest
    from intelloptim.client.auth import resolve_token
    from intelloptim.worker.registry import import_callable
    from intelloptim.worker.slurm import site_id as _auto_site_id
    from intelloptim.worker.slurm import worker_id as _auto_worker_id

    configure_logging(level=loglevel)

    resolved_token = resolve_token(token)
    effective_wid = wid or _auto_worker_id()
    effective_site = site or _auto_site_id()

    _log.info(
        "evaluator starting",
        extra={
            "server": server,
            "import_spec": import_spec,
            "worker_id": effective_wid,
        },
    )

    # 1. Import the objective (triggers @intopt.objective registration).
    objective_fn: Callable[..., Any] = import_callable(import_spec)

    # 2. Health probe — fail fast before doing any real work.
    _health_probe(server, resolved_token)

    # 3. Load study spec and create/join the study.
    spec_dict = _load_spec(spec_file, objective_fn)
    spec = StudyResolveRequest.model_validate(spec_dict)

    import httpx

    from intelloptim.client.api import Client

    http_client = httpx.Client(
        base_url=server,
        headers={"X-Worker-Id": effective_wid, "X-Site": effective_site},
        timeout=httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0),
    )
    with Client(url=server, token=resolved_token, http_client=http_client) as client:
        study = client.create_study(
            study_name=spec.study_name,
            direction=spec.direction,
            parameters=list(spec.parameters),
            sampler_spec=spec.sampler_spec,
            pruner_spec=spec.pruner_spec,
        )
        _log.info(
            "study resolved",
            extra={
                "study_id": study.id,
                "existed": study.existed,
                "worker_id": effective_wid,
            },
        )

        # 4. Run the optimize loop.
        catch: Any = Exception if catch_exceptions else None
        study.optimize(
            objective_fn, n_trials=n_trials, catch=catch, worker_id=effective_wid
        )

        best = study.best_trial
        if best is not None:
            _log.info(
                "evaluate done",
                extra={
                    "best_trial": best.number,
                    "best_value": best.value,
                    "worker_id": effective_wid,
                },
            )
        else:
            _log.info(
                "evaluate done — no completed trials yet",
                extra={"worker_id": effective_wid},
            )


__all__ = ["run_evaluator"]
