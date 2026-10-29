"""Request and response models of the intelloptim REST API.

Every body exchanged between the SDK and the server is defined here: the client
validates what it sends and receives against these models, and the server uses
the same classes. Models reject unknown fields, so a client and a server that
disagree on the protocol fail with a clear error instead of silently dropping
data; :class:`WhoAmIResponse` is the one tolerant exception.

The models are grouped by endpoint: study resolution, ``ask``/``tell``,
``should_prune``, ``enqueue`` and authentication. :class:`TrialState` is shared
by several of them.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from intelloptim.core.parameters import ParameterSpec
from intelloptim.core.pruners import PrunerSpec
from intelloptim.core.samplers import SamplerSpec

API_VERSION = "v1"
"""Version of the wire protocol, bumped on any breaking change to a model."""


class TrialState(str, Enum):
    """Lifecycle state of a trial.

    The string values are part of the wire protocol: the client sends one in
    every ``tell`` request and the server returns one in every ``enqueue``
    response. Only ``COMPLETE``, ``PRUNED`` and ``FAILED`` are valid in a
    ``tell``. Users reach this enum as ``intelloptim.client.TrialState``.
    """

    WAITING = "waiting"
    """Created (e.g. via enqueue) but not yet handed to an evaluator."""

    RUNNING = "running"
    """Handed to an evaluator; awaiting the matching ``tell``."""

    COMPLETE = "complete"
    """Evaluated successfully; ``value`` is set."""

    PRUNED = "pruned"
    """Stopped early by the pruner; ``value`` may be ``None``."""

    FAILED = "failed"
    """The evaluation raised; ``value`` is ``None``."""


class StudyResolveRequest(BaseModel):
    """Body of ``POST /studies/resolve``.

    The server hashes these fields, together with the caller's study key, into
    the study id, so the same request always resolves to the same study.
    Resolving is idempotent: the first call creates the study, later calls
    (from this or another client) return it.

    Attributes:
        study_name (str): Human-readable name; part of the study identity.
        direction (str): ``"minimize"`` or ``"maximize"``.
        parameters (List[ParameterSpec]): The search space, declared once here.
        sampler_spec (SamplerSpec): Sampler kind and its settings.
        pruner_spec (PrunerSpec): Pruner kind and its settings.
    """

    model_config = ConfigDict(extra="forbid")

    study_name: str
    direction: str
    parameters: List[ParameterSpec]
    sampler_spec: SamplerSpec
    pruner_spec: PrunerSpec


class StudyResolveResponse(BaseModel):
    """Response of ``POST /studies/resolve``.

    Attributes:
        study_id (str): Content-addressed id used in every later study URL; the
            server is authoritative, the client never recomputes it.
        existed (bool): ``True`` if the study was already there, ``False`` if
            this call created it.
        n_trials (int): Trials the study held at resolve time.
    """

    model_config = ConfigDict(extra="forbid")

    study_id: str
    existed: bool
    n_trials: int = 0


class AskRequest(BaseModel):
    """Body of ``POST /studies/{study_id}/ask``.

    Attributes:
        config_digest (str): Digest of the study configuration; the server
            answers 409 if it differs from the stored one.
        worker_id (Optional[str]): Free-text label of the caller, recorded on
            the trial to attribute work per worker.
    """

    model_config = ConfigDict(extra="forbid")

    config_digest: str
    worker_id: Optional[str] = None


class AskResponse(BaseModel):
    """Response of ``POST /studies/{study_id}/ask``.

    Attributes:
        trial_id (str): Opaque handle of the trial, to be passed back unchanged
            in the URLs of ``tell`` and ``should_prune``. Clients must not parse
            it: the server may change its format.
        number (int): Zero-based position of the trial within the study, for
            display and ordering.
        params (Dict[str, Any]): Sampled value of every declared parameter,
            keyed by name.
    """

    model_config = ConfigDict(extra="forbid")

    trial_id: str
    number: int
    params: Dict[str, Any]


class TellRequest(BaseModel):
    """Body of ``POST /studies/{study_id}/trials/{trial_id}/tell``.

    Attributes:
        config_digest (str): Digest of the study configuration (see
            :class:`AskRequest`).
        value (Optional[float]): Objective value. Required when ``state`` is
            ``COMPLETE``; optional for ``PRUNED`` and ``FAILED``.
        state (TrialState): Final state of the trial.
    """

    model_config = ConfigDict(extra="forbid")

    config_digest: str
    value: Optional[float] = None
    state: TrialState


class TellResponse(BaseModel):
    """Response of ``POST /studies/{study_id}/trials/{trial_id}/tell``.

    Attributes:
        ack (bool): Always ``True``; failures are reported through HTTP status
            codes, not through this field.
    """

    model_config = ConfigDict(extra="forbid")

    ack: bool = True


class ShouldPruneRequest(BaseModel):
    """Body of ``POST /studies/{study_id}/trials/{trial_id}/should_prune``.

    The client sends only the values the server has not seen yet. The server
    records them and asks the study's pruner whether the trial should stop. The
    call does not count against the quota.

    Attributes:
        config_digest (str): Digest of the study configuration (see
            :class:`AskRequest`).
        intermediate_values (Dict[int, float]): Value reported at each step,
            keyed by step number. May be empty to ask again without new data.
    """

    model_config = ConfigDict(extra="forbid")

    config_digest: str
    intermediate_values: Dict[int, float] = Field(default_factory=dict)


class ShouldPruneResponse(BaseModel):
    """Response of ``POST /studies/{study_id}/trials/{trial_id}/should_prune``.

    Attributes:
        should_prune (bool): ``True`` if the pruner wants the trial stopped;
            the objective should then raise
            ``intelloptim.client.errors.TrialPruned``.
    """

    model_config = ConfigDict(extra="forbid")

    should_prune: bool


class EnqueueRequest(BaseModel):
    """Body of ``POST /studies/{study_id}/enqueue``.

    Registers a trial with fixed parameters, either already evaluated or to be
    evaluated later. If ``value`` is given, the trial is created in
    :attr:`TrialState.COMPLETE` at once, which lets a study be seeded with
    known results. Otherwise it is queued in :attr:`TrialState.WAITING` and the
    next ``ask`` returns its parameters instead of sampling new ones.

    Attributes:
        config_digest (str): Digest of the study configuration (see
            :class:`AskRequest`).
        params (Dict[str, Any]): Value of every declared parameter, keyed by
            name.
        value (Optional[float]): Objective value of an already evaluated trial,
            or ``None`` to queue the trial.
    """

    model_config = ConfigDict(extra="forbid")

    config_digest: str
    params: Dict[str, Any]
    value: Optional[float] = None


class EnqueueResponse(BaseModel):
    """Response of ``POST /studies/{study_id}/enqueue``.

    Attributes:
        trial_id (str): Opaque handle of the created trial (see
            :class:`AskResponse`).
        number (int): Zero-based position of the created trial within the study.
        state (TrialState): ``COMPLETE`` if a value was given, ``WAITING`` if
            the trial was queued.
    """

    model_config = ConfigDict(extra="forbid")

    trial_id: str
    number: int
    state: TrialState


class WhoAmIResponse(BaseModel):
    """Response of ``GET /auth/me``, as the SDK needs it for pre-flight checks.

    The server returns more fields (``is_admin``, ``study_key``, ...) than the
    SDK uses. They are ignored on purpose (``extra="ignore"``), so a server that
    adds fields does not break older clients.

    Attributes:
        id (str): User id.
        email (str): User email.
        is_active (bool): Whether the account may use the service.
        role_name (Optional[str]): Name of the user's role, which sets the
            quotas.
        calls_remaining (Optional[int]): Ask/tell calls left in the current
            period; ``None`` if the budget is unlimited.
        active_studies (int): Studies the user currently has active.
        max_active_studies (Optional[int]): Cap on active studies; ``None`` if
            unlimited.
        period_resets_at (Optional[str]): When the quota resets; ``None`` if
            the quota is unlimited and no rollover applies.
        is_group_context (bool): ``True`` if the token acts on behalf of a
            group.
        group_name (Optional[str]): Name of that group, if any.
        group_calls_remaining (Optional[int]): Calls left in the group budget;
            ``None`` if unlimited.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    email: str
    is_active: bool
    role_name: Optional[str] = None
    calls_remaining: Optional[int] = None
    active_studies: int = 0
    max_active_studies: Optional[int] = None
    period_resets_at: Optional[str] = None
    is_group_context: bool = False
    group_name: Optional[str] = None
    group_calls_remaining: Optional[int] = None


__all__ = [
    "API_VERSION",
    "AskRequest",
    "AskResponse",
    "EnqueueRequest",
    "EnqueueResponse",
    "ParameterSpec",
    "PrunerSpec",
    "SamplerSpec",
    "ShouldPruneRequest",
    "ShouldPruneResponse",
    "StudyResolveRequest",
    "StudyResolveResponse",
    "TellRequest",
    "TellResponse",
    "TrialState",
    "WhoAmIResponse",
]
