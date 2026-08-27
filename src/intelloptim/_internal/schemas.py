"""Wire-protocol pydantic models shared across components.

These models define the over-the-wire shape of every request and response
in the intelloptim REST API. They are the **only** place wire models live;
duplicating them in ``client/`` or ``server/`` is a bug.

Changing a model here is a versioned event — bump ``API_VERSION`` and
update the corresponding consumers.
"""

from __future__ import annotations

from typing import Annotated, Any, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from intelloptim.core.parameters import Categorical, Float, Int
from intelloptim.core.pruners import PrunerSpec
from intelloptim.core.samplers import SamplerSpec
from intelloptim.core.trial import TrialState

API_VERSION = "v1"

ParameterSpec = Annotated[Union[Float, Int, Categorical], Field(discriminator="kind")]
"""Discriminated union of parameter declarations (tagged by ``kind``)."""


class StudyResolveRequest(BaseModel):
    """Body of ``POST /studies/resolve``."""

    model_config = ConfigDict(extra="forbid")

    name: str
    direction: str
    parameters: list[ParameterSpec]
    sampler: SamplerSpec
    pruner: PrunerSpec


class StudyResolveResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    study_id: str
    existed: bool
    n_trials: int = 0


class AskRequest(BaseModel):
    """Body of ``POST /studies/{study_id}/ask``."""

    model_config = ConfigDict(extra="forbid")

    config_digest: str
    worker_id: Optional[str] = None


class AskResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trial_id: str
    number: int
    params: dict[str, Any]


class TellRequest(BaseModel):
    """Body of ``POST /studies/{study_id}/trials/{trial_id}/tell``."""

    model_config = ConfigDict(extra="forbid")

    config_digest: str
    value: Optional[float] = None
    state: TrialState
    intermediate_values: dict[int, float] = Field(default_factory=dict)


class TellResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ack: bool = True


class EnqueueRequest(BaseModel):
    """Body of ``POST /studies/{study_id}/enqueue``.

    If ``value`` is provided, the trial is created in the
    :attr:`TrialState.COMPLETE` state immediately. If ``value`` is
    ``None``, the trial is queued in :attr:`TrialState.WAITING` and the
    next ``ask`` will return its parameters.
    """

    model_config = ConfigDict(extra="forbid")

    config_digest: str
    params: dict[str, Any]
    value: Optional[float] = None


class EnqueueResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trial_id: str
    state: TrialState


class WhoAmIResponse(BaseModel):
    """Response from ``GET /auth/me`` as seen by the SDK (bearer-token path).

    The server returns additional fields (``is_admin``, ``study_key``, etc.)
    that are ignored here; ``extra="ignore"`` prevents validation errors when
    the server adds new fields.

    ``calls_remaining`` and ``group_calls_remaining`` are ``None`` when the
    applicable budget is unlimited. ``period_resets_at`` is ``None`` when the
    quota is unlimited (no monthly rollover applies).
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
    "StudyResolveRequest",
    "StudyResolveResponse",
    "TellRequest",
    "TellResponse",
    "WhoAmIResponse",
]
