"""The persisted state of a quantum policy: an immutable, versioned snapshot.

The learner (the sampler's ``rebuild``) publishes a new snapshot as one JSON
value in the study's user attributes; asks only read it. Each trial records the
snapshot version that generated it, so a later update knows which trials are
on-policy for which snapshot.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from intelloptim.algorithms.backend.quantum.commons.distributions import DistState
from intelloptim.algorithms.backend.quantum.commons.serialization import (
    array_from_json,
    array_to_json,
)

SNAPSHOT_KEY = "intopt:qnt_policy"
"""Study user attribute holding the current snapshot."""

BEHAVIOR_KEY = "intopt:qnt_policy"
"""Trial system attribute holding ``{version, mode, act, cats}`` of the draw."""

SNAPSHOT_FORMAT = 1


def shape_digest(shape: Dict[str, Any], space: Dict[str, Any]) -> str:
    """Hash what fixes the layout of the stored model."""
    payload = json.dumps({"shape": shape, "space": space}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class Snapshot:
    """One published version of the policy.

    Attributes:
        version (int): Starts at ``0`` and grows by one per update.
        names (List[str]): Parameter names, sorted.
        theta (np.ndarray): Circuit angles.
        phi (Dict[str, np.ndarray]): Classical head parameters.
        dist (DistState): The sampling distribution of this version, cached so
            that asks need no circuit.
        digest (str): Hash of the model layout; a mismatch means the study's
            sampler configuration or search space changed.
        adam_theta (Dict[str, Any]): Optimizer state of the angles.
        adam_phi (Dict[str, Any]): Optimizer state of the head.
        resources (Dict[str, int]): Cumulative circuits and shots spent.
        n_updates (int): Number of PPO updates applied so far.
    """

    version: int
    names: List[str]
    theta: np.ndarray
    phi: Dict[str, np.ndarray]
    dist: DistState
    digest: str
    adam_theta: Dict[str, Any] = field(default_factory=dict)
    adam_phi: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, int] = field(
        default_factory=lambda: {"circuits": 0, "shots": 0}
    )
    n_updates: int = 0

    def to_json(self) -> Dict[str, Any]:
        return {
            "format": SNAPSHOT_FORMAT,
            "version": self.version,
            "names": self.names,
            "theta": array_to_json(self.theta),
            "phi": {k: array_to_json(v) for k, v in self.phi.items()},
            "dist": self.dist.to_json(),
            "digest": self.digest,
            "adam_theta": self.adam_theta,
            "adam_phi": self.adam_phi,
            "resources": self.resources,
            "n_updates": self.n_updates,
        }

    @classmethod
    def from_json(cls, raw: Dict[str, Any]) -> "Snapshot":
        if raw.get("format") != SNAPSHOT_FORMAT:
            raise ValueError(f"unsupported snapshot format: {raw.get('format')!r}")
        return cls(
            version=int(raw["version"]),
            names=list(raw["names"]),
            theta=array_from_json(raw["theta"]),
            phi={k: array_from_json(v) for k, v in raw["phi"].items()},
            dist=DistState.from_json(raw["dist"]),
            digest=str(raw["digest"]),
            adam_theta=dict(raw.get("adam_theta", {})),
            adam_phi=dict(raw.get("adam_phi", {})),
            resources={k: int(v) for k, v in raw["resources"].items()},
            n_updates=int(raw.get("n_updates", 0)),
        )


__all__ = ["BEHAVIOR_KEY", "SNAPSHOT_KEY", "Snapshot", "shape_digest"]
