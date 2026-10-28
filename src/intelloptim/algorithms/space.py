"""The search space a study declares, stored on the study itself.

A sampler that draws all parameters of a trial jointly needs the whole search
space before the first trial exists, which Optuna cannot provide on its own: it
only infers a space from completed trials. The server therefore stores the
declared space in the study's user attributes when it creates the study, and
such samplers read it back from there.
"""

from __future__ import annotations

from typing import Dict, Optional

from optuna.distributions import (
    BaseDistribution,
    distribution_to_json,
    json_to_distribution,
)
from optuna.study import Study

SPACE_KEY = "intopt:space"


def store_declared_space(study: Study, space: Dict[str, BaseDistribution]) -> None:
    """Record *space* as the declared search space of *study*.

    Args:
        study (Study): The study the space belongs to.
        space (Dict[str, BaseDistribution]): Parameter name to distribution.
    """
    study.set_user_attr(
        SPACE_KEY, {name: distribution_to_json(d) for name, d in space.items()}
    )


def declared_space(study: Study) -> Optional[Dict[str, BaseDistribution]]:
    """Return the search space stored on *study*, if any.

    Args:
        study (Study): The study to read.

    Returns:
        Optional[Dict[str, BaseDistribution]]: The declared space, in the order
        it was stored, or ``None`` for a study that has none (for instance one
        that was created before the space was recorded).
    """
    stored = study.user_attrs.get(SPACE_KEY)
    if not isinstance(stored, dict) or not stored:
        return None
    return {name: json_to_distribution(raw) for name, raw in stored.items()}


__all__ = ["SPACE_KEY", "declared_space", "store_declared_space"]
