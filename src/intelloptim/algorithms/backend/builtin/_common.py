"""Helpers shared by the built-in samplers.

All randomness in a built-in sampler is derived from a seed and the trial
number, never from state held by the sampler instance, because the
orchestrator builds a fresh sampler for every ask.
"""

from __future__ import annotations

import zlib
from typing import List, Optional

import numpy as np
from optuna.study import Study, StudyDirection
from optuna.trial import FrozenTrial, TrialState


def seed_key(seed: Optional[int], study: Study) -> int:
    """Return the integer that keys every RNG of a sampler for *study*."""
    if seed is not None:
        return seed % (1 << 64)
    return zlib.crc32(study.study_name.encode("utf-8"))


def make_rng(
    seed: Optional[int], study: Study, trial_number: int, *tags: int
) -> np.random.Generator:
    """Return a generator that depends only on the seed, trial and tags."""
    entropy = np.random.SeedSequence([seed_key(seed, study), trial_number, *tags])
    return np.random.default_rng(entropy)


def is_maximize(study: Study, name: str) -> bool:
    if len(study.directions) > 1:
        raise ValueError(f"{name} supports single-objective studies only")
    return bool(study.directions[0] == StudyDirection.MAXIMIZE)


def completed_trials(study: Study) -> List[FrozenTrial]:
    trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    return sorted(trials, key=lambda t: t.number)
