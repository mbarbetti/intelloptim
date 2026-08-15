"""Classical sampler specs.

These declarative classes mirror Optuna's classical samplers. Construction
is pure-Python (no Optuna import); the actual ``optuna.samplers.*``
instance is built by the server's orchestrator at study-resolution time
from the spec's fields.

v1 ships the TPE adapter only. Random, Grid, CMA-ES, and NSGA-II land in
later phases.
"""

from __future__ import annotations

from typing import ClassVar, Literal, Optional

from intelloptim.core.samplers.base import Sampler, SamplerKind


class TPE(Sampler):
    """Tree-structured Parzen Estimator sampler (default).

    Reference: Bergstra et al., *Algorithms for Hyper-Parameter Optimization*,
    NeurIPS 2011. Adapter for ``optuna.samplers.TPESampler``.

    Args:
        seed: Random seed for reproducibility. ``None`` lets Optuna choose.
        n_startup_trials: Number of random trials before TPE takes over.
        n_ei_candidates: Number of candidate samples drawn from the
            "good" KDE per ``ask`` (Optuna default is 24).
        multivariate: When ``True``, model parameter dependencies via a
            multivariate KDE (Optuna's "multivariate TPE").
    """

    kind: Literal[SamplerKind.TPE] = SamplerKind.TPE

    seed: Optional[int] = None
    n_startup_trials: int = 10
    n_ei_candidates: int = 24
    multivariate: bool = False

    runs_remote: ClassVar[bool] = False


__all__ = ["TPE"]
