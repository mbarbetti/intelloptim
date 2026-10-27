"""Runtime for intelloptim's own (non-Optuna) sampling strategies.

Each sampler lives in its own module:

- :class:`~intelloptim.algorithms.backend.builtin.sa.SASampler` — simulated
  annealing, configured by
  :class:`~intelloptim.core.samplers.builtin.SASamplerSpec`.
- :class:`~intelloptim.algorithms.backend.builtin.owg.OWGSampler` — a
  space-filling initial design followed by sampling from a Gaussian fitted with
  the objective values as weights, configured by
  :class:`~intelloptim.core.samplers.builtin.OWGSamplerSpec`. Its fit is an
  explicit, periodic ``rebuild`` step, which makes it a vehicle for exercising
  the rebuild plane.

The orchestrator is stateless and builds a fresh sampler for every ask, so
samplers in this package must not keep state between calls. They derive
everything they need from the study (its trial history, or a model persisted
in the study's user attributes) instead, which also makes their suggestions
reproducible.
"""

from intelloptim.algorithms.backend.builtin.owg import OWGSampler
from intelloptim.algorithms.backend.builtin.sa import SASampler

__all__ = ["OWGSampler", "SASampler"]
