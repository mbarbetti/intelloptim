"""Sampler declarations and plugin contract.

The user-facing surface (``intelloptim.samplers``) is the public re-export
of the concrete spec classes defined here. The plugin contract used by the
server's orchestrator and the optimizer worker lives in
:mod:`intelloptim.core.samplers.base`.
"""

from intelloptim.core.samplers.base import (
    RebuildPolicy,
    Sampler,
    SamplerKind,
    SamplerPlugin,
)

__all__ = ["RebuildPolicy", "Sampler", "SamplerKind", "SamplerPlugin"]
