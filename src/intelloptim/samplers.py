"""Public re-exports for sampler declarations.

Classical and quantum samplers share the same construction surface — only
the implementation backend differs::

    import intelloptim as intopt

    sampler = intopt.samplers.TPE(seed=42)             # classical, ships in v1
    sampler = intopt.samplers.QuantumQAOA(reps=3)      # quantum, requires an
                                                     # external plugin (raises
                                                     # NotImplementedError until
                                                     # one is registered).

See ``docs/samplers/quantum_plugin.md`` for the plugin contract.
"""

from intelloptim.core.samplers import Sampler, SamplerKind
from intelloptim.core.samplers.classical import TPE
from intelloptim.core.samplers.quantum import (
    QuantumQAOA,
    QuantumSampler,
    QuantumVQE,
)
from intelloptim.core.samplers.remote import RemoteRandom

__all__ = [
    "Sampler",
    "SamplerKind",
    "TPE",
    "RemoteRandom",
    "QuantumQAOA",
    "QuantumSampler",
    "QuantumVQE",
]
