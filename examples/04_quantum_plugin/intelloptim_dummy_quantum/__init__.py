"""Dummy quantum sampler plugin for intelloptim.

Reference implementation that demonstrates the entry-point plugin contract.
Both QuantumQAOA and QuantumVQE are resolved to optuna.samplers.RandomSampler
— this is not a real quantum implementation; it only validates that the
plugin wiring works end-to-end.
"""

from intelloptim_dummy_quantum.sampler import resolve_qaoa, resolve_vqe

__all__ = ["resolve_qaoa", "resolve_vqe"]
