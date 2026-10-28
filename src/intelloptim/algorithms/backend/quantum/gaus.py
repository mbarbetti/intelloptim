"""Gaus quantum policy samplers: one bounded distribution per parameter.

The circuit's single-qubit ``Z`` means and the ``ZZ`` moments of neighbouring
qubits feed an affine head that gives the latent mean and standard deviation of
every numeric parameter; categorical parameters get their probabilities from a
register or a softmax head. Two samplers share the engine of
:class:`~intelloptim.algorithms.backend.quantum.commons.base.QuantumPolicySampler`:

- :class:`SimQuantumGausSampler` — circuits are simulated ideally, configured by
  :class:`~intelloptim.core.samplers.quantum.SimQuantumGausSamplerSpec`.
- :class:`QuantumGausSampler` — circuits run on real hardware, configured by
  :class:`~intelloptim.core.samplers.quantum.QuantumGausSamplerSpec`.
"""

from __future__ import annotations

from typing import ClassVar

from intelloptim.algorithms.backend.quantum.commons.base import QuantumPolicySampler


class SimQuantumGausSampler(QuantumPolicySampler):
    """Gaus quantum policy sampler on a simulator (no hardware noise).

    Trains on an exact statevector simulation, or on finite-shot samples of it
    when the spec sets ``shots``. See :class:`QuantumPolicySampler` for the
    arguments and for how a policy is stored, sampled and updated.
    """

    algorithm: ClassVar[str] = "gaus"
    hardware: ClassVar[bool] = False


class QuantumGausSampler(QuantumPolicySampler):
    """Gaus quantum policy sampler on real quantum hardware.

    Updates run on the provider named in the spec, within its shot budget, and
    only if the worker environment sets ``INTELLOPTIM_ALLOW_QUANTUM_HARDWARE=1``.
    Asks never touch the device. See :class:`QuantumPolicySampler` for the
    arguments.
    """

    algorithm: ClassVar[str] = "gaus"
    hardware: ClassVar[bool] = True


__all__ = ["QuantumGausSampler", "SimQuantumGausSampler"]
