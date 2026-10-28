"""Mixture quantum policy samplers: a mode register selects the component.

The measurement probabilities of the mode register are the weights of
``2 ** n_mode_qubits`` components. In each component the geometry register's
``Z`` and ``ZZ`` moments, conditioned on the mode, give the latent mean and
standard deviation of every numeric parameter through a shared affine head;
categorical parameters are encoded as in the Gaus samplers. Two samplers share
the engine of
:class:`~intelloptim.algorithms.backend.quantum.commons.base.QuantumPolicySampler`:

- :class:`SimQuantumMixtureSampler` — ideal simulation, configured by
  :class:`~intelloptim.core.samplers.quantum.SimQuantumMixtureSamplerSpec`.
- :class:`QuantumMixtureSampler` — real hardware, configured by
  :class:`~intelloptim.core.samplers.quantum.QuantumMixtureSamplerSpec`.
"""

from __future__ import annotations

from typing import ClassVar

from intelloptim.algorithms.backend.quantum.commons.base import QuantumPolicySampler


class SimQuantumMixtureSampler(QuantumPolicySampler):
    """Mixture quantum policy sampler on a simulator (no hardware noise).

    Trains on an exact statevector simulation, or on finite-shot samples of it
    when the spec sets ``shots``. See :class:`QuantumPolicySampler` for the
    arguments and for how a policy is stored, sampled and updated.
    """

    algorithm: ClassVar[str] = "mixture"
    hardware: ClassVar[bool] = False


class QuantumMixtureSampler(QuantumPolicySampler):
    """Mixture quantum policy sampler on real quantum hardware.

    Updates run on the provider named in the spec, within its shot budget, and
    only if the worker environment sets ``INTELLOPTIM_ALLOW_QUANTUM_HARDWARE=1``.
    Asks never touch the device. See :class:`QuantumPolicySampler` for the
    arguments.
    """

    algorithm: ClassVar[str] = "mixture"
    hardware: ClassVar[bool] = True


__all__ = ["QuantumMixtureSampler", "SimQuantumMixtureSampler"]
