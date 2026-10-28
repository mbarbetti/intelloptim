"""Specs of the quantum policy samplers.

A quantum policy sampler proposes candidates from a probability distribution
whose parameters are produced by a parameterised quantum circuit, and trains
the circuit and a small classical head with PPO on the objective values of
completed trials. Two algorithms exist, each as a simulated and a hardware
sampler:

- **Gaus** (``quantum_gaus``) — one bounded Gaussian policy.
- **Mixture** (``quantum_mixture``) — a mixture whose weights are the
  measurement probabilities of a mode register, with geometry conditioned on
  the mode.

The names say which is which: a ``Sim`` prefix marks the simulated sampler. The
simulated samplers are ideal: the circuit is simulated exactly, or with
finite-shot sampling noise only. The hardware samplers run on a real device and
see its noise, cost real money and queue time, so they require an explicit shot
budget and a provider. No spec carries credentials; the worker reads those from
its own environment.

Every sampler supports float, integer and categorical parameters. Floats may be
linear or log-scaled, and floats and integers may have a step; these are
sampled from a discretised logistic-normal with an exactly normalised
probability. Categorical parameters are encoded either in extra qubits
(``"register"``) or by a softmax over the circuit's features (``"head"``).

All four are rebuildable: after every ``rebuild_every`` completed trials the
optimizer worker updates the policy, and between updates trials are drawn from
the last published version.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intelloptim.core.samplers.base import BaseRebuildSamplerSpec, SamplerKind

MAX_QUBITS = 16
"""Largest circuit, in qubits, including the registers of categorical parameters."""

_SECRET_KEY = re.compile(r"token|secret|password|passwd|api[_-]?key|credential", re.I)


class IBMRuntimeProviderSpec(BaseModel):
    """IBM Quantum (Qiskit Runtime) as the execution provider.

    Credentials are never part of the spec: the worker reads them from its own
    environment or saved account.

    Args:
        backend_name (Optional[str]): Name of the device, for example
            ``"ibm_brisbane"``. Defaults to ``None``, which picks the least busy
            operational device.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["ibm_runtime"] = "ibm_runtime"
    backend_name: Optional[str] = Field(default=None, min_length=1)


class PluginProviderSpec(BaseModel):
    """A provider supplied by a plugin, such as AWS or Quantinuum.

    The plugin registers an executor under the name ``provider`` in the
    ``intelloptim.quantum_executors`` entry-point group and receives ``options``
    as keyword arguments. Options are stored with the study and may be logged, so
    keys that look like secrets are rejected.

    Args:
        provider (str): Name the plugin registered; not ``"ibm_runtime"``.
        options (Dict[str, Any]): Plain, non-secret options for the executor.
            Defaults to none.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    options: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider")
    @classmethod
    def _not_builtin(cls, value: str) -> str:
        if value == "ibm_runtime":
            raise ValueError("use IBMRuntimeProviderSpec for ibm_runtime")
        return value

    @field_validator("options")
    @classmethod
    def _no_secrets(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        for key in value:
            if _SECRET_KEY.search(key):
                raise ValueError(
                    f"option {key!r} looks like a secret; credentials come from "
                    "the worker environment, never from a spec"
                )
        return value


HardwareProviderSpec = Union[IBMRuntimeProviderSpec, PluginProviderSpec]
"""Where a hardware sampler runs its circuits."""


class BaseQuantumSamplerSpec(BaseRebuildSamplerSpec):
    """Fields shared by every quantum policy sampler.

    Args:
        seed (Optional[int]): Seed of every random draw. ``None`` seeds from the
            study name, so a study stays reproducible. Defaults to ``None``.
        rebuild_every (int): Completed trials between policy updates, which is
            the size of one training cohort. Overrides the default rebuild
            policy trigger. Defaults to ``10``.
        n_layers (int): Layers of the circuit, each with one ``RY`` per qubit
            and a ring of ``CZ`` gates. Defaults to ``3``.
        categorical_encoding (Literal["register", "head"]): How categorical
            parameters get their probabilities: from the measurement
            probabilities of extra qubits (one register per parameter), or from
            a softmax over the circuit's features, which adds no qubits.
            Defaults to ``"register"``.
        min_category_weight (float): Shrinkage added to register masses, so a
            category the circuit never produces keeps a small probability.
            Used by the register encoding only. Defaults to ``0.001``.
        clip_eps (float): PPO clipping range, in ``(0, 1)``. Defaults to ``0.2``.
        ppo_epochs (int): Maximum gradient steps per update. Each one costs
            ``2P + 1`` circuits, with ``P`` the number of angles. Defaults to
            ``4``.
        lr_theta (float): Adam learning rate of the circuit angles. Defaults to
            ``0.02``.
        lr_phi (float): Adam learning rate of the classical head. Defaults to
            ``0.02``.
        kl_stop (float): Estimated KL to the behaviour policy above which a step
            is shrunk or undone. Defaults to ``0.05``.
        entropy_coef (float): Weight of an entropy bonus, zero to disable it.
            Defaults to ``0.0``.
        advantage (Literal["mean", "standardize", "rank"]): How rewards become
            advantages: centred, centred and scaled, or by rank (robust to
            outliers). Defaults to ``"standardize"``.
        sigma_min (float): Floor of every latent standard deviation. Defaults to
            ``0.05``.
        init_scale (float): Initial latent standard deviation; the first policy
            is broad. Must exceed ``sigma_min``. Defaults to ``1.0``.
        theta_init_std (float): Standard deviation of the initial angles.
            Defaults to ``1.0``.
        min_cohort (int): Fewest usable trials needed to update; a smaller
            cohort is left to grow. At most ``rebuild_every``. Defaults to ``4``.
    """

    seed: Optional[int] = None
    rebuild_every: int = Field(default=10, gt=0)
    n_layers: int = Field(default=3, ge=1, le=16)
    categorical_encoding: Literal["register", "head"] = "register"
    min_category_weight: float = Field(default=1e-3, gt=0, le=0.5)
    clip_eps: float = Field(default=0.2, gt=0, lt=1)
    ppo_epochs: int = Field(default=4, ge=1, le=64)
    lr_theta: float = Field(default=0.02, gt=0, le=1)
    lr_phi: float = Field(default=0.02, gt=0, le=1)
    kl_stop: float = Field(default=0.05, gt=0, allow_inf_nan=False)
    entropy_coef: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    advantage: Literal["mean", "standardize", "rank"] = "standardize"
    sigma_min: float = Field(default=0.05, gt=0)
    init_scale: float = Field(default=1.0, gt=0)
    theta_init_std: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    min_cohort: int = Field(default=4, ge=2)

    @property
    def base_qubits(self) -> int:
        """Qubits of the circuit before any categorical register is added."""
        raise NotImplementedError

    @property
    def n_angles(self) -> int:
        """Trainable angles of the base circuit (registers add more)."""
        return self.base_qubits * self.n_layers

    @model_validator(mode="after")
    def _check_consistency(self) -> "BaseQuantumSamplerSpec":
        if self.init_scale <= self.sigma_min:
            raise ValueError("init_scale must exceed sigma_min")
        if self.min_cohort > self.rebuild_every:
            raise ValueError("min_cohort must not exceed rebuild_every")
        if self.base_qubits > MAX_QUBITS:
            raise ValueError(f"at most {MAX_QUBITS} qubits are supported")
        return self


class BaseQuantumGausSpec(BaseQuantumSamplerSpec):
    """Fields of the Gaus policy.

    The circuit's single-qubit ``Z`` means and the ``ZZ`` moments of neighbouring
    qubits feed an affine head that gives the latent mean and standard deviation
    of every numeric parameter (the draw is mapped into the parameter's range
    with a sigmoid, never clipped).

    Args:
        n_qubits (int): Qubits of the base circuit. Defaults to ``6``.
    """

    n_qubits: int = Field(default=6, ge=2, le=MAX_QUBITS)

    @property
    def base_qubits(self) -> int:
        return self.n_qubits


class BaseQuantumMixtureSpec(BaseQuantumSamplerSpec):
    """Fields of the mixture policy.

    The first ``n_mode_qubits`` qubits are the mode register: their measurement
    probabilities are the weights of ``2 ** n_mode_qubits`` components. The
    next ``n_geometry_qubits`` are the geometry register, whose ``Z`` and ``ZZ``
    moments *conditioned on the mode* give each component's means and standard
    deviations through a shared affine head.

    Args:
        n_mode_qubits (int): Qubits of the mode register. Defaults to ``2``
            (four components).
        n_geometry_qubits (int): Qubits of the geometry register. Defaults to
            ``4``.
        min_mode_weight (float): Shrinkage added to a mode's weight when
            conditioning, so that a rare mode does not divide by a near-zero
            probability. Defaults to ``0.001``.
    """

    n_mode_qubits: int = Field(default=2, ge=1, le=4)
    n_geometry_qubits: int = Field(default=4, ge=2, le=12)
    min_mode_weight: float = Field(default=1e-3, gt=0, le=0.5)

    @property
    def base_qubits(self) -> int:
        return self.n_mode_qubits + self.n_geometry_qubits


class _SimulatedFields(BaseModel):
    """Execution fields of the simulated (ideal) samplers.

    Args:
        shots (Optional[int]): Shots per circuit. ``None`` uses exact
            probabilities; a number adds finite-sampling noise only, not
            hardware noise. Defaults to ``None``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    shots: Optional[int] = Field(default=None, ge=1)


class _HardwareFields(BaseModel):
    """Execution fields of the hardware samplers. All of them are required.

    Args:
        provider (HardwareProviderSpec): Where the circuits run.
        shots (int): Shots per circuit.
        max_total_shots (int): Hard budget on the shots spent over the whole
            life of the study. An update that would exceed it is refused before
            any circuit is submitted. It must allow at least one update of the
            base circuit; categorical registers need more, which is checked
            when the update runs.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: HardwareProviderSpec
    shots: int = Field(ge=1)
    max_total_shots: int = Field(ge=1)


def _check_budget(spec: BaseQuantumSamplerSpec, shots: int, budget: int) -> None:
    needed = shots * (spec.ppo_epochs * (2 * spec.n_angles + 1) + 1)
    if budget < needed:
        raise ValueError(
            f"max_total_shots={budget} cannot pay for one update, which needs "
            f"at least {needed} shots"
        )


class SimQuantumGausSamplerSpec(BaseQuantumGausSpec, _SimulatedFields):
    """Gaus quantum policy on a simulator (no hardware noise)."""

    kind: Literal[SamplerKind.SIM_QUANTUM_GAUS] = SamplerKind.SIM_QUANTUM_GAUS


class SimQuantumMixtureSamplerSpec(BaseQuantumMixtureSpec, _SimulatedFields):
    """Mixture quantum policy on a simulator (no hardware noise)."""

    kind: Literal[SamplerKind.SIM_QUANTUM_MIXTURE] = SamplerKind.SIM_QUANTUM_MIXTURE


class QuantumGausSamplerSpec(BaseQuantumGausSpec, _HardwareFields):
    """Gaus quantum policy on real quantum hardware.

    Every update executes ``ppo_epochs * (2P + 1) + 1`` circuits of ``shots``
    shots, so costs are bounded by ``max_total_shots``. The study is paused for
    asks while an update waits for the device.
    """

    kind: Literal[SamplerKind.QUANTUM_GAUS] = SamplerKind.QUANTUM_GAUS

    @model_validator(mode="after")
    def _budget(self) -> "QuantumGausSamplerSpec":
        _check_budget(self, self.shots, self.max_total_shots)
        return self


class QuantumMixtureSamplerSpec(BaseQuantumMixtureSpec, _HardwareFields):
    """Mixture quantum policy on real quantum hardware.

    Every update executes ``ppo_epochs * (2P + 1) + 1`` circuits of ``shots``
    shots, so costs are bounded by ``max_total_shots``. The study is paused for
    asks while an update waits for the device.
    """

    kind: Literal[SamplerKind.QUANTUM_MIXTURE] = SamplerKind.QUANTUM_MIXTURE

    @model_validator(mode="after")
    def _budget(self) -> "QuantumMixtureSamplerSpec":
        _check_budget(self, self.shots, self.max_total_shots)
        return self


__all__ = [
    "MAX_QUBITS",
    "HardwareProviderSpec",
    "IBMRuntimeProviderSpec",
    "PluginProviderSpec",
    "QuantumGausSamplerSpec",
    "QuantumMixtureSamplerSpec",
    "SimQuantumGausSamplerSpec",
    "SimQuantumMixtureSamplerSpec",
]
