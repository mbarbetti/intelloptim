"""Hyperparameters of a quantum policy sampler, as plain values.

The sampler spec in :mod:`intelloptim.core.samplers.quantum` is converted to
this dataclass, so the runtime does not depend on pydantic models and tests can
build a configuration directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class QuantumPolicyConfig:
    """Configuration shared by the Gaus and mixture quantum policies.

    Attributes:
        algorithm (str): ``"gaus"`` (one component) or ``"mixture"``.
        n_mode_qubits (int): Qubits of the mode register; ``0`` for ``"gaus"``,
            otherwise the mixture has ``2 ** n`` components.
        n_feature_qubits (int): Qubits whose ``Z`` and ``ZZ`` moments feed the
            head (the geometry register of a mixture).
        n_layers (int): Layers of the ansatz.
        seed (Optional[int]): Seed of every random draw; ``None`` derives it
            from the study name.
        rebuild_every (int): Completed trials between policy updates.
        categorical_encoding (str): ``"register"`` or ``"head"``.
        min_category_weight (float): Shrinkage of register masses.
        clip_eps (float): PPO clipping range.
        ppo_epochs (int): Gradient steps per update.
        lr_theta (float): Adam learning rate of the circuit angles.
        lr_phi (float): Adam learning rate of the classical head.
        kl_stop (float): Estimated KL above which a step is shrunk or undone.
        entropy_coef (float): Weight of the entropy bonus.
        advantage (str): ``"mean"``, ``"standardize"`` or ``"rank"``.
        sigma_min (float): Floor of every latent standard deviation.
        init_scale (float): Initial latent standard deviation (broad).
        theta_init_std (float): Standard deviation of the initial angles.
        min_cohort (int): Fewest usable trials needed to update the policy.
        min_mode_weight (float): Shrinkage of conditioned features for rare
            modes (mixture only).
        shots (Optional[int]): Shots per circuit; ``None`` is exact.
        max_total_shots (Optional[int]): Shot budget over the study's life.
    """

    algorithm: str
    n_mode_qubits: int = 0
    n_feature_qubits: int = 6
    n_layers: int = 3
    seed: Optional[int] = None
    rebuild_every: int = 10
    categorical_encoding: str = "register"
    min_category_weight: float = 1e-3
    clip_eps: float = 0.2
    ppo_epochs: int = 4
    lr_theta: float = 0.02
    lr_phi: float = 0.02
    kl_stop: float = 0.05
    entropy_coef: float = 0.0
    advantage: str = "standardize"
    sigma_min: float = 0.05
    init_scale: float = 1.0
    theta_init_std: float = 1.0
    min_cohort: int = 4
    min_mode_weight: float = 1e-3
    shots: Optional[int] = None
    max_total_shots: Optional[int] = None

    @classmethod
    def from_spec(cls, spec: Any) -> "QuantumPolicyConfig":
        """Build the configuration a quantum sampler spec describes."""
        if hasattr(spec, "n_mode_qubits"):
            shape: Dict[str, Any] = {
                "algorithm": "mixture",
                "n_mode_qubits": spec.n_mode_qubits,
                "n_feature_qubits": spec.n_geometry_qubits,
                "min_mode_weight": spec.min_mode_weight,
            }
        else:
            shape = {"algorithm": "gaus", "n_feature_qubits": spec.n_qubits}
        return cls(
            n_layers=spec.n_layers,
            seed=spec.seed,
            rebuild_every=spec.rebuild_every,
            categorical_encoding=spec.categorical_encoding,
            min_category_weight=spec.min_category_weight,
            clip_eps=spec.clip_eps,
            ppo_epochs=spec.ppo_epochs,
            lr_theta=spec.lr_theta,
            lr_phi=spec.lr_phi,
            kl_stop=spec.kl_stop,
            entropy_coef=spec.entropy_coef,
            advantage=spec.advantage,
            sigma_min=spec.sigma_min,
            init_scale=spec.init_scale,
            theta_init_std=spec.theta_init_std,
            min_cohort=spec.min_cohort,
            shots=spec.shots,
            max_total_shots=getattr(spec, "max_total_shots", None),
            **shape,
        )

    def model_shape(self) -> Dict[str, Any]:
        """Fields that fix the layout of the stored model."""
        return {
            "algorithm": self.algorithm,
            "n_layers": self.n_layers,
            "n_mode_qubits": self.n_mode_qubits,
            "n_feature_qubits": self.n_feature_qubits,
            "categorical_encoding": self.categorical_encoding,
        }


__all__ = ["QuantumPolicyConfig"]
