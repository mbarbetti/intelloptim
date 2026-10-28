"""One PPO update of a quantum policy from a cohort of on-policy trials.

The learner is the only writer of the policy. It is meant to run in the
optimizer worker (the ``rebuild`` step), never inside an ask, because it may
execute many circuits.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

import numpy as np

from intelloptim._internal.logging import get_logger
from intelloptim.algorithms.backend.quantum.commons.config import QuantumPolicyConfig
from intelloptim.algorithms.backend.quantum.commons.distributions import (
    DistGrads,
    entropy_grads,
    joint_log_prob,
    joint_log_prob_grads,
)
from intelloptim.algorithms.backend.quantum.commons.executor import QuantumExecutor
from intelloptim.algorithms.backend.quantum.commons.measurement import (
    circuits_per_gradient,
    parameter_shift,
)
from intelloptim.algorithms.backend.quantum.commons.policy import QuantumPolicy
from intelloptim.algorithms.backend.quantum.commons.ppo import (
    Adam,
    advantages,
    clipped_surrogate,
    kl_estimate,
)
from intelloptim.algorithms.backend.quantum.commons.resources import (
    QuantumBudgetExceededError,
    ResourceLedger,
)
from intelloptim.algorithms.backend.quantum.commons.snapshot import (
    Snapshot,
    shape_digest,
)

_log = get_logger(__name__)

_BACKTRACK_FRACTIONS = (0.5, 0.25, 0.125)


@dataclass(frozen=True)
class Cohort:
    """On-policy trials of one snapshot version.

    Attributes:
        ks (np.ndarray): Sampled component of each trial, shape ``(B,)``.
        actions (np.ndarray): Numeric actions of each trial (latent draw or bin
            index), shape ``(B, d)``.
        cats (np.ndarray): Category indices of each trial, shape ``(B, n_cat)``.
        rewards (np.ndarray): Reward of each trial (larger is better).
    """

    ks: np.ndarray
    actions: np.ndarray
    cats: np.ndarray
    rewards: np.ndarray


def initial_snapshot(
    config: QuantumPolicyConfig,
    policy: QuantumPolicy,
    executor: QuantumExecutor,
    rng: np.random.Generator,
) -> Snapshot:
    """Return version ``0``: a broad policy defined by the seed and the layout.

    It is a deterministic function of the configuration, so it never needs to be
    stored before the first update. Its distribution is computed with *executor*,
    which is an exact simulator, so no hardware is used to define the prior.
    """
    theta = policy.init_theta(rng)
    phi = policy.init_phi(rng)
    p = executor.probabilities(policy.ansatz, theta[None, :])[0]
    return Snapshot(
        version=0,
        names=list(policy.layout.names),
        theta=theta,
        phi=phi,
        dist=policy.distribution(p, phi),
        digest=shape_digest(config.model_shape(), policy.layout.digest_payload()),
    )


def update(
    config: QuantumPolicyConfig,
    policy: QuantumPolicy,
    executor: QuantumExecutor,
    snapshot: Snapshot,
    cohort: Cohort,
) -> "tuple[Snapshot, Dict[str, Any]]":
    """Apply one batched PPO update and return the next snapshot and a summary.

    The behaviour log-probabilities come from the distribution cached in
    *snapshot*, i.e. the exact one that generated the cohort. Each epoch
    evaluates the circuit and its parameter-shift Jacobian once, shared by all
    trials. If the estimated KL to the behaviour policy exceeds ``kl_stop``, the
    last step is shrunk (halved up to three times, one circuit each) until it
    fits, or undone if none does, and the update stops.

    Raises:
        QuantumBudgetExceededError: Before any circuit runs, if the planned
            update does not fit in the executor's remaining shot budget.
    """
    ansatz = policy.ansatz
    layout = policy.layout
    planned = config.ppo_epochs * circuits_per_gradient(ansatz) + 1
    shots = executor.shots if hasattr(executor, "shots") and executor.shots else 0
    if shots and not executor.ledger.would_fit(planned, shots):
        raise QuantumBudgetExceededError(
            f"an update needs {planned * shots} shots but the remaining budget "
            f"is {(executor.ledger.max_shots or 0) - executor.ledger.shots}"
        )

    adv = advantages(cohort.rewards, config.advantage)
    logp_old = joint_log_prob(
        snapshot.dist, layout, cohort.ks, cohort.actions, cohort.cats
    )
    theta, phi = snapshot.theta.copy(), {k: v.copy() for k, v in snapshot.phi.items()}
    adam_theta, adam_phi = Adam(config.lr_theta), Adam(config.lr_phi)
    if snapshot.adam_theta:
        adam_theta.load_json(snapshot.adam_theta)
    if snapshot.adam_phi:
        adam_phi.load_json(snapshot.adam_phi)

    prev: Optional[Dict[str, Any]] = None
    kl, surrogate, epochs_run, backtracks, stopped_early = 0.0, 0.0, 0, 0, False
    p: np.ndarray
    for epoch in range(config.ppo_epochs):
        p, jac = parameter_shift(executor, ansatz, theta)
        dist = policy.distribution(p, phi)
        logp = joint_log_prob(dist, layout, cohort.ks, cohort.actions, cohort.cats)
        kl = kl_estimate(logp, logp_old)
        if epoch > 0 and kl > config.kl_stop:
            # The last step left the trust region: shrink it (one circuit per
            # try) and stop there; undo it entirely if no fraction fits.
            assert prev is not None
            stopped_early = True
            accepted = False
            for fraction in _BACKTRACK_FRACTIONS:
                backtracks += 1
                cand_theta = prev["theta"] + fraction * (theta - prev["theta"])
                cand_phi = {
                    k: prev["phi"][k] + fraction * (phi[k] - prev["phi"][k])
                    for k in phi
                }
                cand_p = executor.probabilities(ansatz, cand_theta[None, :])[0]
                cand_dist = policy.distribution(cand_p, cand_phi)
                cand_kl = kl_estimate(
                    joint_log_prob(
                        cand_dist, layout, cohort.ks, cohort.actions, cohort.cats
                    ),
                    logp_old,
                )
                if cand_kl <= config.kl_stop:
                    theta, phi, p, kl = cand_theta, cand_phi, cand_p, cand_kl
                    accepted = True
                    break
            if not accepted:
                theta, phi, p = prev["theta"], prev["phi"], prev["p"]
                kl = prev["kl"]
            break
        prev = {
            "theta": theta.copy(),
            "phi": {k: v.copy() for k, v in phi.items()},
            "p": p,
            "kl": kl,
        }
        surrogate, d_logp = clipped_surrogate(logp, logp_old, adv, config.clip_eps)
        grads: DistGrads = joint_log_prob_grads(
            dist, layout, cohort.ks, cohort.actions, cohort.cats, d_logp
        )
        if config.entropy_coef > 0:
            eg = entropy_grads(dist)
            grads.weights = grads.weights + config.entropy_coef * eg.weights
            grads.sigmas = grads.sigmas + config.entropy_coef * eg.sigmas
            grads.cats = [
                g + config.entropy_coef * e
                for g, e in zip(grads.cats, eg.cats, strict=True)
            ]
        grad_p, grad_phi = policy.backward(p, phi, grads)
        grad_theta = jac @ grad_p
        # Ascent on the objective is descent on its negative.
        theta = adam_theta.step({"t": theta}, {"t": -grad_theta})["t"]
        phi = adam_phi.step(phi, {k: -v for k, v in grad_phi.items()})
        epochs_run += 1
    else:
        p = executor.probabilities(ansatz, theta[None, :])[0]

    new_dist = policy.distribution(p, phi)
    ledger: ResourceLedger = executor.ledger
    new = replace(
        snapshot,
        version=snapshot.version + 1,
        theta=theta,
        phi=phi,
        dist=new_dist,
        adam_theta=adam_theta.to_json(),
        adam_phi=adam_phi.to_json(),
        resources=ledger.to_json(),
        n_updates=snapshot.n_updates + 1,
    )
    summary: Dict[str, Any] = {
        "version": new.version,
        "cohort_size": int(cohort.rewards.shape[0]),
        "epochs": epochs_run,
        "stopped_early": stopped_early,
        "backtracks": backtracks,
        "kl": kl,
        "surrogate": surrogate,
        "circuits_total": ledger.circuits,
        "shots_total": ledger.shots,
    }
    _log.info("quantum policy updated", extra=summary)
    return new, summary


__all__ = ["Cohort", "initial_snapshot", "update"]
