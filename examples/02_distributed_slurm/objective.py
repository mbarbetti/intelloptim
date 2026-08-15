"""Example objective for the distributed SLURM demo.

Minimises the 2-D Branin function (global minimum ≈ 0.398).

This module is imported by the evaluator workers via::

    intelloptim-worker evaluator --import examples.02_distributed_slurm.objective:branin

The ``@intopt.objective`` decorator registers the function *and* attaches
the study spec so workers can join (or create) the study without a
separate spec file.
"""

from __future__ import annotations

import math

import intelloptim as intopt


@intopt.objective(
    "slurm-demo-branin",
    study_spec={
        "name": "slurm-demo-branin",
        "direction": "minimize",
        "sampler": {"kind": "tpe", "seed": 42},
        "parameters": [
            {"kind": "float", "name": "x", "low": -5.0, "high": 10.0},
            {"kind": "float", "name": "y", "low": 0.0, "high": 15.0},
        ],
        "pruner": {"kind": "nop"},
    },
)
def branin(trial: intopt.Trial) -> float:
    """Branin function — standard BO benchmark, global min ≈ 0.398."""
    x = trial.suggest_float("x", -5.0, 10.0)
    y = trial.suggest_float("y", 0.0, 15.0)
    a = 1.0
    b = 5.1 / (4 * math.pi**2)
    c = 5.0 / math.pi
    r, s, t = 6.0, 10.0, 1.0 / (8 * math.pi)
    return a * (y - b * x**2 + c * x - r) ** 2 + s * (1 - t) * math.cos(x) + s
