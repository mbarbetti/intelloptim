"""SLURM submission driver for the distributed evaluator demo.

This script:

1. Creates (or joins) the study on the server so the study_id is
   established before any array-job worker starts.
2. Writes ``spec.json`` so workers can load it via ``--spec-file`` if
   needed (redundant when the objective carries ``study_spec=`` metadata,
   but useful for cross-language or external evaluators).
3. Renders ``deploy/slurm/worker-evaluator.sbatch.j2``.
4. Optionally submits via ``sbatch`` (pass ``--submit``).

Usage
-----

On the login node::

    export INTELLOPTIM_URL=https://intelloptim.hpc.example.com
    export INTELLOPTIM_TOKEN=$(cat ~/.config/intelloptim/token)

    python examples/02_distributed_slurm/submit.py \\
        --n-workers 20 \\
        --n-trials-per-worker 50 \\
        --time 02:00:00 \\
        --partition gpu \\
        --submit

Without ``--submit`` the rendered sbatch script is printed for inspection.

Token handling
--------------
The token is *never* written into the sbatch script. SLURM inherits
``INTELLOPTIM_TOKEN`` from the submission environment by default
(``--export=ALL``). Set it in your shell before calling ``sbatch``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

# Make the repo importable when running directly from the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_REPO_ROOT, "src"), _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_objective_module() -> None:
    """Import objective.py by file path (module name starts with a digit)."""
    obj_path = os.path.join(_HERE, "objective.py")
    spec = importlib.util.spec_from_file_location("_slurm_demo_objective", obj_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {obj_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    sys.modules["_slurm_demo_objective"] = mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Submit the distributed SLURM evaluator demo."
    )
    p.add_argument(
        "--url",
        default=os.environ.get("INTELLOPTIM_URL", "http://127.0.0.1:8000"),
        help="Server URL (default: $INTELLOPTIM_URL or http://127.0.0.1:8000).",
    )
    p.add_argument(
        "--n-workers", type=int, default=10, help="Array size (default: 10)."
    )
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Max simultaneous tasks (default: n-workers).",
    )
    p.add_argument(
        "--n-trials-per-worker",
        type=int,
        default=50,
        help="Trials each worker contributes (default: 50).",
    )
    p.add_argument("--time", default="04:00:00", help="SLURM --time limit.")
    p.add_argument("--mem", default="4G", help="SLURM --mem per task.")
    p.add_argument("--cpus", type=int, default=1, help="SLURM --cpus-per-task.")
    p.add_argument("--partition", default=None, help="SLURM partition.")
    p.add_argument("--account", default=None, help="SLURM account.")
    p.add_argument(
        "--log-dir",
        default=os.path.join(_HERE, "logs"),
        help="Directory for SLURM stdout/stderr (default: examples/02_distributed_slurm/logs/).",
    )
    p.add_argument(
        "--conda-env", default="intelloptim", help="Conda env name on cluster."
    )
    p.add_argument(
        "--conda-base", default="/opt/conda", help="Conda base path on cluster."
    )
    p.add_argument(
        "--submit",
        action="store_true",
        help="Submit via sbatch. Without this flag the script is printed only.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Import objective module (registers @intopt.objective side-effect).
    _load_objective_module()

    import intelloptim as intopt
    from intelloptim._internal.schemas import StudyResolveRequest
    from intelloptim.core.pruners import NopPruner
    from intelloptim.worker.slurm import ensure_log_dir, render_sbatch, submit_job

    sampler = intopt.samplers.TPE(seed=42)
    parameters = [
        intopt.params.Float("x", -5.0, 10.0),
        intopt.params.Float("y", 0.0, 15.0),
    ]

    # 1. Create / join the study up-front so the study_id is confirmed.
    with intopt.Client(url=args.url) as client:
        study = client.create_study(
            name="slurm-demo-branin",
            direction="minimize",
            sampler=sampler,
            parameters=parameters,
        )
    action = "joined" if study.existed else "created"
    print(f"Study {study.id} {action}.")

    # 2. Write spec.json for workers that prefer --spec-file over study_spec metadata.
    req = StudyResolveRequest(
        name="slurm-demo-branin",
        direction="minimize",
        sampler=sampler,
        parameters=parameters,
        pruner=NopPruner(),
    )
    spec_path = os.path.join(_HERE, "spec.json")
    with open(spec_path, "w") as fh:
        json.dump(req.model_dump(mode="json"), fh, indent=2)
    print(f"Spec written → {spec_path}")

    # 3. Render sbatch template.
    template_path = os.path.join(
        _REPO_ROOT, "deploy", "slurm", "worker-evaluator.sbatch.j2"
    )
    ensure_log_dir(args.log_dir)

    # The import spec uses the registered module alias we set in _load_objective_module.
    # Workers must add _REPO_ROOT/src to PYTHONPATH and load objective.py the same way
    # the template's evaluator command does (via --import and the registered name lookup).
    # For simplicity the example uses the objective's study_spec kwarg path, no --spec-file.
    rendered = render_sbatch(
        template_path,
        job_name="intopt-branin",
        n_workers=args.n_workers,
        max_concurrent=args.max_concurrent or args.n_workers,
        time_limit=args.time,
        mem=args.mem,
        cpus=args.cpus,
        partition=args.partition,
        account=args.account,
        log_dir=args.log_dir,
        conda_env=args.conda_env,
        conda_base=args.conda_base,
        server_url=args.url,
        # Use full file path as the import spec so workers don't need the examples/
        # directory on PYTHONPATH — they load objective.py directly via the evaluator CLI.
        import_spec=f"{os.path.join(_HERE, 'objective.py')}:branin",
        spec_file=spec_path,
        n_trials=args.n_trials_per_worker,
    )

    # 4. Submit or print.
    if args.submit:
        job_id = submit_job(rendered)
        total = args.n_workers * args.n_trials_per_worker
        print(
            f"Submitted array job {job_id} — "
            f"{args.n_workers} workers × {args.n_trials_per_worker} trials = {total} total."
        )
    else:
        print("\n--- rendered sbatch (pass --submit to submit) ---")
        print(rendered)


if __name__ == "__main__":
    main()
