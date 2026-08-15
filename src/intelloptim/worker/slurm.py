"""SLURM bootstrap helpers for the intelloptim evaluator worker.

These utilities are used by:

* ``intelloptim-worker evaluator`` — to self-identify with a stable worker ID
  derived from SLURM job/array env vars.
* ``examples/02_distributed_slurm/submit.py`` — to render and submit the
  sbatch template.

None of this code assumes a SLURM environment at *import* time; all SLURM
env vars are read lazily so the same module loads cleanly on laptops.
"""

from __future__ import annotations

import os
import socket
import subprocess
import tempfile
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Worker identity
# ---------------------------------------------------------------------------


def slurm_env() -> Dict[str, str]:
    """Return a dict of SLURM environment variables present in the process.

    Returns an empty dict when running outside a SLURM job.
    """
    keys = (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_NAME",
        "SLURM_NODELIST",
        "SLURM_CPUS_PER_TASK",
        "SLURM_MEM_PER_NODE",
    )
    return {k: os.environ[k] for k in keys if k in os.environ}


def worker_id(prefix: str = "eval") -> str:
    """Build a stable worker identifier for this process.

    The ID is composed from (in priority order):
    1. ``SLURM_ARRAY_JOB_ID`` + ``SLURM_ARRAY_TASK_ID`` when inside a SLURM
       array job — guarantees uniqueness across array tasks.
    2. ``SLURM_JOB_ID`` when inside a plain SLURM job.
    3. Hostname + PID as a fallback for local/interactive runs.

    Args:
        prefix: Short label prepended to the ID (default ``"eval"``).

    Returns:
        A string safe for use as an HTTP header value and log field.
    """
    array_job = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if array_job and array_task:
        return f"{prefix}-slurm-{array_job}-{array_task}"

    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        return f"{prefix}-slurm-{job_id}"

    return f"{prefix}-{socket.gethostname()}-{os.getpid()}"


def site_id() -> str:
    """Build a stable *site* identifier grouping workers by submission or host.

    Priority: ``SLURM_ARRAY_JOB_ID`` (all array tasks of one submission share
    a site), then ``SLURM_JOB_ID``, then the hostname for local/interactive
    runs. Sent as the ``X-Site`` header so the dashboard's federation view can
    roll individual workers up by the cluster or machine they ran on.

    Returns:
        A string safe for use as an HTTP header value.
    """
    array_job = os.environ.get("SLURM_ARRAY_JOB_ID")
    if array_job:
        return f"slurm-{array_job}"

    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        return f"slurm-{job_id}"

    return socket.gethostname()


# ---------------------------------------------------------------------------
# sbatch template rendering
# ---------------------------------------------------------------------------


def render_sbatch(template_path: str, **context: Any) -> str:
    """Render a Jinja2 sbatch template with *context* variables.

    Args:
        template_path: Absolute or relative path to a ``.j2`` template file.
        **context: Variables passed to the Jinja2 template environment.

    Returns:
        The rendered script as a string.

    Raises:
        ImportError: If ``jinja2`` is not installed (add it to the
            ``[worker]`` extra or ``conda install jinja2``).
        FileNotFoundError: If *template_path* does not exist.
    """
    try:
        import jinja2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "jinja2 is required for sbatch template rendering; "
            "install it with: conda install jinja2"
        ) from exc

    template_path = os.path.abspath(template_path)
    template_dir = os.path.dirname(template_path)
    template_name = os.path.basename(template_path)

    loader = jinja2.FileSystemLoader(template_dir)
    env = jinja2.Environment(loader=loader, keep_trailing_newline=True)
    template = env.get_template(template_name)
    rendered: str = template.render(**context)
    return rendered


# ---------------------------------------------------------------------------
# Job submission
# ---------------------------------------------------------------------------


def submit_job(
    script_content: str,
    *,
    extra_args: Optional[list[str]] = None,
) -> str:
    """Submit an sbatch job from an in-memory script string.

    Writes *script_content* to a temporary file, calls ``sbatch``, and
    returns the job ID extracted from sbatch's stdout.

    Args:
        script_content: The fully rendered sbatch script.
        extra_args: Additional arguments appended to the ``sbatch`` command
            (e.g. ``["--parsable"]`` is already included by default).

    Returns:
        The SLURM job ID as a string (e.g. ``"12345678"``).

    Raises:
        FileNotFoundError: If ``sbatch`` is not on ``$PATH``.
        RuntimeError: If ``sbatch`` exits non-zero.
    """
    args = extra_args or []
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", delete=False, prefix="intopt_"
    ) as tmp:
        tmp.write(script_content)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["sbatch", "--parsable"] + args + [tmp_path],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        os.unlink(tmp_path)

    if result.returncode != 0:
        raise RuntimeError(
            f"sbatch exited with code {result.returncode}:\n{result.stderr.strip()}"
        )
    # --parsable prints "jobid" or "jobid;cluster" — take the first token.
    return result.stdout.strip().split(";")[0]


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def ensure_log_dir(path: str) -> None:
    """Create *path* (and parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)


__all__ = [
    "ensure_log_dir",
    "render_sbatch",
    "site_id",
    "slurm_env",
    "submit_job",
    "worker_id",
]
