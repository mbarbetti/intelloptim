"""Worker and site identity for the evaluator worker.

``intelloptim-worker evaluator`` identifies itself to the server with a worker
id and a site id built from the SLURM job and array variables, or from the
host name and process id outside SLURM. The environment is read when the
functions are called, so importing this module is safe on any machine.
"""

from __future__ import annotations

import os
import socket


def worker_id(prefix: str = "eval") -> str:
    """Build a stable worker identifier for this process.

    The ID is composed from (in priority order):
    1. ``SLURM_ARRAY_JOB_ID`` + ``SLURM_ARRAY_TASK_ID`` when inside a SLURM
       array job — guarantees uniqueness across array tasks.
    2. ``SLURM_JOB_ID`` when inside a plain SLURM job.
    3. Hostname + PID as a fallback for local/interactive runs.

    Args:
        prefix (str): Short label placed first in the id. Defaults to
            ``"eval"``.

    Returns:
        str: An id safe for use as an HTTP header value and a log field.
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
        str: An id safe for use as an HTTP header value.
    """
    array_job = os.environ.get("SLURM_ARRAY_JOB_ID")
    if array_job:
        return f"slurm-{array_job}"

    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        return f"slurm-{job_id}"

    return socket.gethostname()


__all__ = ["site_id", "worker_id"]
