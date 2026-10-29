"""Worker and site ids for ``intelloptim-worker evaluator``.

A worker identifies itself to the server with a worker id, which names the
process, and a site id, which names the submission or machine it runs on. Both
are built from the SLURM job variables, or from the host name (and process id)
outside SLURM. The environment is read when the functions are called, so
importing this module is safe on any machine.
"""

from __future__ import annotations

import os
import socket


def worker_id(prefix: str = "eval") -> str:
    """Return an identifier for the current worker process.

    The id is built from the first of these that is available:

    1. ``SLURM_ARRAY_JOB_ID`` and ``SLURM_ARRAY_TASK_ID``, in an array job:
       ``<prefix>-slurm-<array job>-<task>``, unique for each array task.
    2. ``SLURM_JOB_ID``, in a plain SLURM job: ``<prefix>-slurm-<job>``.
    3. The host name and process id, elsewhere: ``<prefix>-<host>-<pid>``.

    Args:
        prefix (str): Label placed first in the id. Defaults to ``"eval"``.

    Returns:
        str: The worker id. It is the same for the lifetime of a SLURM task;
        outside SLURM it changes with the process.
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
    """Return an identifier that groups workers by submission or machine.

    The id is built from the first of these that is available:

    1. ``SLURM_ARRAY_JOB_ID``: ``slurm-<array job>``, shared by all the tasks
       of one array job.
    2. ``SLURM_JOB_ID``: ``slurm-<job>``.
    3. The host name, outside SLURM.

    It is sent as the ``X-Site`` header, so the trials of many workers can be
    grouped by where they ran.

    Returns:
        str: The site id.
    """
    array_job = os.environ.get("SLURM_ARRAY_JOB_ID")
    if array_job:
        return f"slurm-{array_job}"

    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        return f"slurm-{job_id}"

    return socket.gethostname()


__all__ = ["site_id", "worker_id"]
