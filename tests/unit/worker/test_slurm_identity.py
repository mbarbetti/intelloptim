"""Worker/site identity derivation in ``worker/slurm.py``.

``site_id()`` groups evaluator workers by submission (SLURM array/job)
or by host for local runs; it is sent as the ``X-Site`` header so the
dashboard's federation view can roll workers up by cluster. The env is
read lazily, so these tests just toggle the relevant SLURM variables.
"""

from __future__ import annotations

import socket

import pytest

from intelloptim.worker.slurm import site_id, worker_id

_SLURM_VARS = ("SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_ID")


@pytest.fixture(autouse=True)
def _clean_slurm_env(monkeypatch):
    """Ensure no ambient SLURM vars leak into these assertions."""
    for var in _SLURM_VARS:
        monkeypatch.delenv(var, raising=False)
    yield


def test_site_id_falls_back_to_hostname_locally() -> None:
    assert site_id() == socket.gethostname()


def test_site_id_uses_array_job_over_plain_job(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "555")
    monkeypatch.setenv("SLURM_JOB_ID", "999")
    # Array-job id identifies the whole submission; all tasks share a site.
    assert site_id() == "slurm-555"


def test_site_id_uses_plain_job_when_not_an_array(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "999")
    assert site_id() == "slurm-999"


def test_site_groups_workers_that_differ_by_task() -> None:
    """Two array tasks share a site but keep distinct worker ids."""
    import os

    os.environ["SLURM_ARRAY_JOB_ID"] = "700"
    try:
        os.environ["SLURM_ARRAY_TASK_ID"] = "1"
        site_a, worker_a = site_id(), worker_id()
        os.environ["SLURM_ARRAY_TASK_ID"] = "2"
        site_b, worker_b = site_id(), worker_id()
    finally:
        for var in ("SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID"):
            os.environ.pop(var, None)
    assert site_a == site_b == "slurm-700"
    assert worker_a != worker_b
