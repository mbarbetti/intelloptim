"""``intelloptim-worker`` command line — option parsing and hand-off."""

from __future__ import annotations

from typing import Any, Dict, Iterator

import pytest
from click.testing import CliRunner

from intelloptim.worker import cli as worker_cli
from intelloptim.worker import evaluator_cli


@pytest.fixture
def calls(monkeypatch) -> Iterator[Dict[str, Any]]:
    """Replace the evaluator loop and record the arguments it receives."""
    seen: Dict[str, Any] = {}
    monkeypatch.setattr(evaluator_cli, "run_evaluator", lambda **kw: seen.update(kw))
    for var in ("INTELLOPTIM_URL", "INTELLOPTIM_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    yield seen


def _run(*args: str):
    return CliRunner().invoke(worker_cli.main, list(args))


def test_help_lists_the_evaluator_command() -> None:
    result = _run("--help")
    assert result.exit_code == 0
    assert "evaluator" in result.output


def test_evaluator_passes_every_option(calls, tmp_path) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text("{}")
    result = _run(
        "evaluator",
        "--server", "https://example.org",
        "--import", "pkg.mod:fn",
        "--token", "tok",
        "--spec-file", str(spec),
        "--n-trials", "7",
        "--worker-id", "w1",
        "--site", "s1",
        "--loglevel", "debug",
        "--no-catch",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert calls == {
        "server": "https://example.org",
        "import_spec": "pkg.mod:fn",
        "token": "tok",
        "spec_file": str(spec),
        "n_trials": 7,
        "wid": "w1",
        "site": "s1",
        "loglevel": "DEBUG",
        "catch_exceptions": False,
    }


def test_defaults(calls) -> None:
    result = _run("evaluator", "--server", "https://example.org", "--import", "m:f")
    assert result.exit_code == 0, result.output
    assert calls["n_trials"] == 50
    assert calls["loglevel"] == "INFO"
    assert calls["catch_exceptions"] is True
    assert calls["token"] is None
    assert calls["wid"] is None and calls["site"] is None
    assert calls["spec_file"] is None


def test_server_and_token_come_from_the_environment(calls, monkeypatch) -> None:
    monkeypatch.setenv("INTELLOPTIM_URL", "https://env.example")
    monkeypatch.setenv("INTELLOPTIM_TOKEN", "env-token")
    result = _run("evaluator", "--import", "m:f")
    assert result.exit_code == 0, result.output
    assert calls["server"] == "https://env.example"
    assert calls["token"] == "env-token"


@pytest.mark.parametrize(
    "args",
    [
        ["evaluator", "--import", "m:f"],  # no server
        ["evaluator", "--server", "https://example.org"],  # no import spec
    ],
)
def test_required_options(calls, args) -> None:
    result = _run(*args)
    assert result.exit_code == 2
    assert "Missing option" in result.output
    assert calls == {}


def test_spec_file_must_exist(calls, tmp_path) -> None:
    result = _run(
        "evaluator", "--server", "u", "--import", "m:f",
        "--spec-file", str(tmp_path / "absent.json"),
    )  # fmt: skip
    assert result.exit_code == 2
    assert calls == {}


def test_loglevel_is_validated(calls) -> None:
    result = _run("evaluator", "--server", "u", "--import", "m:f", "--loglevel", "x")
    assert result.exit_code == 2
    assert calls == {}
