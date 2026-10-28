"""Token resolution (explicit / env var / config file)."""

from __future__ import annotations

import os

import pytest

from intelloptim.client.auth import resolve_token
from intelloptim.client.errors import IntelloptimError, MissingTokenError


def test_explicit_wins(monkeypatch) -> None:
    monkeypatch.setenv("INTELLOPTIM_TOKEN", "from-env")
    assert resolve_token("explicit") == "explicit"


def test_env_var(monkeypatch) -> None:
    monkeypatch.setenv("INTELLOPTIM_TOKEN", "from-env")
    assert resolve_token() == "from-env"


def test_strips_whitespace(monkeypatch) -> None:
    monkeypatch.setenv("INTELLOPTIM_TOKEN", "  padded-token  \n")
    assert resolve_token() == "padded-token"


def test_config_file(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("INTELLOPTIM_TOKEN", raising=False)
    fake_home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(fake_home))
    config_dir = fake_home / ".config" / "intelloptim"
    config_dir.mkdir(parents=True)
    (config_dir / "token").write_text("from-file\n")
    assert resolve_token() == "from-file"


def test_missing_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("INTELLOPTIM_TOKEN", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "no-such-home"))
    with pytest.raises(MissingTokenError):
        resolve_token()


def test_empty_env_does_not_satisfy(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("INTELLOPTIM_TOKEN", "")
    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))
    with pytest.raises(MissingTokenError):
        resolve_token()


def test_blank_explicit_falls_through_to_env(monkeypatch) -> None:
    monkeypatch.setenv("INTELLOPTIM_TOKEN", "from-env")
    assert resolve_token("   ") == "from-env"


def test_blank_env_falls_through_to_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("INTELLOPTIM_TOKEN", "  \n")
    fake_home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(fake_home))
    config_dir = fake_home / ".config" / "intelloptim"
    config_dir.mkdir(parents=True)
    (config_dir / "token").write_text("from-file\n")
    assert resolve_token() == "from-file"


def test_blank_everywhere_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("INTELLOPTIM_TOKEN", "  ")
    monkeypatch.setenv("HOME", str(tmp_path / "no-such-home"))
    with pytest.raises(MissingTokenError):
        resolve_token("\t")


def test_blank_token_file_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("INTELLOPTIM_TOKEN", raising=False)
    fake_home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(fake_home))
    config_dir = fake_home / ".config" / "intelloptim"
    config_dir.mkdir(parents=True)
    (config_dir / "token").write_text("\n  \n")
    with pytest.raises(MissingTokenError):
        resolve_token()


def test_missing_token_error_is_a_sdk_error() -> None:
    import intelloptim.client as client

    assert issubclass(MissingTokenError, IntelloptimError)
    assert client.errors.MissingTokenError is MissingTokenError


def _write_token_file(tmp_path, monkeypatch, mode: int) -> None:
    monkeypatch.delenv("INTELLOPTIM_TOKEN", raising=False)
    fake_home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(fake_home))
    config_dir = fake_home / ".config" / "intelloptim"
    config_dir.mkdir(parents=True)
    token_file = config_dir / "token"
    token_file.write_text("from-file\n")
    token_file.chmod(mode)


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
@pytest.mark.parametrize("mode", [0o644, 0o640, 0o604, 0o666, 0o660])
def test_shared_token_file_warns(tmp_path, monkeypatch, log_records, mode: int) -> None:
    _write_token_file(tmp_path, monkeypatch, mode)
    assert resolve_token() == "from-file"
    messages = [r.getMessage() for r in log_records.records]
    assert any("accessible by other users" in m for m in messages)
    assert all("from-file" not in m for m in messages)


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
@pytest.mark.parametrize("mode", [0o600, 0o400])
def test_private_token_file_does_not_warn(
    tmp_path, monkeypatch, log_records, mode: int
) -> None:
    _write_token_file(tmp_path, monkeypatch, mode)
    assert resolve_token() == "from-file"
    assert not [r for r in log_records.records if "token file" in r.getMessage()]


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_no_permission_warning_for_other_sources(
    tmp_path, monkeypatch, log_records
) -> None:
    _write_token_file(tmp_path, monkeypatch, 0o644)
    monkeypatch.setenv("INTELLOPTIM_TOKEN", "from-env")
    assert resolve_token() == "from-env"
    assert resolve_token("explicit") == "explicit"
    assert not log_records.records


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_log_fixture_sees_the_warning(tmp_path, monkeypatch, log_records) -> None:
    """Guards the capture mechanism the negative tests above rely on."""
    _write_token_file(tmp_path, monkeypatch, 0o666)
    resolve_token()
    assert log_records.records
