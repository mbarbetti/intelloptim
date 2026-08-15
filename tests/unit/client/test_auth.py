"""Token resolution (explicit / env var / config file)."""

from __future__ import annotations

import pytest

from intelloptim.client.auth import MissingTokenError, resolve_token


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
