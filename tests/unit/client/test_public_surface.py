"""The public surface of ``intelloptim.client``."""

import intelloptim.client as client
from intelloptim.client import errors


def test_all_names_resolve() -> None:
    for name in client.__all__:
        assert hasattr(client, name), name


def test_errors_module_is_exposed_not_each_error() -> None:
    assert client.errors is errors
    assert "errors" in client.__all__
    assert "IntelloptimError" not in client.__all__
    assert issubclass(client.errors.StudyStoppedError, client.errors.IntelloptimError)


def test_internals_are_not_exported() -> None:
    assert "DEFAULT_TIMEOUT" not in client.__all__
    assert "resolve_token" not in client.__all__
