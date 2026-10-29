"""``worker/registry.py`` — resolving ``module:callable`` and ``file.py:callable``."""

from __future__ import annotations

import sys
from typing import Any, Iterator

import pytest

from intelloptim.worker.registry import ImportSpecError, import_callable


@pytest.fixture(autouse=True)
def _forget_loaded_modules() -> Iterator[None]:
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        if name.startswith(("_intopt_", "intopt_registry_")):
            del sys.modules[name]


def _write(tmp_path: Any, name: str, body: str) -> str:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return str(path)


class TestModulePath:
    def test_imports_a_callable_from_a_module(self, tmp_path, monkeypatch) -> None:
        _write(tmp_path, "intopt_registry_mod.py", "def f():\n    return 7\n")
        monkeypatch.syspath_prepend(str(tmp_path))
        assert import_callable("intopt_registry_mod:f")() == 7

    def test_stdlib_callable(self) -> None:
        assert import_callable("math:sqrt")(9.0) == 3.0

    def test_missing_module(self) -> None:
        with pytest.raises(ImportSpecError, match="cannot import module"):
            import_callable("intopt_registry_no_such_module:f")

    def test_missing_attribute(self) -> None:
        with pytest.raises(ImportSpecError, match="no attribute 'nope'"):
            import_callable("math:nope")

    def test_attribute_that_is_not_callable(self) -> None:
        with pytest.raises(ImportSpecError, match="non-callable float"):
            import_callable("math:pi")


class TestFilePath:
    def test_imports_a_callable_from_a_file(self, tmp_path) -> None:
        path = _write(tmp_path, "my_objective.py", "def f():\n    return 'file'\n")
        assert import_callable(f"{path}:f")() == "file"

    def test_relative_path(self, tmp_path, monkeypatch) -> None:
        _write(tmp_path, "rel_objective.py", "def f():\n    return 1\n")
        monkeypatch.chdir(tmp_path)
        assert import_callable("./rel_objective.py:f")() == 1

    def test_same_file_is_loaded_once(self, tmp_path) -> None:
        path = _write(
            tmp_path, "once.py", "COUNT = []\nCOUNT.append(1)\ndef f(): ...\n"
        )
        first = import_callable(f"{path}:f")
        second = import_callable(f"{path}:f")
        assert first.__globals__["COUNT"] == [1]
        assert first.__globals__ is second.__globals__

    def test_missing_file(self, tmp_path) -> None:
        with pytest.raises(ImportSpecError, match="file not found"):
            import_callable(f"{tmp_path}/absent.py:f")

    def test_file_that_raises_while_loading_can_be_retried(self, tmp_path) -> None:
        path = _write(tmp_path, "broken.py", "raise RuntimeError('boom')\n")
        with pytest.raises(ImportSpecError, match="boom"):
            import_callable(f"{path}:f")
        assert "_intopt_broken" not in sys.modules
        _write(tmp_path, "broken.py", "def f():\n    return 'fixed'\n")
        assert import_callable(f"{path}:f")() == "fixed"

    def test_missing_attribute_in_file(self, tmp_path) -> None:
        path = _write(tmp_path, "noattr.py", "x = 1\n")
        with pytest.raises(ImportSpecError, match="no attribute 'f'"):
            import_callable(f"{path}:f")


class TestMalformedSpec:
    @pytest.mark.parametrize("spec", ["math", "", ":f", "math:", ":"])
    def test_rejected_before_importing_anything(self, spec: str) -> None:
        with pytest.raises(ImportSpecError, match="must be in"):
            import_callable(spec)

    def test_error_is_a_value_error(self) -> None:
        assert issubclass(ImportSpecError, ValueError)
