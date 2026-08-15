"""Objective import helper for the evaluator CLI.

Parses import specs in two forms and returns the resolved callable:

* ``"module.path:callable_name"`` — standard ``importlib.import_module``
  path; the colon part before ``:`` is a dotted Python module path.
* ``"/abs/path/to/file.py:callable_name"`` (or ``"./rel/path.py:name"``) —
  load a source file by filesystem path; useful when the file lives outside
  any installed package (e.g. the digit-prefixed example directories).

Importing the module is sufficient to trigger ``@intopt.objective``
registration side-effects.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import types
from typing import Any, Callable, cast


class ImportSpecError(ValueError):
    """Raised when an import spec string cannot be parsed or resolved."""


def import_callable(spec: str) -> Callable[..., Any]:
    """Import a callable from an import spec string.

    Supported forms:

    * ``"package.module:attr_name"`` — standard module import.
    * ``"/path/to/file.py:attr_name"`` or ``"./rel/file.py:attr_name"`` —
      load by file path (the module is registered as ``_intopt_<stem>``
      in ``sys.modules`` to avoid repeated loading).

    Args:
        spec: Import spec string in one of the two forms above.

    Returns:
        The resolved callable.

    Raises:
        ImportSpecError: If the spec is malformed, the module cannot be
            imported, or the attribute does not exist on the module.
    """
    if ":" not in spec:
        raise ImportSpecError(
            f"import spec {spec!r} must be in 'module.path:callable' "
            "or '/path/to/file.py:callable' form"
        )
    module_part, _, attr_name = spec.partition(":")
    if not module_part or not attr_name:
        raise ImportSpecError(
            f"import spec {spec!r} must be in 'module.path:callable' "
            "or '/path/to/file.py:callable' form"
        )

    # File-path form: module_part ends with .py or is a path-like string.
    is_file_path = (
        module_part.endswith(".py")
        or os.sep in module_part
        or module_part.startswith(".")
    )
    if is_file_path:
        module = _import_from_file(module_part)
    else:
        try:
            module = importlib.import_module(module_part)
        except ImportError as exc:
            raise ImportSpecError(
                f"cannot import module {module_part!r}: {exc}"
            ) from exc

    try:
        obj = getattr(module, attr_name)
    except AttributeError as exc:
        raise ImportSpecError(
            f"module {module_part!r} has no attribute {attr_name!r}"
        ) from exc
    if not callable(obj):
        raise ImportSpecError(
            f"{module_part}:{attr_name} resolved to a non-callable "
            f"{type(obj).__name__}"
        )
    return cast(Callable[..., Any], obj)


def _import_from_file(file_path: str) -> types.ModuleType:
    import sys

    abs_path = os.path.abspath(file_path)
    if not os.path.isfile(abs_path):
        raise ImportSpecError(f"file not found: {abs_path!r}")

    stem = os.path.splitext(os.path.basename(abs_path))[0]
    mod_name = f"_intopt_{stem}"

    if mod_name in sys.modules:
        return sys.modules[mod_name]

    file_spec = importlib.util.spec_from_file_location(mod_name, abs_path)
    if file_spec is None or file_spec.loader is None:
        raise ImportSpecError(f"cannot create module spec for {abs_path!r}")
    mod = importlib.util.module_from_spec(file_spec)
    sys.modules[mod_name] = mod
    try:
        file_spec.loader.exec_module(mod)
    except Exception as exc:
        del sys.modules[mod_name]
        raise ImportSpecError(f"error loading {abs_path!r}: {exc}") from exc
    return mod


__all__ = ["ImportSpecError", "import_callable"]
