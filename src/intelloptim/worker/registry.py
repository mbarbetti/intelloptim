"""Resolve the ``--objective`` argument of the evaluator CLI to a callable.

An import spec has the form ``<where>:<name>``, with ``<where>`` one of:

* a dotted module path, such as ``"my_package.objectives:run"``, imported with
  :mod:`importlib`;
* a path to a Python file, such as ``"/abs/path/objective.py:run"`` or
  ``"./objective.py:run"``, for code that is not installed as a package, for
  example a project directory on a cluster.

Importing the module also runs its ``@intopt.objective`` decorators, which is
how the study spec attached to the objective gets registered.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from typing import Any, Callable, cast


class ImportSpecError(ValueError):
    """Raised when an import spec is malformed or cannot be resolved."""


def import_callable(spec: str) -> Callable[..., Any]:
    """Import the callable that an import spec points to.

    The part before the colon is a file path when it ends with ``.py``,
    contains a path separator or starts with a dot; otherwise it is a dotted
    module path. A file is loaded once: later calls for the same file name
    reuse the module, which is registered in ``sys.modules`` as
    ``_intopt_<stem>``.

    Args:
        spec (str): ``"package.module:name"``, ``"/path/to/file.py:name"`` or
            ``"./relative/file.py:name"``.

    Returns:
        Callable[..., Any]: The object called *name* in that module.

    Raises:
        ImportSpecError: If the spec has no colon or an empty part, the module
            or file cannot be loaded, the module has no such attribute, or the
            attribute is not callable.
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
    """Load the Python file at *file_path* as a module, once.

    The module is registered in ``sys.modules`` as ``_intopt_<stem>``, so a
    second call for the same file name reuses it. A file that fails to load is
    not kept, so a later call tries again.

    Args:
        file_path (str): Path of the ``.py`` file, absolute or relative.

    Returns:
        types.ModuleType: The loaded module.

    Raises:
        ImportSpecError: If the file does not exist or raises while loading.
    """
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
