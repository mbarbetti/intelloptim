"""Canonical serialization and hashing for content-addressed studies.

Study identity uses an opaque per-entity ``study_key`` (a stable UUID4 hex
string) rather than the raw API bearer token. The server resolves the token
to the appropriate ``study_key`` (user-scoped or group-scoped) before
calling :func:`study_identity`; the SDK never sees or stores the key.

This module must produce **byte-identical** output across Python versions,
OS, and CPU architecture for equivalent inputs. The tests in
``tests/unit/core/test_hashing.py`` are the authoritative spec; do not
change the canonicalizer without updating them.

Rules:
    * Dict keys are sorted lexicographically at every depth.
    * Sequences keep their declared order (parameter order is meaningful).
    * Floats round-trip via :func:`repr` (shortest round-trippable
      representation; identical across CPython versions).
    * No whitespace, no trailing newlines, ASCII-only output.
    * NaN and infinities are rejected; they have no useful study meaning.
"""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING, Any, Mapping, Sequence, Tuple

if TYPE_CHECKING:
    from intelloptim._internal.schemas import StudyResolveRequest

_HASH_PREFIX = "suid:"


def canonicalize(value: Any) -> str:
    """Return a byte-stable canonical JSON-like string for *value*.

    Args:
        value: A composition of ``None``, ``bool``, ``int``, ``float``,
            ``str``, ``list``/``tuple``, and ``dict`` with string keys.
            Pydantic models should be converted with ``model_dump()``
            before being passed in.

    Returns:
        The canonical string representation. Always ASCII; no whitespace.

    Raises:
        TypeError: If *value* contains a type that is not in the
            supported set (e.g. ``set``, ``bytes``, custom objects).
        ValueError: If a ``float`` is ``NaN`` or infinite, or a ``dict``
            contains a non-string key.
    """
    return _emit(value)


def study_id(
    study_key: str,
    name: str,
    parameters: Sequence[Mapping[str, Any]],
    sampler: Mapping[str, Any],
    pruner: Mapping[str, Any],
    direction: str,
) -> str:
    """Compute the content-addressed study identifier.

    Two callers that submit equivalent specs with the same ``study_key``
    produce the same id; any difference (parameter range, sampler kwarg,
    pruner config, or key) produces a different id by construction.

    The ``study_key`` is an opaque stable UUID4 assigned per-user or
    per-group by the server. It isolates studies across users/groups
    without embedding the raw bearer token in the hash.

    Args:
        study_key: Opaque stable key (UUID4 hex) for the calling entity.
            Supplied by the server; never the raw API token.
        name: Human-readable study label.
        parameters: Ordered sequence of parameter dicts; each must include
            ``kind`` plus type-specific fields.
        sampler: Sampler spec dict (``kind`` + kwargs).
        pruner: Pruner spec dict (``kind`` + kwargs).
        direction: ``"minimize"`` or ``"maximize"`` (or a list, for
            multi-objective).

    Returns:
        ``"suid:<hex>"`` — 69 characters total.
    """
    payload = {
        "study_key": study_key,
        "name": name,
        "parameters": list(parameters),
        "sampler": dict(sampler),
        "pruner": dict(pruner),
        "direction": direction,
    }
    return _digest(canonicalize(payload))


def config_digest(
    parameters: Sequence[Mapping[str, Any]],
    sampler: Mapping[str, Any],
    pruner: Mapping[str, Any],
    direction: str,
) -> str:
    """Compute the per-request config digest (no token mixed in).

    The server compares this against the stored study config on every
    ``ask`` / ``tell`` / ``enqueue`` and returns ``409 Conflict`` on
    mismatch. Defence-in-depth against silent drift between two clients
    that nominally share a study.
    """
    payload = {
        "parameters": list(parameters),
        "sampler": dict(sampler),
        "pruner": dict(pruner),
        "direction": direction,
    }
    return _digest(canonicalize(payload))


def study_identity(study_key: str, request: "StudyResolveRequest") -> Tuple[str, str]:
    """Compute ``(study_id, config_digest)`` from a resolve request.

    This is the canonical entry point used **server-side** to derive
    identifiers. The spec is always serialised through pydantic's
    ``model_dump(mode="json")`` so that defaults like ``log=False`` are
    present; constructing the payload from raw dicts produces a different
    hash and is a bug.

    Args:
        study_key: Opaque stable key for the calling entity (user or group).
            Resolved server-side from the bearer token — never the raw token.
        request: A :class:`~intelloptim._internal.schemas.StudyResolveRequest`
            (or any object with the same shape).

    Returns:
        ``(study_id, config_digest)``. ``study_id`` mixes in the
        ``study_key``; ``config_digest`` does not.
    """
    parameters = [p.model_dump(mode="json") for p in request.parameters]
    sampler = request.sampler.model_dump(mode="json")
    pruner = request.pruner.model_dump(mode="json")
    direction = request.direction
    sid = study_id(
        study_key=study_key,
        name=request.name,
        parameters=parameters,
        sampler=sampler,
        pruner=pruner,
        direction=direction,
    )
    digest = config_digest(
        parameters=parameters,
        sampler=sampler,
        pruner=pruner,
        direction=direction,
    )
    return sid, digest


def _digest(canonical: str) -> str:
    h = hashlib.sha256(canonical.encode("ascii")).hexdigest()
    return f"{_HASH_PREFIX}{h}"


def _emit(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(
                "NaN and infinite floats are not allowed in canonical form"
            )
        return repr(value)
    if isinstance(value, str):
        return _emit_string(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_emit(v) for v in value) + "]"
    if isinstance(value, Mapping):
        items: list[tuple[str, Any]] = []
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError(
                    f"dict keys must be str, got {type(k).__name__}: {k!r}"
                )
            items.append((k, v))
        items.sort(key=lambda kv: kv[0])
        body = ",".join(f"{_emit_string(k)}:{_emit(v)}" for k, v in items)
        return "{" + body + "}"
    raise TypeError(f"unsupported type for canonicalization: {type(value).__name__}")


def _emit_string(value: str) -> str:
    out = ['"']
    for ch in value:
        code = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\b":
            out.append("\\b")
        elif ch == "\f":
            out.append("\\f")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif code < 0x20 or code > 0x7E:
            out.append(f"\\u{code:04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)
