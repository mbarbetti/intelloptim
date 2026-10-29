"""Canonical serialisation and hashing that give studies their identity.

A study is content-addressed: its id is the SHA-256 of a canonical string built
from the owner's key, the study name and the full study configuration. Equal
inputs give equal ids in every process, so independent clients that submit the
same configuration land on the same study.

The owner key (``study_key``) is an opaque, stable identifier of a user or a
group, never the API token itself; the server resolves it from the token. The
SDK only computes :func:`config_digest`, the same hash without key and name,
which it sends with every request so the server can detect a configuration that
differs between clients.

The output is identical across Python versions, operating systems and CPU
architectures.

Canonical form:

* Dict keys are sorted lexicographically at every depth.
* Sequences keep their declared order (parameter order is meaningful).
* Floats are written with :func:`repr` (shortest round-trippable form), so
  ``1`` and ``1.0`` hash differently.
* No whitespace and no trailing newline; the output is ASCII only.
* ``NaN`` and infinities are rejected: they have no useful meaning in a study.
"""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING, Any, Mapping, Sequence, Tuple

if TYPE_CHECKING:
    from intelloptim._internal.schemas import StudyResolveRequest

_HASH_PREFIX = "suid:"


def canonicalize(value: Any) -> str:
    """Serialise *value* to its canonical string.

    The result is valid JSON: dict keys are sorted, non-ASCII characters are
    escaped as ``\\uXXXX`` and lists and tuples give the same output.

    Args:
        value (Any): A composition of ``None``, ``bool``, ``int``, ``float``,
            ``str``, ``list``/``tuple`` and mappings with ``str`` keys. Convert
            pydantic models with ``model_dump(mode="json")`` first.

    Returns:
        str: The canonical string: ASCII only, no whitespace.

    Raises:
        TypeError: If *value* contains an unsupported type, for example a
            ``set``, ``bytes`` or a custom object.
        ValueError: If a ``float`` is ``NaN`` or infinite, or a mapping has a
            non-``str`` key.
    """
    return _emit(value)


def study_id(
    study_key: str,
    study_name: str,
    direction: str,
    parameters: Sequence[Mapping[str, Any]],
    sampler_spec: Mapping[str, Any],
    pruner_spec: Mapping[str, Any],
) -> str:
    """Compute the content-addressed id of a study.

    Callers that submit equivalent specifications with the same ``study_key``
    get the same id. Any difference in the name, direction, parameters, sampler
    or pruner, or a different key, gives a different id. The key keeps the
    studies of different users and groups apart without putting the API token in
    the hash. :func:`study_identity` computes this id, and the digest, directly
    from a resolve request.

    Args:
        study_key (str): Opaque key (UUID4 hex) of the calling user or group,
            resolved by the server; never the raw API token.
        study_name (str): Human-readable study name.
        direction (str): ``"minimize"`` or ``"maximize"``.
        parameters (Sequence[Mapping[str, Any]]): Declared search space as
            parameter dicts in declared order, each with a ``kind`` field.
        sampler_spec (Mapping[str, Any]): Sampler spec dict (``kind`` plus its
            settings).
        pruner_spec (Mapping[str, Any]): Pruner spec dict (``kind`` plus its
            settings).

    Returns:
        str: ``"suid:"`` followed by 64 hex characters (69 in total).
    """
    # Key names and values are hashed: renaming one changes every study id.
    payload = {
        "study_key": study_key,
        "study_name": study_name,
        "direction": direction,
        "parameters": list(parameters),
        "sampler_spec": dict(sampler_spec),
        "pruner_spec": dict(pruner_spec),
    }
    return _digest(canonicalize(payload))


def config_digest(
    direction: str,
    parameters: Sequence[Mapping[str, Any]],
    sampler_spec: Mapping[str, Any],
    pruner_spec: Mapping[str, Any],
) -> str:
    """Compute the digest of a study configuration, without owner or name.

    Clients send it with every study request. The server compares it with the
    digest of the stored configuration and answers ``409 Conflict`` on a
    mismatch, which catches two clients that think they share a study but hold
    different configurations. It hashes the same fields as :func:`study_id`
    minus ``study_key`` and ``study_name``, so it is the same for every owner.

    Args:
        direction (str): ``"minimize"`` or ``"maximize"``.
        parameters (Sequence[Mapping[str, Any]]): Declared search space as
            parameter dicts in declared order.
        sampler_spec (Mapping[str, Any]): Sampler spec dict (``kind`` plus its
            settings).
        pruner_spec (Mapping[str, Any]): Pruner spec dict (``kind`` plus its
            settings).

    Returns:
        str: ``"suid:"`` followed by 64 hex characters.
    """
    # Key names and values are hashed: renaming one changes every digest.
    payload = {
        "direction": direction,
        "parameters": list(parameters),
        "sampler_spec": dict(sampler_spec),
        "pruner_spec": dict(pruner_spec),
    }
    return _digest(canonicalize(payload))


def study_identity(study_key: str, request: "StudyResolveRequest") -> Tuple[str, str]:
    """Compute ``(study_id, config_digest)`` from a resolve request.

    The specs are serialised with ``model_dump(mode="json")``, so default values
    such as ``log=False`` are part of the hash and a spec that relies on a
    default gets the same id as one that sets it explicitly.

    Args:
        study_key (str): Opaque key of the calling user or group, resolved by
            the server from the bearer token; never the raw token.
        request (StudyResolveRequest): The resolve request, or any object with
            the same fields.

    Returns:
        Tuple[str, str]: ``(study_id, config_digest)``. Only the id depends on
        ``study_key`` and the study name.
    """
    parameters = [p.model_dump(mode="json") for p in request.parameters]
    direction = request.direction
    sampler_spec = request.sampler_spec.model_dump(mode="json")
    pruner_spec = request.pruner_spec.model_dump(mode="json")
    suid = study_id(
        study_key=study_key,
        study_name=request.study_name,
        direction=direction,
        parameters=parameters,
        sampler_spec=sampler_spec,
        pruner_spec=pruner_spec,
    )
    digest = config_digest(
        direction=direction,
        parameters=parameters,
        sampler_spec=sampler_spec,
        pruner_spec=pruner_spec,
    )
    return suid, digest


def _digest(canonical: str) -> str:
    """Return ``"suid:"`` plus the SHA-256 hex digest of a canonical string."""
    h = hashlib.sha256(canonical.encode("ascii")).hexdigest()
    return f"{_HASH_PREFIX}{h}"


def _emit(value: Any) -> str:
    """Serialise one value recursively; see :func:`canonicalize`."""
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
    """Quote *value* and escape it to printable ASCII.

    Quote, backslash and the usual control characters use their short escapes;
    every other character outside ``0x20``-``0x7E`` becomes ``\\uXXXX``, and
    characters above ``U+FFFF`` a surrogate pair, as in JSON.
    """
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
        elif code > 0xFFFF:
            offset = code - 0x10000
            high = 0xD800 + (offset >> 10)
            low = 0xDC00 + (offset & 0x3FF)
            out.append(f"\\u{high:04x}\\u{low:04x}")
        elif code < 0x20 or code > 0x7E:
            out.append(f"\\u{code:04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


__all__ = ["config_digest", "study_identity"]
