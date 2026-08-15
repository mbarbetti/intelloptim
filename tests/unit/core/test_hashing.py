"""Hashing and canonicalization — the federation contract.

These fixtures are intentionally explicit and over-specified. The
canonical output is a public ABI: changing it breaks federation silently
across deployed clients. Update tests *first* if a deliberate change is
required, and treat any mismatch with these fixtures as a design red flag.
"""

from __future__ import annotations

import math

import pytest

from intelloptim._internal.hashing import (
    canonicalize,
    config_digest,
    study_id,
)


class TestCanonicalize:
    def test_primitives(self) -> None:
        assert canonicalize(None) == "null"
        assert canonicalize(True) == "true"
        assert canonicalize(False) == "false"
        assert canonicalize(0) == "0"
        assert canonicalize(-42) == "-42"
        assert canonicalize("") == '""'

    def test_dict_keys_are_sorted(self) -> None:
        a = canonicalize({"z": 1, "a": 2, "m": 3})
        b = canonicalize({"a": 2, "m": 3, "z": 1})
        assert a == b == '{"a":2,"m":3,"z":1}'

    def test_list_order_is_preserved(self) -> None:
        a = canonicalize([1, 2, 3])
        b = canonicalize([3, 2, 1])
        assert a == "[1,2,3]"
        assert b == "[3,2,1]"
        assert a != b

    def test_tuple_treated_as_list(self) -> None:
        assert canonicalize((1, 2, 3)) == canonicalize([1, 2, 3])

    def test_nested(self) -> None:
        out = canonicalize({"x": [1, {"k": "v"}], "y": None})
        assert out == '{"x":[1,{"k":"v"}],"y":null}'

    def test_float_uses_repr(self) -> None:
        assert canonicalize(1.0) == "1.0"
        assert canonicalize(0.1) == "0.1"
        assert canonicalize(-1.5) == "-1.5"

    def test_nan_rejected(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            canonicalize(float("nan"))

    def test_inf_rejected(self) -> None:
        with pytest.raises(ValueError, match="infinite"):
            canonicalize(float("inf"))
        with pytest.raises(ValueError, match="infinite"):
            canonicalize(-math.inf)

    def test_unsupported_type(self) -> None:
        with pytest.raises(TypeError, match="set"):
            canonicalize({1, 2, 3})
        with pytest.raises(TypeError, match="bytes"):
            canonicalize(b"hello")

    def test_non_string_dict_keys_rejected(self) -> None:
        with pytest.raises(ValueError, match="dict keys must be str"):
            canonicalize({1: "a"})

    def test_string_escaping(self) -> None:
        assert canonicalize('a"b') == '"a\\"b"'
        assert canonicalize("a\\b") == '"a\\\\b"'
        assert canonicalize("a\nb") == '"a\\nb"'
        assert canonicalize("a\tb") == '"a\\tb"'

    def test_non_ascii_string_escaped(self) -> None:
        out = canonicalize("café")
        assert out.isascii()
        assert "\\u00e9" in out

    def test_empty_containers(self) -> None:
        assert canonicalize([]) == "[]"
        assert canonicalize({}) == "{}"


class TestStudyId:
    def _spec(self):
        return {
            "token": "tok_abc",
            "name": "demo",
            "parameters": [
                {"kind": "float", "name": "x", "low": -1.0, "high": 1.0},
                {"kind": "int", "name": "n", "low": 1, "high": 10},
            ],
            "sampler": {"kind": "tpe", "seed": 42},
            "pruner": {"kind": "nop"},
            "direction": "minimize",
        }

    def test_deterministic(self) -> None:
        s = self._spec()
        assert study_id(**s) == study_id(**s)

    def test_prefix(self) -> None:
        sid = study_id(**self._spec())
        assert sid.startswith("suid:")
        assert len(sid) == len("suid:") + 64

    def test_token_change_changes_id(self) -> None:
        a = study_id(**self._spec())
        s = self._spec()
        s["token"] = "tok_other"
        b = study_id(**s)
        assert a != b

    def test_param_range_change_changes_id(self) -> None:
        a = study_id(**self._spec())
        s = self._spec()
        s["parameters"][0] = {
            **s["parameters"][0],
            "high": 2.0,
        }
        b = study_id(**s)
        assert a != b

    def test_sampler_kwarg_change_changes_id(self) -> None:
        a = study_id(**self._spec())
        s = self._spec()
        s["sampler"] = {"kind": "tpe", "seed": 43}
        b = study_id(**s)
        assert a != b

    def test_param_order_matters(self) -> None:
        a = study_id(**self._spec())
        s = self._spec()
        s["parameters"] = list(reversed(s["parameters"]))
        b = study_id(**s)
        assert a != b

    def test_dict_key_order_does_not_matter(self) -> None:
        s_natural = self._spec()
        s_reordered = {
            "direction": s_natural["direction"],
            "pruner": s_natural["pruner"],
            "sampler": s_natural["sampler"],
            "parameters": s_natural["parameters"],
            "name": s_natural["name"],
            "token": s_natural["token"],
        }
        assert study_id(**s_natural) == study_id(**s_reordered)


class TestConfigDigest:
    def test_excludes_token(self) -> None:
        params = [{"kind": "float", "name": "x", "low": 0.0, "high": 1.0}]
        sampler = {"kind": "tpe", "seed": 1}
        pruner = {"kind": "nop"}
        direction = "minimize"
        a = config_digest(params, sampler, pruner, direction)
        b = config_digest(params, sampler, pruner, direction)
        assert a == b
        assert a.startswith("suid:")

    def test_differs_from_study_id(self) -> None:
        params = [{"kind": "float", "name": "x", "low": 0.0, "high": 1.0}]
        sampler = {"kind": "tpe"}
        pruner = {"kind": "nop"}
        sid = study_id("tok", "demo", params, sampler, pruner, "minimize")
        digest = config_digest(params, sampler, pruner, "minimize")
        assert sid != digest
