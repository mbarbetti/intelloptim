"""Hashing and canonicalization — the federation contract.

These fixtures are intentionally explicit and over-specified. The
canonical output is a public ABI: changing it breaks federation silently
across deployed clients. Update tests *first* if a deliberate change is
required, and treat any mismatch with these fixtures as a design red flag.
"""

from __future__ import annotations

import json
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

    def test_non_bmp_character_is_a_surrogate_pair(self) -> None:
        assert canonicalize("\U0001f600") == '"\\ud83d\\ude00"'

    @pytest.mark.parametrize(
        "text",
        ["café", "naïve 日本語", "\U0001f600 run", "a\U00010000b", "\x7f\x00\x1f", ""],
    )
    def test_output_is_valid_json_that_round_trips(self, text: str) -> None:
        out = canonicalize({"k": text, text: [text]})
        assert out.isascii()
        assert json.loads(out) == {"k": text, text: [text]}

    def test_empty_containers(self) -> None:
        assert canonicalize([]) == "[]"
        assert canonicalize({}) == "{}"


class TestStudyId:
    _KEY_A = "aabbccddeeff00112233445566778899"
    _KEY_B = "99887766554433221100ffeeddccbbaa"

    def _spec(self, study_key: str = _KEY_A):
        return {
            "study_key": study_key,
            "study_name": "demo",
            "direction": "minimize",
            "parameters": [
                {"kind": "float", "name": "x", "low": -1.0, "high": 1.0},
                {"kind": "int", "name": "n", "low": 1, "high": 10},
            ],
            "sampler_spec": {"kind": "tpe", "seed": 42},
            "pruner_spec": {"kind": "nop"},
        }

    def test_id_is_byte_stable(self) -> None:
        """Golden value: any change to the hashed payload breaks every study id.

        If this fails on purpose (a versioned change to the payload), update the
        value together with the server and the web app.
        """
        assert study_id(**self._spec()) == (
            "suid:d1514b90f4384c7c0825c020933139d6c3d210d92c005b75c87624f3ca2a2553"
        )

    def test_deterministic(self) -> None:
        s = self._spec()
        assert study_id(**s) == study_id(**s)

    def test_prefix(self) -> None:
        sid = study_id(**self._spec())
        assert sid.startswith("suid:")
        assert len(sid) == len("suid:") + 64

    def test_same_study_key_same_id(self) -> None:
        """Two different callers with the same study_key produce the same id."""
        a = study_id(**self._spec(self._KEY_A))
        b = study_id(**self._spec(self._KEY_A))
        assert a == b

    def test_different_study_key_different_id(self) -> None:
        """Different study_keys (different users/groups) produce different ids."""
        a = study_id(**self._spec(self._KEY_A))
        b = study_id(**self._spec(self._KEY_B))
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
        s["sampler_spec"] = {"kind": "tpe", "seed": 43}
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
            "pruner_spec": s_natural["pruner_spec"],
            "sampler_spec": s_natural["sampler_spec"],
            "parameters": s_natural["parameters"],
            "direction": s_natural["direction"],
            "study_name": s_natural["study_name"],
            "study_key": s_natural["study_key"],
        }
        assert study_id(**s_natural) == study_id(**s_reordered)


class TestConfigDigest:
    def test_excludes_token(self) -> None:
        params = [{"kind": "float", "name": "x", "low": 0.0, "high": 1.0}]
        sampler = {"kind": "tpe", "seed": 1}
        pruner = {"kind": "nop"}
        direction = "minimize"
        a = config_digest(direction, params, sampler, pruner)
        b = config_digest(direction, params, sampler, pruner)
        assert a == b
        assert a.startswith("suid:")

    def test_digest_is_byte_stable(self) -> None:
        """Golden value, same rule as ``TestStudyId.test_id_is_byte_stable``."""
        digest = config_digest(
            "minimize",
            [
                {"kind": "float", "name": "x", "low": -1.0, "high": 1.0},
                {"kind": "int", "name": "n", "low": 1, "high": 10},
            ],
            {"kind": "tpe", "seed": 42},
            {"kind": "nop"},
        )
        assert digest == (
            "suid:c42d36e1e61c57e38ad306145a18f956eec8059efed76aa13b1b9fc016737db5"
        )

    def test_differs_from_study_id(self) -> None:
        params = [{"kind": "float", "name": "x", "low": 0.0, "high": 1.0}]
        sampler = {"kind": "tpe"}
        pruner = {"kind": "nop"}
        key = "aabbccddeeff00112233445566778899"
        sid = study_id(key, "demo", "minimize", params, sampler, pruner)
        digest = config_digest("minimize", params, sampler, pruner)
        assert sid != digest


class TestStudyIdentityFromRequest:
    """``study_identity`` must hash exactly what ``study_id`` hashes, whatever
    the field names of the wire model are called."""

    _KEY = "aabbccddeeff00112233445566778899"

    def test_matches_the_canonical_study_id(self) -> None:
        from intelloptim._internal.hashing import config_digest, study_identity
        from intelloptim._internal.schemas import StudyResolveRequest
        from intelloptim.core.parameters import FloatSpec
        from intelloptim.core.pruners import NopPrunerSpec
        from intelloptim.core.samplers import TPESamplerSpec

        request = StudyResolveRequest(
            study_name="demo",
            direction="minimize",
            parameters=[FloatSpec("x", -1.0, 1.0)],
            sampler_spec=TPESamplerSpec(seed=42),
            pruner_spec=NopPrunerSpec(),
        )
        parameters = [p.model_dump(mode="json") for p in request.parameters]
        sampler = request.sampler_spec.model_dump(mode="json")
        pruner = request.pruner_spec.model_dump(mode="json")

        sid, digest = study_identity(self._KEY, request)

        assert sid == study_id(
            self._KEY, "demo", "minimize", parameters, sampler, pruner
        )
        assert digest == config_digest("minimize", parameters, sampler, pruner)
