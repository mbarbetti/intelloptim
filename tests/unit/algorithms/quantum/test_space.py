"""Layout of a mixed search space and the latent-to-parameter transforms."""

from __future__ import annotations

import numpy as np
import pytest
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from intelloptim.algorithms.backend.quantum.commons.space import (
    NumericDim,
    SpaceLayout,
    sigmoid,
)


def _dim(dist) -> NumericDim:
    return NumericDim.from_distribution("p", dist)


class TestKinds:
    def test_float_is_continuous(self) -> None:
        dim = _dim(FloatDistribution(-1.0, 3.0))
        assert not dim.discrete and not dim.is_int

    def test_stepped_float_is_discrete(self) -> None:
        dim = _dim(FloatDistribution(0.0, 1.0, step=0.25))
        assert dim.discrete and dim.n_bins == 5 and not dim.is_int

    @pytest.mark.parametrize(
        "dist, n_bins",
        [
            (IntDistribution(1, 10), 10),
            (IntDistribution(0, 10, step=2), 6),
            (IntDistribution(1, 100, log=True), 100),
            (IntDistribution(5, 5), 1),
        ],
    )
    def test_int_bins(self, dist, n_bins: int) -> None:
        dim = _dim(dist)
        assert dim.discrete and dim.is_int and dim.n_bins == n_bins


class TestBins:
    @pytest.mark.parametrize(
        "dist",
        [
            IntDistribution(1, 10),
            IntDistribution(0, 10, step=2),
            IntDistribution(1, 64, log=True),
            FloatDistribution(0.0, 1.0, step=0.25),
        ],
    )
    def test_edges_partition_the_unit_interval(self, dist) -> None:
        dim = _dim(dist)
        edges = dim.boundary(np.arange(dim.n_bins + 1))
        assert edges[0] == pytest.approx(0.0) and edges[-1] == pytest.approx(1.0)
        assert np.all(np.diff(edges) > 0)

    def test_first_and_last_logit_edges_are_infinite(self) -> None:
        dim = _dim(IntDistribution(1, 5))
        logits = dim.logit_boundary(np.arange(6))
        assert logits[0] == -np.inf and logits[-1] == np.inf
        assert np.all(np.isfinite(logits[1:-1]))

    @pytest.mark.parametrize(
        "dist",
        [
            IntDistribution(1, 10),
            IntDistribution(1, 64, log=True),
            IntDistribution(0, 9, step=3),
        ],
    )
    def test_bin_of_inverts_the_edges(self, dist) -> None:
        dim = _dim(dist)
        mids = (
            dim.boundary(np.arange(dim.n_bins))
            + dim.boundary(np.arange(1, dim.n_bins + 1))
        ) / 2
        assert np.array_equal(dim.bin_of(mids), np.arange(dim.n_bins))
        assert dim.bin_of(np.array([1e-12]))[0] == 0
        assert dim.bin_of(np.array([1 - 1e-12]))[0] == dim.n_bins - 1

    def test_values_follow_the_grid(self) -> None:
        assert [
            _dim(IntDistribution(0, 10, step=2)).value_of_bin(j) for j in range(6)
        ] == [0, 2, 4, 6, 8, 10]
        assert _dim(FloatDistribution(1.0, 2.0, step=0.5)).value_of_bin(2) == 2.0
        assert isinstance(_dim(IntDistribution(1, 5)).value_of_bin(2), int)
        assert [
            _dim(IntDistribution(3, 6, log=True)).value_of_bin(j) for j in range(4)
        ] == [3, 4, 5, 6]

    def test_log_int_bins_are_wider_for_larger_values(self) -> None:
        dim = _dim(IntDistribution(1, 100, log=True))
        widths = np.diff(dim.boundary(np.arange(101)))
        assert widths[0] > widths[-1]


class TestContinuous:
    def test_linear_float_stays_in_range(self) -> None:
        dim = _dim(FloatDistribution(-2.0, 3.0))
        values = [dim.value_of_latent(u) for u in (-50.0, -1.0, 0.0, 1.0, 50.0)]
        assert all(-2.0 <= v <= 3.0 for v in values)
        assert dim.value_of_latent(0.0) == pytest.approx(0.5)

    def test_log_float_is_geometric_at_the_centre(self) -> None:
        dim = _dim(FloatDistribution(1e-4, 1.0, log=True))
        assert dim.value_of_latent(0.0) == pytest.approx(1e-2)
        assert dim.value_of_latent(-80.0) >= 1e-4 and dim.value_of_latent(80.0) <= 1.0

    def test_degenerate_range(self) -> None:
        assert _dim(FloatDistribution(2.0, 2.0)).value_of_latent(3.0) == 2.0


class TestLayout:
    SPACE = {
        "b_cat": CategoricalDistribution(["x", "y", "z"]),
        "a_lr": FloatDistribution(1e-4, 1.0, log=True),
        "c_n": IntDistribution(1, 8),
        "d_flag": CategoricalDistribution([True, False, None, 3]),
        "e_one": CategoricalDistribution(["only"]),
    }

    def test_parameters_are_sorted_within_groups(self) -> None:
        layout = SpaceLayout.from_space(self.SPACE, "register")
        assert [d.name for d in layout.numeric] == ["a_lr", "c_n"]
        assert [c.name for c in layout.categorical] == ["b_cat", "d_flag", "e_one"]
        assert layout.names == tuple(sorted(self.SPACE))

    @pytest.mark.parametrize(
        "n_choices, qubits", [(1, 0), (2, 1), (3, 2), (4, 2), (5, 3), (8, 3), (9, 4)]
    )
    def test_register_qubits_per_categorical(self, n_choices: int, qubits: int) -> None:
        layout = SpaceLayout.from_space(
            {"c": CategoricalDistribution(list(range(n_choices)))}, "register"
        )
        assert layout.categorical[0].n_register_qubits == qubits

    def test_head_encoding_uses_no_register_qubits(self) -> None:
        layout = SpaceLayout.from_space(self.SPACE, "head")
        assert layout.n_register_qubits == 0
        assert SpaceLayout.from_space(self.SPACE, "register").n_register_qubits == 2 + 2

    def test_params_return_original_objects(self) -> None:
        layout = SpaceLayout.from_space(self.SPACE, "register")
        params = layout.params(np.array([0.0, 3.0]), np.array([2, 2, 0]))
        assert (
            params["b_cat"] == "z"
            and params["d_flag"] is None
            and params["e_one"] == "only"
        )
        assert params["c_n"] == 4 and isinstance(params["c_n"], int)
        assert 1e-4 <= params["a_lr"] <= 1.0

    def test_digest_changes_with_the_space_and_the_encoding(self) -> None:
        a = SpaceLayout.from_space(self.SPACE, "register").digest_payload()
        b = SpaceLayout.from_space(self.SPACE, "head").digest_payload()
        changed = dict(self.SPACE, c_n=IntDistribution(1, 9))
        c = SpaceLayout.from_space(changed, "register").digest_payload()
        assert a != b and a != c

    def test_rejects_empty_space_and_unknown_encoding(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            SpaceLayout.from_space({}, "register")
        with pytest.raises(ValueError, match="encoding"):
            SpaceLayout.from_space(self.SPACE, "nope")


def test_sigmoid_is_stable() -> None:
    assert sigmoid(np.array([-1000.0]))[0] == 0.0
    assert sigmoid(np.array([1000.0]))[0] == 1.0
    assert sigmoid(np.array([0.0]))[0] == 0.5
