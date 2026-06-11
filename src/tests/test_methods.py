from __future__ import annotations

import numpy as np

from methods.dfr import Dfr
from methods.grit import Grit, projection_from_diffs, view_drop_diffs
from methods.tent import Tent


def test_dfr_subset_balances_strata() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    groups = np.array(["a", "a", "b", "a", "b", "b"], dtype=object)

    sub = Dfr(seed=0).subset_indices(y, groups, budget=4 / 6, seed=0)
    strata = {(int(y[i]), str(groups[i])) for i in sub}

    assert len(sub) == 4
    assert strata == {(0, "a"), (0, "b"), (1, "a"), (1, "b")}


def test_method_variants_match_task_families() -> None:
    from methods import dfr, grit, tent

    assert dfr.variants("regression") == {}
    assert tent.variants("regression") == {}
    assert "tent" in tent.variants("binary")
    assert "dfr" in dfr.variants("multiclass")
    assert "grit_viewdrop_r4" in grit.variants("regression")


def test_projection_from_diffs() -> None:
    diffs = np.array([[2.0, 0.0], [-3.0, 0.0], [4.0, 0.0]], dtype=np.float32)

    proj = projection_from_diffs(diffs, rank=1)

    transformed = np.array([[7.0, 5.0]], dtype=np.float32) @ proj
    np.testing.assert_allclose(transformed, np.array([[0.0, 5.0]], dtype=np.float32), atol=1e-6)


def test_grit_conditional_projection() -> None:
    x = np.array(
        [
            [2.0, 0.0],
            [-2.0, 0.0],
            [2.0, 1.0],
            [-2.0, 1.0],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 1, 1])
    groups = np.array(["a", "b", "a", "b"], dtype=object)

    transform = Grit(rank=1, matching="conditional", max_pairs=8, seed=0).fit(x, y, groups)
    out = transform.transform(np.array([[5.0, 3.0]], dtype=np.float32))

    assert abs(out[0, 0]) < 1e-5
    assert abs(out[0, 1] - 3.0) < 1e-5


def test_view_drop_diffs_requires_aligned_shapes() -> None:
    x = np.zeros((3, 2), dtype=np.float32)
    x_bad = np.zeros((4, 2), dtype=np.float32)

    try:
        view_drop_diffs(x, x_bad, max_pairs=8, seed=0)
    except ValueError as exc:
        assert "same samples" in str(exc)
    else:
        raise AssertionError("view_drop_diffs should reject shape mismatch")


def test_tent_records_target_features() -> None:
    class FakeProbe:
        named_steps = {}

    x_target = np.ones((2, 3), dtype=np.float32)
    x_test = np.arange(6, dtype=np.float32).reshape(2, 3)

    tent = Tent(seed=0).fit(np.zeros((2, 3), dtype=np.float32), x_paired=x_target)
    out = tent.adapt_test_features(FakeProbe(), x_test)

    np.testing.assert_array_equal(tent.x_target_, x_target)
    np.testing.assert_array_equal(out, x_test)
