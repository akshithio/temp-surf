from __future__ import annotations

import numpy as np

from evals import evals as EV


def test_filter_conditions_by_axes() -> None:
    conditions = [
        ("clean", "none", 0.0),
        ("sensor_off_s2", "s2", 0.0),
        ("temporal_drop_50", "none", 0.5),
        ("s2_off_tdrop50", "s2", 0.5),
    ]

    sensorial = EV.filter_conditions_by_axes(conditions, ["sensorial"])
    both = EV.filter_conditions_by_axes(conditions, ["sensorial", "temporal"])

    assert [c[0] for c in sensorial] == ["clean", "sensor_off_s2"]
    assert [c[0] for c in both] == ["clean", "sensor_off_s2", "temporal_drop_50", "s2_off_tdrop50"]


def test_make_strict_holdout_splits() -> None:
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    groups = np.array(["src1", "src1", "src2", "src2", "hold", "hold", "src3", "src3"], dtype=object)

    train, val, test, train_val = EV.make_strict_holdout_splits(y, groups, "hold", seed=0)

    assert set(test.tolist()) == {4, 5}
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)
    assert set(train_val.tolist()) == {0, 1, 2, 3, 6, 7}
    assert np.all(groups[train] != "hold")
    assert np.all(groups[val] != "hold")


def test_subset_indices() -> None:
    y = np.array([0, 1, 0, 1, 0])

    tiny = EV.subset_indices(y, budget=0.01, seed=4, stratify=True)
    full = EV.subset_indices(y, budget=1.0, seed=4, stratify=True)

    assert len(tiny) == 2
    np.testing.assert_array_equal(full, np.arange(len(y)))


def test_expected_calibration_error() -> None:
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.0, 0.0, 1.0, 1.0])

    assert EV.expected_calibration_error(y, prob, n_bins=2) == 0.0


def test_regression_source_budget_rows_include_budget_metadata() -> None:
    rng = np.random.default_rng(2)
    x_train = rng.normal(size=(12, 4)).astype(np.float32)
    x_test = rng.normal(size=(5, 4)).astype(np.float32)
    y_train = np.linspace(0, 11, 12, dtype=np.float32)
    y_test = np.linspace(2, 6, 5, dtype=np.float32)

    rows: list[dict] = []
    EV.run_probes_regression(
        rows,
        x_train,
        x_test,
        y_train,
        y_test,
        seed=0,
        budgets=[0.25, 1.0],
        meta={"task": "unit", "method": "erm"},
    )

    assert [r["budget_type"] for r in rows] == ["source", "source"]
    assert [r["label_budget"] for r in rows] == [0.25, 1.0]
    assert [r["n_train_sub"] for r in rows] == [3, 12]
    assert all("rmse" in r and "mae" in r and "r2" in r for r in rows)


def test_regression_target_budget_moves_target_labels_out_of_test_set() -> None:
    rng = np.random.default_rng(3)
    x_source = rng.normal(size=(10, 3)).astype(np.float32)
    x_target = rng.normal(size=(6, 3)).astype(np.float32)
    y_source = np.linspace(0, 9, 10, dtype=np.float32)
    y_target = np.linspace(10, 15, 6, dtype=np.float32)

    rows: list[dict] = []
    EV.run_probes_regression_target(
        rows,
        x_source,
        x_target,
        y_source,
        y_target,
        seed=0,
        budgets=[0, 2],
        meta={"task": "unit", "method": "erm"},
    )

    assert [r["budget_type"] for r in rows] == ["target", "target"]
    assert [r["label_budget"] for r in rows] == [0, 2]
    assert [r["n_train_sub"] for r in rows] == [10, 12]
    assert [r["n_test"] for r in rows] == [6, 4]
